"""Веб-приложение для масок фруктов: HOG находит маски -> вы отбираете хорошие -> правите -> сохраняете.

Запуск:
    python mask_app.py            # откроет браузер на http://127.0.0.1:8766

Шаги для каждого фото:
  1. Найти маски — HOG ищет круглые фрукты, GrabCut строит маску каждого.
  2. Отбор — ЛКМ по маске: хорошая, ПКМ: плохая. «Оставить только хорошие» стирает остальные.
  3. Редактирование — кисть / ластик / новая маска / удалить маску.
  4. Сохранить — в папку результатов пишутся:
       <фото>_mask.png    — маска целиком (белое — фрукты, чёрное — фон), размер как у фото;
       <фото>_masks.json  — для каждой маски: id, площадь, рамка, контур и список пикселей [x, y].

Незаконченная работа автоматически сохраняется в <папка результатов>/.drafts и
восстанавливается при следующем открытии фото.
"""

import argparse
import base64
import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np

from hog_detector import IMAGE_EXTS, DetectParams, detect, imread, imwrite, label_map, list_images


class State:
    def __init__(self):
        self.src: Path | None = None
        self.out: Path | None = None
        self.lock = threading.Lock()

    def open(self, src: str, out: str):
        src_p = Path(src).expanduser()
        if not src.strip() or not src_p.is_dir():
            raise ValueError(f"Папка с фото не найдена: {src}")
        out_p = Path(out).expanduser() if out.strip() else src_p.parent / f"{src_p.name}_masks"
        out_p.mkdir(parents=True, exist_ok=True)
        self.src, self.out = src_p.resolve(), out_p.resolve()

    def image_path(self, name: str) -> Path:
        if not self.src:
            raise ValueError("Сначала откройте папку с фото")
        path = self.src / Path(name).name
        if not path.is_file():
            raise ValueError(f"Нет файла {name}")
        return path

    def draft_path(self, name: str) -> Path:
        return self.out / ".drafts" / f"{Path(name).stem}.json"

    def result_paths(self, name: str) -> tuple[Path, Path]:
        stem = Path(name).stem
        return self.out / f"{stem}_mask.png", self.out / f"{stem}_masks.json"

    def sources(self) -> list[dict]:
        if not self.src:
            return []
        out = []
        for p in list_images(self.src):
            status = "new"
            if self.result_paths(p.name)[1].exists():
                status = "saved"
            elif self.draft_path(p.name).exists():
                status = "draft"
            out.append({"name": p.name, "status": status})
        return out


def encode_labels(labels: np.ndarray) -> str:
    return base64.b64encode(labels.astype("<u2").tobytes()).decode()


def decode_labels(data: str, width: int, height: int) -> np.ndarray:
    arr = np.frombuffer(base64.b64decode(data), dtype="<u2")
    if arr.size != width * height:
        raise ValueError("размер маски не совпадает с размером фото")
    return arr.reshape(height, width)


def write_results(state: State, name: str, labels: np.ndarray) -> dict:
    """Пишет <фото>_mask.png и <фото>_masks.json. Маски перенумеровываются 1..N."""
    h, w = labels.shape
    masks = []
    for new_id, old_id in enumerate((i for i in np.unique(labels) if i != 0), start=1):
        m = labels == old_id
        ys, xs = np.nonzero(m)
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        masks.append({
            "id": new_id,
            "area": int(xs.size),
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
            "contours": [c.reshape(-1, 2).tolist() for c in contours if len(c) >= 3],
            "pixels": np.stack([xs, ys], axis=1).tolist(),
        })
    mask_png, mask_json = state.result_paths(name)
    imwrite(mask_png, np.where(labels > 0, 255, 0).astype(np.uint8))
    doc = {"image": Path(name).name, "width": w, "height": h, "count": len(masks),
           "pixel_format": "[x, y]", "masks": masks}
    mask_json.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return {"count": len(masks), "mask": mask_png.name, "json": mask_json.name,
            "json_kb": round(mask_json.stat().st_size / 1024)}


def load_saved(state: State, name: str) -> dict | None:
    """Восстанавливает маски из сохранённого <фото>_masks.json, чтобы его можно было доредактировать."""
    _, mask_json = state.result_paths(name)
    if not mask_json.exists():
        return None
    doc = json.loads(mask_json.read_text(encoding="utf-8"))
    labels = np.zeros((doc["height"], doc["width"]), np.uint16)
    masks = []
    for m in doc["masks"]:
        px = np.asarray(m["pixels"], dtype=np.int64).reshape(-1, 2)
        if px.size:
            labels[px[:, 1], px[:, 0]] = m["id"]
        masks.append({"id": m["id"], "score": None, "status": "good"})
    return {"width": doc["width"], "height": doc["height"], "labels": encode_labels(labels),
            "masks": masks, "step": 4}


def browse_folder(initial: str) -> str:
    """Системный диалог выбора папки (в отдельном процессе, чтобы не мешать серверу)."""
    code = (
        "import sys, tkinter as tk\n"
        "from tkinter import filedialog\n"
        "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
        "print(filedialog.askdirectory(initialdir=sys.argv[1] or None, mustexist=False) or '')\n"
    )
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    res = subprocess.run([sys.executable, "-c", code, initial], capture_output=True, env=env, timeout=600)
    return res.stdout.decode("utf-8", "replace").strip()


def params_from(data: dict) -> tuple[DetectParams, str]:
    p = DetectParams(
        min_size=max(8, int(data.get("min_size", DetectParams.min_size))),
        max_size=max(0, int(data.get("max_size", DetectParams.max_size))),
        threshold=float(data.get("threshold", DetectParams.threshold)),
        nms_iou=float(data.get("nms_iou", DetectParams.nms_iou)),
    )
    mode = data.get("mode", "grabcut")
    return p, mode if mode in ("grabcut", "ellipse", "box") else "grabcut"


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def do_GET(self):
            url = urlparse(self.path)
            path, query = url.path, parse_qs(url.query)
            try:
                if path == "/":
                    self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                elif path == "/api/state":
                    self._json({
                        "src": str(state.src or ""), "out": str(state.out or ""), "sources": state.sources(),
                        "defaults": {"min_size": DetectParams.min_size, "max_size": DetectParams.max_size,
                                     "threshold": DetectParams.threshold, "nms_iou": DetectParams.nms_iou},
                    })
                elif path == "/api/load":
                    name = query.get("name", [""])[0]
                    state.image_path(name)
                    draft = state.draft_path(name)
                    if draft.exists():
                        self._json(json.loads(draft.read_text(encoding="utf-8")))
                    else:
                        self._json(load_saved(state, name) or {})
                elif path.startswith("/src/"):
                    p = state.image_path(unquote(path[5:]))
                    self._send(200, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "application/octet-stream")
                elif path.startswith("/out/"):
                    p = (state.out / Path(unquote(path[5:])).name) if state.out else None
                    if not p or p.suffix.lower() not in (".png", ".json") or not p.is_file():
                        return self._send(404, b"not found", "text/plain")
                    extra = {"Content-Disposition": f'attachment; filename="{p.name}"'} if "download" in query else None
                    self._send(200, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "application/octet-stream", extra)
                else:
                    self._send(404, b"not found", "text/plain")
            except Exception as e:
                self._json({"error": str(e)}, 400)

        def do_POST(self):
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(length) or b"{}")
            try:
                handler = {"/api/browse": self.browse, "/api/open": self.open, "/api/detect": self.detect,
                           "/api/draft": self.draft, "/api/save": self.save}.get(path)
                if handler is None:
                    return self._json({"error": "not found"}, 404)
                handler(data)
            except Exception as e:
                self._json({"error": str(e)}, 400)

        def browse(self, data):
            self._json({"path": browse_folder(data.get("initial", ""))})

        def open(self, data):
            state.open(data.get("src", ""), data.get("out", ""))
            self._json({"src": str(state.src), "out": str(state.out), "sources": state.sources()})

        def detect(self, data):
            name = data["name"]
            img = imread(state.image_path(name))
            params, mode = params_from(data)
            t = time.time()
            dets = detect(img, params)
            labels, info = label_map(img, dets, mode)
            for m in info:
                m["status"] = None
            result = {"width": img.shape[1], "height": img.shape[0], "labels": encode_labels(labels),
                      "masks": info, "step": 2, "seconds": round(time.time() - t, 1)}
            self._write_draft(name, result)
            self._json(result)

        def _write_draft(self, name: str, doc: dict):
            p = state.draft_path(name)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            tmp.replace(p)

        def draft(self, data):
            img_path = state.image_path(data["name"])
            decode_labels(data["labels"], int(data["width"]), int(data["height"]))  # проверка размера
            self._write_draft(img_path.name, {k: data[k] for k in ("width", "height", "labels", "masks", "step")})
            self._json({"ok": True})

        def save(self, data):
            name = state.image_path(data["name"]).name
            labels = decode_labels(data["labels"], int(data["width"]), int(data["height"]))
            res = write_results(state, name, labels)
            self._write_draft(name, {k: data[k] for k in ("width", "height", "labels", "masks")} | {"step": 4})
            self._json(res)

    return Handler


PAGE = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Маски фруктов</title>
<style>
  :root { --bg:#14161a; --panel:#1d2026; --line:#2c3038; --text:#e6e8eb; --muted:#8a919c;
          --good:#3ddc5a; --bad:#e5484d; --accent:#5b9cf5; --warn:#ffb74d; }
  * { box-sizing:border-box; }
  [hidden] { display:none !important; }
  body { margin:0; background:var(--bg); color:var(--text); font:14px/1.4 system-ui, sans-serif;
         height:100vh; display:flex; flex-direction:column; overflow:hidden; }
  input, select, button { background:var(--bg); color:var(--text); border:1px solid var(--line); border-radius:6px;
                          padding:5px 9px; font:inherit; }
  button { cursor:pointer; } button:hover:not(:disabled) { border-color:var(--accent); }
  button:disabled { opacity:.45; cursor:default; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  button.on { background:var(--accent); border-color:var(--accent); color:#fff; }
  input[type=number] { width:76px; }
  kbd { background:var(--line); border-radius:4px; padding:0 5px; font-size:12px; }
  .muted { color:var(--muted); }
  .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  .grow { flex:1; min-width:180px; }
  header { padding:8px 12px; background:var(--panel); border-bottom:1px solid var(--line); display:flex; flex-direction:column; gap:6px; }
  header h1 { font-size:16px; margin:0 8px 0 0; }
  .steps { display:flex; gap:4px; }
  .steps button { border-radius:16px; }
  .steps button.done { border-color:var(--good); }
  main { flex:1; min-height:0; display:grid; grid-template-columns:200px 1fr 300px; }
  #photos { list-style:none; margin:0; padding:6px; overflow-y:auto; border-right:1px solid var(--line); background:var(--panel); }
  #photos li { padding:5px 8px; border-radius:5px; cursor:pointer; display:flex; gap:6px; align-items:center; }
  #photos li:hover { background:var(--line); } #photos li.on { background:var(--accent); color:#fff; }
  #photos li .dot { width:8px; height:8px; border-radius:50%; flex:none; background:var(--line); }
  #photos li .dot.draft { background:var(--warn); } #photos li .dot.saved { background:var(--good); }
  #photos li span { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .stagewrap { position:relative; min-width:0; min-height:0; background:#0b0c0e; overflow:hidden; }
  #stage { position:absolute; inset:0; display:block; }
  #hud { position:absolute; left:8px; bottom:8px; background:#000b; padding:3px 8px; border-radius:6px; pointer-events:none; }
  #busy { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; background:#0008; font-size:16px; }
  aside { border-left:1px solid var(--line); background:var(--panel); display:flex; flex-direction:column; min-height:0; }
  .panel { padding:10px; display:flex; flex-direction:column; gap:8px; border-bottom:1px solid var(--line); }
  .panel h2 { font-size:14px; margin:0; }
  .panel p { margin:0; }
  .grid2 { display:grid; grid-template-columns:auto 1fr; gap:6px 8px; align-items:center; }
  #chips { flex:1; min-height:0; overflow-y:auto; padding:8px; display:grid; grid-template-columns:repeat(auto-fill, minmax(78px, 1fr));
           grid-auto-rows:max-content; gap:6px; }
  .chip { border:3px solid var(--line); border-radius:6px; cursor:pointer; position:relative; background:#000; }
  .chip canvas { display:block; width:100%; aspect-ratio:1; }
  .chip .n { position:absolute; left:3px; top:2px; font-size:11px; background:#000a; padding:0 4px; border-radius:4px; }
  .chip.good { border-color:var(--good); } .chip.bad { border-color:var(--bad); }
  .chip.sel { outline:2px solid #fff; outline-offset:1px; }
  .result img { width:100%; background:#000; border:1px solid var(--line); border-radius:6px; }
  .result a { color:var(--accent); }
  .toast { position:fixed; right:16px; bottom:16px; background:var(--panel); border:1px solid var(--line);
           padding:10px 14px; border-radius:8px; display:none; white-space:pre-line; max-width:420px; z-index:10; }
  @media (max-width:900px) { main { grid-template-columns:1fr; grid-template-rows:110px 1fr 260px; }
    #photos { border-right:none; border-bottom:1px solid var(--line); } aside { border-left:none; border-top:1px solid var(--line); } }
</style>
</head>
<body>
<header>
  <div class="row">
    <h1>Маски фруктов</h1>
    <label class="muted">Фото</label><input id="src" class="grow" placeholder="папка с фотографиями">
    <button data-browse="src">Обзор…</button>
    <label class="muted">Результаты</label><input id="out" class="grow" placeholder="по умолчанию &lt;папка&gt;_masks">
    <button data-browse="out">Обзор…</button>
    <button id="openBtn" class="primary">Открыть</button>
  </div>
  <div class="row">
    <div class="steps">
      <button data-step="1">1. Найти маски</button>
      <button data-step="2">2. Отбор</button>
      <button data-step="3">3. Редактирование</button>
      <button data-step="4">4. Сохранить</button>
    </div>
    <span id="title" class="muted"></span>
  </div>
</header>
<main>
  <ul id="photos"></ul>
  <div class="stagewrap" id="wrap">
    <canvas id="stage"></canvas>
    <div id="hud"></div>
    <div id="busy" hidden>Ищу фрукты…</div>
  </div>
  <aside>
    <!-- шаг 1 -->
    <div class="panel" data-for="1">
      <h2>1. Найти маски (HOG)</h2>
      <div class="grid2">
        <label>Мин. фрукт, px</label><input id="min_size" type="number" min="8" step="2">
        <label>Макс. фрукт, px</label><input id="max_size" type="number" min="0" step="2" title="0 = половина меньшей стороны фото">
        <label>Порог</label><input id="threshold" type="number" min="-1" max="1" step="0.05">
        <label>NMS</label><input id="nms_iou" type="number" min="0.05" max="0.9" step="0.05">
        <label>Маска</label><select id="mode"><option value="grabcut">grabcut</option><option value="ellipse">эллипс</option><option value="box">рамка</option></select>
      </div>
      <p class="muted">Размер — примерный диаметр фрукта на фото. Порог — насколько окно должно быть похоже на круг: ниже — больше масок, но и больше мусора.</p>
      <button id="detectBtn" class="primary">Найти маски</button>
    </div>
    <!-- шаг 2 -->
    <div class="panel" data-for="2">
      <h2>2. Отбор</h2>
      <p><b>ЛКМ</b> по маске — хорошая, <b>ПКМ</b> — плохая, <b>Shift+клик</b> — снять отметку. То же — по миниатюрам ниже.</p>
      <p id="reviewStats" class="muted"></p>
      <div class="row">
        <button id="restGood">Неотмеченные → хорошие</button>
        <button id="restBad">Неотмеченные → плохие</button>
      </div>
      <button id="keepGood" class="primary">Стереть плохие, оставить хорошие →</button>
    </div>
    <!-- шаг 3 -->
    <div class="panel" data-for="3">
      <h2>3. Редактирование</h2>
      <div class="row" id="tools">
        <button data-tool="select">Выбор <kbd>V</kbd></button>
        <button data-tool="brush">Кисть <kbd>B</kbd></button>
        <button data-tool="erase">Ластик <kbd>E</kbd></button>
      </div>
      <div class="row">
        <button id="newMask">Новая маска <kbd>N</kbd></button>
        <button id="delMask">Удалить маску <kbd>Del</kbd></button>
      </div>
      <div class="row"><label>Кисть</label><input id="brush" type="range" min="1" max="60" value="8" class="grow"><span id="brushVal"></span></div>
      <label class="row"><input type="checkbox" id="protect" checked> не закрашивать чужие маски</label>
      <p class="muted">Кисть добавляет к выбранной маске (ПКМ кистью — стирает). Ластик стирает выбранную маску, а если ничего не выбрано — любую.
        Колесо — масштаб, средняя кнопка или пробел+перетаскивание — сдвиг, <kbd>F</kbd> — вписать, <kbd>Ctrl+Z</kbd> — отменить.</p>
      <button id="toSave" class="primary">Готово → сохранить</button>
    </div>
    <!-- шаг 4 -->
    <div class="panel" data-for="4">
      <h2>4. Сохранить</h2>
      <button id="saveBtn" class="primary">Сохранить маску и JSON</button>
      <div class="result" id="result"></div>
    </div>
    <div id="chips"></div>
  </aside>
</main>
<div class="toast" id="toast"></div>

<script>
const $ = id => document.getElementById(id);
const PARAMS = ["min_size", "max_size", "threshold", "nms_iou"];

async function api(url, body) {
  const r = await fetch(url, body === undefined ? {} : { method: "POST", body: JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
function toast(text, ms = 4000) {
  const t = $("toast"); t.textContent = text; t.style.display = "block";
  clearTimeout(t._h); t._h = setTimeout(() => t.style.display = "none", ms);
}

// ================================================================ данные фото
let photos = [], cur = null;          // имя текущего фото
let img = null, W = 0, H = 0;
let labels = null;                    // Uint16Array W*H: 0 — фон, иначе id маски
let masks = new Map();                // id -> {id, score, status, bbox:[x0,y0,x1,y1], area}
let step = 1, selected = 0, tool = "brush";
let undoStack = [];
const overlay = document.createElement("canvas");
let octx = null, odata = null;

function b64FromLabels() {
  const bytes = new Uint8Array(labels.buffer);
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(s);
}
function labelsFromB64(b64) {
  const s = atob(b64), bytes = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i);
  return new Uint16Array(bytes.buffer);
}

function recomputeMaskStats() {
  for (const m of masks.values()) { m.bbox = [Infinity, Infinity, -1, -1]; m.area = 0; }
  for (let y = 0, i = 0; y < H; y++) for (let x = 0; x < W; x++, i++) {
    const id = labels[i];
    if (!id) continue;
    let m = masks.get(id);
    if (!m) { m = { id, score: null, status: "good", bbox: [Infinity, Infinity, -1, -1], area: 0 }; masks.set(id, m); }
    m.area++;
    const b = m.bbox;
    if (x < b[0]) b[0] = x; if (y < b[1]) b[1] = y; if (x > b[2]) b[2] = x; if (y > b[3]) b[3] = y;
  }
  for (const [id, m] of masks) if (!m.area) masks.delete(id);
  if (selected && !masks.has(selected) && tool !== "brush") selected = 0;
}

// ================================================================ отрисовка
const colorCache = [];
function idColor(id) {
  if (colorCache[id]) return colorCache[id];
  const h = (id * 137.508) % 360, s = 0.75, l = 0.55;
  const f = n => { const k = (n + h / 30) % 12; return Math.round(255 * (l - s * Math.min(l, 1 - l) * Math.max(-1, Math.min(k - 3, 9 - k, 1)))); };
  return colorCache[id] = [f(0), f(8), f(4)];
}
const GOOD = [61, 220, 90], BAD = [229, 72, 77];

function maskColor(id) {
  if (step === 2) {
    const st = masks.get(id)?.status;
    if (st === "good") return GOOD;
    if (st === "bad") return BAD;
  }
  return idColor(id);
}

// перерисовать оверлей в прямоугольнике [x0, x1) x [y0, y1)
function paintOverlay(x0 = 0, y0 = 0, x1 = W, y1 = H) {
  x0 = Math.max(0, Math.floor(x0)); y0 = Math.max(0, Math.floor(y0));
  x1 = Math.min(W, Math.ceil(x1)); y1 = Math.min(H, Math.ceil(y1));
  if (x1 <= x0 || y1 <= y0) return;
  const d = odata.data;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const i = y * W + x, id = labels[i], o = i * 4;
      if (!id) { d[o + 3] = 0; continue; }
      const edge = (x === 0 || labels[i - 1] !== id) || (x === W - 1 || labels[i + 1] !== id) ||
                   (y === 0 || labels[i - W] !== id) || (y === H - 1 || labels[i + W] !== id);
      const c = maskColor(id), sel = id === selected;
      if (edge) {
        const e = sel ? [255, 255, 255] : c;
        d[o] = e[0]; d[o + 1] = e[1]; d[o + 2] = e[2]; d[o + 3] = 255;
      } else {
        d[o] = c[0]; d[o + 1] = c[1]; d[o + 2] = c[2];
        d[o + 3] = sel ? 150 : (step === 2 && !masks.get(id)?.status ? 70 : 105);
      }
    }
  }
  octx.putImageData(odata, 0, 0, x0, y0, x1 - x0, y1 - y0);
}
function paintMask(id) {
  const m = masks.get(id);
  if (m) paintOverlay(m.bbox[0] - 1, m.bbox[1] - 1, m.bbox[2] + 2, m.bbox[3] + 2);
}

const stage = $("stage"), sctx = stage.getContext("2d");
let view = { k: 1, ox: 0, oy: 0 }, mouse = { x: -1, y: -1, inside: false };

function resizeStage() {
  const r = $("wrap").getBoundingClientRect();
  stage.width = Math.max(1, Math.round(r.width)); stage.height = Math.max(1, Math.round(r.height));
  draw();
}
function fitView() {
  if (!W) return;
  const k = Math.min(stage.width / W, stage.height / H) * 0.98;
  view = { k, ox: (stage.width - W * k) / 2, oy: (stage.height - H * k) / 2 };
  draw();
}
function draw() {
  sctx.setTransform(1, 0, 0, 1, 0, 0);
  sctx.clearRect(0, 0, stage.width, stage.height);
  if (!img || !W) return;
  sctx.setTransform(view.k, 0, 0, view.k, view.ox, view.oy);
  sctx.imageSmoothingEnabled = view.k < 1;
  sctx.drawImage(img, 0, 0);
  sctx.imageSmoothingEnabled = false;
  if (labels) sctx.drawImage(overlay, 0, 0);
  if (step === 3 && tool !== "select" && mouse.inside) {
    const r = +$("brush").value;
    const [ix, iy] = toImage(mouse.x, mouse.y);
    sctx.lineWidth = 1.5 / view.k;
    sctx.strokeStyle = tool === "erase" ? "#ff8080" : "#ffffff";
    sctx.beginPath(); sctx.arc(ix, iy, r, 0, Math.PI * 2); sctx.stroke();
  }
}
const toImage = (sx, sy) => [(sx - view.ox) / view.k, (sy - view.oy) / view.k];
new ResizeObserver(resizeStage).observe($("wrap"));

// ================================================================ миниатюры
function renderChips() {
  const box = $("chips");
  box.innerHTML = "";
  if (step === 1 || !labels) return;
  const list = [...masks.values()].sort((a, b) => a.id - b.id);
  for (const m of list) {
    const div = document.createElement("div");
    div.className = "chip";
    div.dataset.id = m.id;
    const c = document.createElement("canvas");
    c.width = c.height = 72;
    div.appendChild(c);
    const n = document.createElement("span"); n.className = "n"; n.textContent = m.id; div.appendChild(n);
    div.onclick = e => chipClick(m.id, e);
    div.oncontextmenu = e => { e.preventDefault(); chipClick(m.id, e, true); };
    box.appendChild(div);
    drawChip(m.id);
  }
  updateChipClasses();
}
function drawChip(id) {
  const m = masks.get(id), div = $("chips").querySelector(`[data-id="${id}"]`);
  if (!m || !div) return;
  const c = div.querySelector("canvas"), g = c.getContext("2d");
  const bw = m.bbox[2] - m.bbox[0] + 1, bh = m.bbox[3] - m.bbox[1] + 1, side = Math.max(bw, bh) * 1.25;
  const sx = (m.bbox[0] + m.bbox[2]) / 2 - side / 2, sy = (m.bbox[1] + m.bbox[3]) / 2 - side / 2;
  g.clearRect(0, 0, 72, 72);
  g.drawImage(img, sx, sy, side, side, 0, 0, 72, 72);
  g.imageSmoothingEnabled = false;
  g.drawImage(overlay, sx, sy, side, side, 0, 0, 72, 72);
}
function updateChipClasses() {
  for (const div of $("chips").children) {
    const m = masks.get(+div.dataset.id);
    div.className = "chip" + (step === 2 && m?.status ? " " + m.status : "") + (+div.dataset.id === selected ? " sel" : "");
  }
  const vals = [...masks.values()];
  const g = vals.filter(m => m.status === "good").length, b = vals.filter(m => m.status === "bad").length;
  $("reviewStats").textContent = `Масок: ${vals.length} · хороших ${g} · плохих ${b} · не отмечено ${vals.length - g - b}`;
}
function chipClick(id, e, right = false) {
  if (step === 2) setStatus(id, e.shiftKey ? null : right ? "bad" : "good", true);
  else if (step === 3) { select(id); centerOn(id); }
}
function centerOn(id) {
  const m = masks.get(id);
  if (!m) return;
  const cx = (m.bbox[0] + m.bbox[2]) / 2, cy = (m.bbox[1] + m.bbox[3]) / 2;
  const [x, y] = toImage(0, 0), [x2, y2] = toImage(stage.width, stage.height);
  if (cx < x || cx > x2 || cy < y || cy > y2) { view.ox = stage.width / 2 - cx * view.k; view.oy = stage.height / 2 - cy * view.k; draw(); }
}

// ================================================================ шаги
const stepTitle = { 1: "Найти маски", 2: "Отбор", 3: "Редактирование", 4: "Сохранить" };
function setStep(s) {
  if (s > 1 && !labels) { toast("Сначала найдите маски (шаг 1)"); s = 1; }
  if (s === 3 && [...masks.values()].some(m => m.status !== "good")) {
    toast("Сначала завершите отбор: «Стереть плохие, оставить хорошие»"); s = 2;
  }
  step = s;
  document.querySelectorAll("[data-for]").forEach(p => p.hidden = +p.dataset.for !== s);
  document.querySelectorAll("[data-step]").forEach(b => b.classList.toggle("on", +b.dataset.step === s));
  if (s !== 3) selected = 0;
  if (labels) { paintOverlay(); renderChips(); }
  if (s === 4) showResult(null);
  draw();
  scheduleDraft();
}
document.querySelectorAll("[data-step]").forEach(b => b.onclick = () => setStep(+b.dataset.step));

// ================================================================ папки и фото
function params() { const p = { mode: $("mode").value }; PARAMS.forEach(k => p[k] = Number($(k).value)); return p; }

async function init() {
  const s = await api("/api/state");
  PARAMS.forEach(k => $(k).value = s.defaults[k]);
  $("src").value = s.src; $("out").value = s.out;
  setPhotos(s.sources);
  setStep(1);
}
function setPhotos(list) {
  photos = list;
  const ul = $("photos"); ul.innerHTML = "";
  for (const p of photos) {
    const li = document.createElement("li");
    li.innerHTML = `<i class="dot ${p.status}"></i><span></span>`;
    li.querySelector("span").textContent = p.name;
    li.title = { new: "не начато", draft: "в работе", saved: "сохранено" }[p.status] + ": " + p.name;
    li.className = p.name === cur ? "on" : "";
    li.onclick = () => openPhoto(p.name);
    ul.appendChild(li);
  }
  if (!cur && photos.length) openPhoto(photos[0].name);
}
async function refreshPhotos() { try { setPhotos((await api("/api/state")).sources); } catch (e) {} }

async function openFolders() {
  try {
    await flushDraft();
    const r = await api("/api/open", { src: $("src").value, out: $("out").value });
    $("src").value = r.src; $("out").value = r.out;
    cur = null; setPhotos(r.sources);
    toast(`Фото: ${r.sources.length}\nРезультаты: ${r.out}`);
  } catch (e) { toast(e.message); }
}
$("openBtn").onclick = openFolders;
["src", "out"].forEach(id => $(id).addEventListener("keydown", e => { if (e.key === "Enter") openFolders(); }));
document.querySelectorAll("[data-browse]").forEach(b => b.onclick = async () => {
  b.disabled = true;
  try {
    const r = await api("/api/browse", { initial: $(b.dataset.browse).value });
    if (r.path) { $(b.dataset.browse).value = r.path; if (b.dataset.browse === "src") $("out").value = ""; await openFolders(); }
  } catch (e) { toast(e.message); }
  b.disabled = false;
});

async function openPhoto(name) {
  await flushDraft();
  cur = name;
  [...$("photos").children].forEach((li, i) => li.className = photos[i].name === name ? "on" : "");
  $("title").textContent = name;
  labels = null; masks = new Map(); undoStack = []; selected = 0; W = H = 0;
  img = new Image();
  await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = "/src/" + encodeURIComponent(name); });
  W = img.naturalWidth; H = img.naturalHeight;
  overlay.width = W; overlay.height = H;
  octx = overlay.getContext("2d"); odata = octx.createImageData(W, H);
  fitView();
  try {
    const d = await api("/api/load?name=" + encodeURIComponent(name));
    if (d.labels) { applyDoc(d); setStep(d.step || 2); return; }
  } catch (e) { toast(e.message); }
  setStep(1);
}

function applyDoc(d) {
  if (d.width !== W || d.height !== H) throw new Error("размер маски не совпадает с фото");
  labels = labelsFromB64(d.labels);
  masks = new Map(d.masks.map(m => [m.id, { id: m.id, score: m.score ?? null, status: m.status ?? null }]));
  recomputeMaskStats();
}

// ================================================================ шаг 1: поиск
$("detectBtn").onclick = async () => {
  if (!cur) return toast("Откройте папку с фото");
  if (labels && !confirm("Найти маски заново? Текущие маски этого фото (отбор и правки) будут заменены.")) return;
  $("busy").hidden = false;
  try {
    const d = await api("/api/detect", { name: cur, ...params() });
    undoStack = [];
    applyDoc(d);
    setStep(2);
    toast(`Найдено масок: ${masks.size} (${d.seconds} с)`);
    refreshPhotos();
  } catch (e) { toast(e.message); }
  $("busy").hidden = true;
};

// ================================================================ отмена
function pushUndo() {
  undoStack.push({ labels: labels.slice(), masks: JSON.stringify([...masks.values()].map(({ id, score, status }) => ({ id, score, status }))), selected, step });
  if (undoStack.length > 30) undoStack.shift();
}
function undo() {
  const u = undoStack.pop();
  if (!u) return;
  labels = u.labels;
  masks = new Map(JSON.parse(u.masks).map(m => [m.id, m]));
  selected = u.selected;
  recomputeMaskStats();
  if (u.step !== step) { setStep(u.step); if (step === 3) select(u.selected); return; }
  paintOverlay(); renderChips(); draw(); scheduleDraft();
}

// ================================================================ шаг 2: отбор
function setStatus(id, status, record = true) {
  const m = masks.get(id);
  if (!m) return;
  if (record) pushUndo();
  m.status = m.status === status ? null : status;
  paintMask(id); drawChip(id); updateChipClasses(); draw(); scheduleDraft();
}
function setRest(status) {
  pushUndo();
  for (const m of masks.values()) if (!m.status) m.status = status;
  paintOverlay(); updateChipClasses(); renderChips(); draw(); scheduleDraft();
}
$("restGood").onclick = () => setRest("good");
$("restBad").onclick = () => setRest("bad");
$("keepGood").onclick = () => {
  const vals = [...masks.values()];
  const bad = vals.filter(m => m.status === "bad").length, none = vals.filter(m => !m.status).length;
  if (!vals.some(m => m.status === "good") && !confirm("Нет ни одной хорошей маски. Стереть все?")) return;
  if (none && !confirm(`${none} масок не отмечено — они тоже будут стёрты. Продолжить?`)) return;
  pushUndo();
  for (let i = 0; i < labels.length; i++) if (labels[i] && masks.get(labels[i])?.status !== "good") labels[i] = 0;
  for (const [id, m] of masks) if (m.status !== "good") masks.delete(id);
  recomputeMaskStats();
  toast(`Стёрто масок: ${bad + none}, осталось: ${masks.size}`);
  tool = "select"; updateTools();
  setStep(3);
};

// ================================================================ шаг 3: редактирование
function updateTools() {
  document.querySelectorAll("[data-tool]").forEach(b => b.classList.toggle("on", b.dataset.tool === tool));
  stage.style.cursor = tool === "select" ? "pointer" : "none";
  draw();
}
document.querySelectorAll("[data-tool]").forEach(b => b.onclick = () => { tool = b.dataset.tool; updateTools(); });
$("brush").oninput = () => { $("brushVal").textContent = $("brush").value; draw(); };
$("brushVal").textContent = $("brush").value;

function select(id) {
  const prev = selected;
  selected = id;
  if (prev) paintMask(prev);
  if (id) paintMask(id);
  updateChipClasses(); draw();
}
function newMask() {
  let id = 1;
  for (const k of masks.keys()) id = Math.max(id, k + 1);
  if (id > 65535) return toast("Слишком много масок");
  masks.set(id, { id, score: null, status: "good", bbox: [Infinity, Infinity, -1, -1], area: 0 });
  const prev = selected; selected = id; if (prev) paintMask(prev);
  tool = "brush"; updateTools(); updateChipClasses();
  toast(`Новая маска №${id}: закрасьте фрукт кистью`, 2500);
}
function deleteSelected() {
  if (!selected) return toast("Выберите маску");
  pushUndo();
  const m = masks.get(selected), id = selected;
  for (let i = 0; i < labels.length; i++) if (labels[i] === id) labels[i] = 0;
  masks.delete(id); selected = 0;
  if (m?.area) paintOverlay(m.bbox[0] - 1, m.bbox[1] - 1, m.bbox[2] + 2, m.bbox[3] + 2);
  renderChips(); draw(); scheduleDraft();
}
$("newMask").onclick = newMask;
$("delMask").onclick = deleteSelected;
$("toSave").onclick = () => setStep(4);

let stroke = null; // {mode: "add"|"erase", last:[x,y], dirty:[x0,y0,x1,y1]}
function stamp(cx, cy, r, mode, dirty) {
  const x0 = Math.max(0, Math.floor(cx - r)), x1 = Math.min(W - 1, Math.ceil(cx + r));
  const y0 = Math.max(0, Math.floor(cy - r)), y1 = Math.min(H - 1, Math.ceil(cy + r));
  const r2 = r * r, protect = $("protect").checked;
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    if ((x - cx) ** 2 + (y - cy) ** 2 > r2) continue;
    const i = y * W + x, v = labels[i];
    if (mode === "add") { if (v === selected || (protect && v)) continue; labels[i] = selected; }
    else { if (!v || (selected && v !== selected)) continue; labels[i] = 0; }
  }
  dirty[0] = Math.min(dirty[0], x0); dirty[1] = Math.min(dirty[1], y0);
  dirty[2] = Math.max(dirty[2], x1); dirty[3] = Math.max(dirty[3], y1);
}
function strokeTo(ix, iy) {
  const r = +$("brush").value, [lx, ly] = stroke.last;
  const dist = Math.hypot(ix - lx, iy - ly), n = Math.max(1, Math.ceil(dist / Math.max(1, r / 3)));
  for (let s = 1; s <= n; s++) stamp(lx + (ix - lx) * s / n, ly + (iy - ly) * s / n, r, stroke.mode, stroke.dirty);
  stroke.last = [ix, iy];
  const d = stroke.dirty;
  paintOverlay(d[0] - 1, d[1] - 1, d[2] + 2, d[3] + 2);
  stroke.dirty = [Infinity, Infinity, -1, -1];
  draw();
}

// ================================================================ мышь и клавиатура
let pan = null, spaceDown = false;
stage.addEventListener("contextmenu", e => e.preventDefault());
stage.addEventListener("mousedown", e => {
  if (!labels && step !== 1) return;
  const [ix, iy] = toImage(e.offsetX, e.offsetY);
  if (e.button === 1 || (e.button === 0 && spaceDown) || step === 1 || step === 4) {
    pan = { x: e.clientX, y: e.clientY, ox: view.ox, oy: view.oy }; e.preventDefault(); return;
  }
  const inside = ix >= 0 && iy >= 0 && ix < W && iy < H;
  const id = inside ? labels[Math.floor(iy) * W + Math.floor(ix)] : 0;
  if (step === 2) {
    if (id) setStatus(id, e.shiftKey ? null : e.button === 2 ? "bad" : "good");
    return;
  }
  if (step === 3) {
    if (tool === "select") { select(id); return; }
    const mode = tool === "erase" || e.button === 2 ? "erase" : "add";
    if (mode === "add" && !selected) {
      if (id) { select(id); } else { toast("Выберите маску (V) или создайте новую (N)", 2500); return; }
    }
    pushUndo();
    stroke = { mode, last: [ix, iy], dirty: [Infinity, Infinity, -1, -1] };
    strokeTo(ix, iy);
  }
});
window.addEventListener("mousemove", e => {
  const r = stage.getBoundingClientRect();
  mouse = { x: e.clientX - r.left, y: e.clientY - r.top, inside: e.target === stage };
  if (pan) { view.ox = pan.ox + e.clientX - pan.x; view.oy = pan.oy + e.clientY - pan.y; draw(); return; }
  if (stroke) { const [ix, iy] = toImage(mouse.x, mouse.y); strokeTo(ix, iy); return; }
  if (mouse.inside && labels) {
    const [ix, iy] = toImage(mouse.x, mouse.y);
    const id = ix >= 0 && iy >= 0 && ix < W && iy < H ? labels[Math.floor(iy) * W + Math.floor(ix)] : 0;
    const m = masks.get(id);
    $("hud").textContent = `x ${Math.floor(ix)}, y ${Math.floor(iy)}` + (m ? ` · маска №${id}` + (m.score != null ? ` (похожесть ${m.score})` : "") : "");
  }
  if (step === 3) draw();
});
window.addEventListener("mouseup", () => {
  if (pan) pan = null;
  if (stroke) {
    stroke = null;
    recomputeMaskStats();
    const known = new Set([...$("chips").children].map(d => +d.dataset.id));
    if (known.size !== masks.size || [...masks.keys()].some(id => !known.has(id))) renderChips();
    else for (const id of masks.keys()) drawChip(id);
    updateChipClasses();
    scheduleDraft();
  }
});
stage.addEventListener("mouseleave", () => { mouse.inside = false; draw(); });
stage.addEventListener("wheel", e => {
  e.preventDefault();
  const f = e.deltaY < 0 ? 1.2 : 1 / 1.2;
  const [ix, iy] = toImage(e.offsetX, e.offsetY);
  view.k = Math.min(40, Math.max(0.05, view.k * f));
  view.ox = e.offsetX - ix * view.k; view.oy = e.offsetY - iy * view.k;
  draw();
}, { passive: false });

document.addEventListener("keydown", e => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName) && e.target.type !== "range" && e.target.type !== "checkbox") return;
  const k = e.key.toLowerCase();
  if (k === " ") { spaceDown = true; e.preventDefault(); return; }
  if ((e.ctrlKey || e.metaKey) && (k === "z" || k === "я")) { e.preventDefault(); undo(); return; }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (k === "f" || k === "а") fitView();
  if (step !== 3) return;
  if (k === "v" || k === "м") { tool = "select"; updateTools(); }
  else if (k === "b" || k === "и") { tool = "brush"; updateTools(); }
  else if (k === "e" || k === "у") { tool = "erase"; updateTools(); }
  else if (k === "n" || k === "т") newMask();
  else if (k === "delete") deleteSelected();
  else if (k === "[" || k === "х") { $("brush").value = Math.max(1, +$("brush").value - 1); $("brush").oninput(); }
  else if (k === "]" || k === "ъ") { $("brush").value = Math.min(60, +$("brush").value + 1); $("brush").oninput(); }
  else if (k === "escape") select(0);
});
document.addEventListener("keyup", e => { if (e.key === " ") spaceDown = false; });

// ================================================================ черновик и сохранение
let draftTimer = null, draftDirty = false;
function docBody() {
  return { name: cur, width: W, height: H, labels: b64FromLabels(), step,
           masks: [...masks.values()].map(({ id, score, status }) => ({ id, score, status })) };
}
function scheduleDraft() {
  if (!labels || !cur) return;
  draftDirty = true;
  clearTimeout(draftTimer);
  draftTimer = setTimeout(flushDraft, 800);
}
async function flushDraft() {
  clearTimeout(draftTimer);
  if (!draftDirty || !labels || !cur) return;
  draftDirty = false;
  try { await api("/api/draft", docBody()); } catch (e) { toast("Не удалось сохранить черновик: " + e.message); }
}
window.addEventListener("beforeunload", () => {
  if (draftDirty && labels) navigator.sendBeacon("/api/draft", JSON.stringify(docBody()));
});

function showResult(r) {
  const box = $("result");
  if (!r) { box.innerHTML = `<p class="muted">Масок: ${masks.size}. Будут сохранены маска целиком (PNG) и JSON с пикселями каждой маски.</p>`; return; }
  const t = Date.now(), q = s => encodeURIComponent(s);
  box.innerHTML = `<p>Сохранено масок: <b>${r.count}</b></p>
    <img src="/out/${q(r.mask)}?t=${t}" alt="маска">
    <p><a href="/out/${q(r.mask)}?download=1">${r.mask}</a><br>
       <a href="/out/${q(r.json)}?download=1">${r.json}</a> <span class="muted">(${r.json_kb} КБ)</span></p>
    <p class="muted">Папка: ${$("out").value}</p>`;
}
$("saveBtn").onclick = async () => {
  if (!labels) return;
  try {
    const r = await api("/api/save", docBody());
    draftDirty = false;
    showResult(r);
    toast(`Сохранено: ${r.mask}, ${r.json}`);
    refreshPhotos();
  } catch (e) { toast(e.message); }
};

init();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description="Маски фруктов: HOG -> отбор -> редактирование -> PNG + JSON")
    ap.add_argument("--src", default="", help="папка с фото (можно выбрать в приложении)")
    ap.add_argument("--out", default="", help="папка для результатов (по умолчанию <src>_masks)")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    state = State()
    if args.src:
        state.open(args.src, args.out)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    url = f"http://127.0.0.1:{args.port}"
    print(f"Открой {url}  (Ctrl+C — выход)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
