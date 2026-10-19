"""Веб-приложение: HOG-вырезание яблок из фото куч + разметка fresh / rotten / garbage.

Запуск:
    python apple_app.py            # откроет браузер на http://127.0.0.1:8765

Вкладка «1. Вырезание»: выбрать папку с фото куч -> подобрать параметры по предпросмотру ->
«Обработать всю папку». Каждое найденное яблоко сохраняется на белый квадрат
в папку для яблок (+ detections.csv: откуда вырезано).

Вкладка «2. Разметка»: размечать вырезанные яблоки клавишами 1/2/3.
Метки сразу пишутся в <папка для яблок>/labels.csv.
«Экспорт» раскладывает размеченное в <папка экспорта>/train/{rotten,fresh,garbage}.

Сам алгоритм HOG (без обучения: окна сравниваются с HOG эталонного круга) — в hog_detector.py.
"""

import argparse
import base64
import csv
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2

from hog_detector import IMAGE_EXTS, DetectParams, cutout_on_white, detect, imread, list_images, process_folder

# ==================== РАЗМЕР БЕЛОГО КВАДРАТА ====================
# Сторона квадрата (px), на который кладётся каждое вырезанное яблоко.
# Значение по умолчанию; в приложении его тоже можно поменять.
SQUARE_SIZE = 128
# ================================================================

ROOT = Path(__file__).resolve().parent
LABELS = ("rotten", "fresh", "garbage")
PREVIEW_CUTS = 16
CROP_RE = re.compile(r"(.+)_(\d{3,})\.png", re.IGNORECASE)


class LabelStore:
    """Метки картинок одной папки, хранятся в <папка>/labels.csv."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.csv_path = folder / "labels.csv"
        self.lock = threading.Lock()
        self.labels: dict[str, str] = {}
        if self.csv_path.exists():
            with self.csv_path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("label") in LABELS:
                        self.labels[row["filename"]] = row["label"]

    def _save(self):
        tmp = self.csv_path.with_suffix(".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "label"])
            for k in sorted(self.labels):
                w.writerow([k, self.labels[k]])
        tmp.replace(self.csv_path)

    def set_label(self, name: str, label: str | None):
        with self.lock:
            if label in LABELS:
                self.labels[name] = label
            else:
                self.labels.pop(name, None)
            self._save()

    def drop(self, names: set[str]):
        with self.lock:
            if names & self.labels.keys():
                for n in names:
                    self.labels.pop(n, None)
                self._save()

    def export(self, out_dir: Path) -> dict:
        with self.lock:
            items = dict(self.labels)
        counts = {}
        for label in LABELS:
            dst = out_dir / "train" / label
            # папка класса собирается заново, чтобы перемаркированные картинки не остались в старом классе
            if dst.exists():
                shutil.rmtree(dst)
            dst.mkdir(parents=True)
            names = sorted(n for n, l in items.items() if l == label and (self.folder / n).is_file())
            for n in names:
                shutil.copy2(self.folder / n, dst / n)
            counts[label] = len(names)
        return counts


class State:
    def __init__(self, export_dir: Path):
        self.lock = threading.Lock()
        self.src: Path | None = None
        self.out: Path | None = None
        self.export_dir = export_dir
        self.store: LabelStore | None = None
        self.job = {"running": False, "done": 0, "total": 0, "current": "", "found": 0, "error": None}

    def open(self, src: str, out: str):
        src_p = Path(src).expanduser() if src.strip() else None
        if src_p is not None and not src_p.is_dir():
            raise ValueError(f"Папка не найдена: {src_p}")
        if out.strip():
            out_p = Path(out).expanduser()
        elif src_p is not None:
            out_p = src_p.parent / f"{src_p.name}_apples"
        else:
            raise ValueError("Укажите папку с фото или папку с яблоками")
        out_p.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.src = src_p.resolve() if src_p else None
            self.out = out_p.resolve()
            self.store = LabelStore(self.out)

    def sources(self) -> list[str]:
        return [p.name for p in list_images(self.src)] if self.src else []

    def items(self) -> list[dict]:
        if not self.out:
            return []
        sources = {}
        det = self.out / "detections.csv"
        if det.exists():
            with det.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    sources[row["crop"]] = row["source"]
        return [{
            "name": p.name,
            "label": self.store.labels.get(p.name),
            "source": sources.get(p.name),
            "v": int(p.stat().st_mtime),
        } for p in list_images(self.out)]


def params_from(data: dict) -> tuple[DetectParams, int, str]:
    p = DetectParams(
        min_size=max(8, int(data.get("min_size", DetectParams.min_size))),
        max_size=max(0, int(data.get("max_size", DetectParams.max_size))),
        threshold=float(data.get("threshold", DetectParams.threshold)),
        nms_iou=float(data.get("nms_iou", DetectParams.nms_iou)),
    )
    square = min(4096, max(8, int(data.get("square", SQUARE_SIZE))))
    mode = data.get("mode", "grabcut")
    if mode not in ("grabcut", "ellipse", "box"):
        mode = "grabcut"
    return p, square, mode


def run_job(state: State, src: Path, out: Path, params: DetectParams, square: int, mode: str):
    job = state.job
    try:
        stems = {p.stem for p in list_images(src)}
        # старые вырезки этих же фото удаляем, их метки сбрасываем: под теми же именами будут новые яблоки
        old = [p for p in list_images(out) if (m := CROP_RE.fullmatch(p.name)) and m.group(1) in stems]
        for p in old:
            p.unlink()
        state.store.drop({p.name for p in old})

        def on_progress(i, n, name, count):
            job.update(done=i, total=n, current=name, found=job["found"] + count)

        process_folder(src, out, square, params, mode, on_progress)
    except Exception as e:
        job["error"] = str(e)
    finally:
        job["running"] = False


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


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def _file(self, folder: Path | None, name: str):
            file = folder / Path(name).name if folder else None
            if not file or file.suffix.lower() not in IMAGE_EXTS or not file.is_file():
                return self._send(404, b"not found", "text/plain")
            self._send(200, file.read_bytes(), mimetypes.guess_type(file.name)[0] or "application/octet-stream")

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._json({
                    "src": str(state.src or ""), "out": str(state.out or ""), "export": str(state.export_dir),
                    "job": state.job,
                    "sources": state.sources(),
                    "defaults": {"square": SQUARE_SIZE, "min_size": DetectParams.min_size,
                                 "max_size": DetectParams.max_size, "threshold": DetectParams.threshold,
                                 "nms_iou": DetectParams.nms_iou},
                })
            elif path == "/api/job":
                self._json(state.job)
            elif path == "/api/items":
                self._json({"out": str(state.out or ""), "items": state.items()})
            elif path.startswith("/src/"):
                self._file(state.src, unquote(path[5:]))
            elif path.startswith("/img/"):
                self._file(state.out, unquote(path[5:]))
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(length) or b"{}")
            try:
                handler = {
                    "/api/browse": self.browse, "/api/open": self.open, "/api/preview": self.preview,
                    "/api/process": self.process, "/api/label": self.label, "/api/export": self.export,
                }.get(path)
                if handler is None:
                    return self._json({"error": "not found"}, 404)
                handler(data)
            except Exception as e:
                self._json({"error": str(e)}, 400)

        def browse(self, data):
            self._json({"path": browse_folder(data.get("initial", ""))})

        def open(self, data):
            if state.job["running"]:
                raise ValueError("Идёт обработка, дождитесь окончания")
            state.open(data.get("src", ""), data.get("out", ""))
            self._json({"src": str(state.src or ""), "out": str(state.out), "sources": state.sources()})

        def preview(self, data):
            if not state.src:
                raise ValueError("Сначала откройте папку с фото")
            params, square, mode = params_from(data)
            img = imread(state.src / Path(data["name"]).name)
            t = time.time()
            dets = detect(img, params)
            cuts = []
            for box in dets[:PREVIEW_CUTS]:
                ok, buf = cv2.imencode(".png", cutout_on_white(img, box, square, mode))
                cuts.append("data:image/png;base64," + base64.b64encode(buf.tobytes()).decode())
            self._json({
                "width": img.shape[1], "height": img.shape[0], "seconds": time.time() - t,
                "boxes": [[round(v, 1) for v in d] for d in dets], "cuts": cuts,
            })

        def process(self, data):
            if not state.src or not state.out:
                raise ValueError("Сначала откройте папку с фото")
            if state.job["running"]:
                raise ValueError("Обработка уже идёт")
            stems = {p.stem for p in list_images(state.src)}
            old = [p.name for p in list_images(state.out) if (m := CROP_RE.fullmatch(p.name)) and m.group(1) in stems]
            labeled = sum(1 for n in old if n in state.store.labels)
            if old and not data.get("confirm"):
                return self._json({"need_confirm": True, "existing": len(old), "labeled": labeled})
            params, square, mode = params_from(data)
            state.job.update(running=True, done=0, total=len(stems), current="", found=0, error=None)
            threading.Thread(target=run_job, args=(state, state.src, state.out, params, square, mode),
                             daemon=True).start()
            self._json({"ok": True})

        def label(self, data):
            if not state.store:
                raise ValueError("Папка с яблоками не открыта")
            name = Path(data["name"]).name
            if not (state.out / name).is_file():
                raise ValueError(f"Нет файла {name}")
            state.store.set_label(name, data.get("label"))
            self._json({"ok": True})

        def export(self, data):
            if not state.store:
                raise ValueError("Папка с яблоками не открыта")
            if data.get("dir", "").strip():
                state.export_dir = Path(data["dir"]).expanduser().resolve()
            counts = state.store.export(state.export_dir)
            self._json({"ok": True, "dir": str(state.export_dir), "counts": counts})

    return Handler


PAGE = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Яблоки: HOG и разметка</title>
<style>
  :root { --bg:#14161a; --panel:#1d2026; --line:#2c3038; --text:#e6e8eb; --muted:#8a919c;
          --rot:#d9534f; --fresh:#4caf50; --garb:#9e9e9e; --accent:#5b9cf5; }
  * { box-sizing:border-box; }
  [hidden] { display:none !important; }
  body { margin:0; background:var(--bg); color:var(--text); font:14px/1.4 system-ui, sans-serif;
         height:100vh; display:flex; flex-direction:column; }
  header { display:flex; gap:12px; align-items:center; padding:8px 16px; background:var(--panel);
           border-bottom:1px solid var(--line); flex-wrap:wrap; }
  header h1 { font-size:16px; margin:0 12px 0 0; }
  .tab { border-radius:6px 6px 0 0; }
  .tab.on { border-color:var(--accent); background:var(--accent); color:#fff; }
  input, select, button { background:var(--bg); color:var(--text); border:1px solid var(--line); border-radius:6px;
                          padding:6px 10px; font:inherit; }
  button { cursor:pointer; } button:hover { border-color:var(--accent); }
  button:disabled { opacity:.5; cursor:default; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  kbd { background:var(--line); border-radius:4px; padding:1px 6px; font-size:12px; }
  .muted { color:var(--muted); }
  section { flex:1; min-height:0; display:flex; flex-direction:column; }
  .row { display:flex; gap:8px; align-items:center; padding:6px 16px; flex-wrap:wrap; }
  .row label { color:var(--muted); }
  .grow { flex:1; min-width:200px; }
  input[type=number] { width:80px; }
  .bar { flex:1; min-width:120px; height:8px; background:var(--line); border-radius:4px; overflow:hidden; display:flex; }
  .bar div { height:100%; transition:width .2s; }

  /* --- вырезание --- */
  .cutbody { flex:1; min-height:0; display:grid; grid-template-columns:220px 1fr; gap:8px; padding:0 16px; }
  #srcList { list-style:none; margin:0; padding:4px; overflow-y:auto; background:var(--panel); border:1px solid var(--line); border-radius:8px; }
  #srcList li { padding:5px 8px; border-radius:4px; cursor:pointer; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  #srcList li:hover { background:var(--line); } #srcList li.on { background:var(--accent); color:#fff; }
  .stagewrap { position:relative; min-height:0; background:#000; border-radius:8px; overflow:hidden; }
  #stage { position:absolute; inset:0; }
  #stageInfo { position:absolute; left:8px; top:8px; background:#000a; padding:4px 8px; border-radius:6px; }
  #cutsRow { display:flex; gap:6px; overflow-x:auto; padding:8px 16px; min-height:112px; }
  #cutsRow img { width:96px; height:96px; border-radius:6px; flex:none; background:#fff; }

  /* --- разметка --- */
  .labbody { flex:1; min-height:0; display:grid; grid-template-columns:1fr 300px; }
  .viewer { display:flex; flex-direction:column; align-items:center; gap:12px; padding:12px 16px; min-height:0; }
  .imgwrap { flex:1; min-height:0; width:100%; display:flex; align-items:center; justify-content:center;
             border:4px solid transparent; border-radius:10px; }
  .imgwrap.rotten { border-color:var(--rot); } .imgwrap.fresh { border-color:var(--fresh); } .imgwrap.garbage { border-color:var(--garb); }
  .imgwrap img { width:100%; height:100%; object-fit:contain; }
  .meta { color:var(--muted); text-align:center; } .meta b { color:var(--text); }
  .actions { display:flex; gap:10px; flex-wrap:wrap; justify-content:center; }
  .actions button { padding:10px 20px; font-size:16px; border-width:2px; }
  .b-rot { border-color:var(--rot); } .b-rot.on { background:var(--rot); }
  .b-fresh { border-color:var(--fresh); } .b-fresh.on { background:var(--fresh); }
  .b-garb { border-color:var(--garb); } .b-garb.on { background:var(--garb); }
  aside { border-left:1px solid var(--line); background:var(--panel); overflow-y:auto; padding:8px;
          display:grid; grid-template-columns:repeat(auto-fill, minmax(84px, 1fr)); grid-auto-rows:84px; gap:6px; align-content:start; }
  .thumb { border:3px solid var(--line); border-radius:6px; overflow:hidden; cursor:pointer; background:#fff; }
  .thumb img { width:100%; height:100%; object-fit:cover; }
  .thumb.rotten { border-color:var(--rot); } .thumb.fresh { border-color:var(--fresh); } .thumb.garbage { border-color:var(--garb); }
  .thumb.cur { outline:3px solid var(--accent); outline-offset:1px; }
  footer { padding:6px 16px; color:var(--muted); background:var(--panel); border-top:1px solid var(--line); font-size:12px; }
  .toast { position:fixed; right:16px; bottom:40px; background:var(--panel); border:1px solid var(--line);
           padding:10px 14px; border-radius:8px; display:none; white-space:pre-line; max-width:420px; z-index:10; }
  @media (max-width:760px) {
    .cutbody { grid-template-columns:1fr; grid-template-rows:120px 1fr; }
    .labbody { grid-template-columns:1fr; grid-template-rows:1fr 160px; }
    aside { border-left:none; border-top:1px solid var(--line); }
  }
</style>
</head>
<body>
<header>
  <h1>Яблоки</h1>
  <button class="tab on" data-tab="cut">1. Вырезание (HOG)</button>
  <button class="tab" data-tab="label">2. Разметка</button>
</header>

<section id="tab-cut">
  <div class="row">
    <label>Фото куч</label><input id="src" class="grow" placeholder="папка с фотографиями куч яблок">
    <button data-browse="src">Обзор…</button>
    <label>Яблоки</label><input id="out" class="grow" placeholder="куда сохранять (по умолчанию &lt;папка&gt;_apples)">
    <button data-browse="out">Обзор…</button>
    <button id="openBtn" class="primary">Открыть</button>
  </div>
  <div class="row">
    <label>Квадрат, px</label><input id="square" type="number" min="8" step="16">
    <label>Мин. яблоко, px</label><input id="min_size" type="number" min="8" step="2">
    <label>Макс. яблоко, px</label><input id="max_size" type="number" min="0" step="2" title="0 = половина меньшей стороны фото">
    <label>Порог</label><input id="threshold" type="number" min="-1" max="1" step="0.05">
    <label>NMS</label><input id="nms_iou" type="number" min="0.05" max="0.9" step="0.05">
    <label>Вырезание</label>
    <select id="mode"><option value="grabcut">grabcut</option><option value="ellipse">эллипс</option><option value="box">рамка</option></select>
    <button id="previewBtn">Предпросмотр</button>
  </div>
  <div class="row muted" style="padding-top:0">Главное — размер яблока: задайте мин./макс. так, чтобы рамки совпадали с яблоками. Порог — насколько окно должно быть похоже на круг: ниже — больше яблок, но и больше мусора.</div>
  <div class="cutbody">
    <ul id="srcList"></ul>
    <div class="stagewrap"><canvas id="stage"></canvas><div id="stageInfo">Откройте папку с фото</div></div>
  </div>
  <div id="cutsRow"></div>
  <div class="row" style="padding-bottom:10px">
    <button id="runBtn" class="primary">Обработать всю папку</button>
    <div class="bar"><div id="jobBar" style="background:var(--accent);width:0"></div></div>
    <span id="jobText" class="muted"></span>
  </div>
</section>

<section id="tab-label" hidden>
  <div class="row">
    <div class="bar"><div id="pRot" style="background:var(--rot)"></div><div id="pFresh" style="background:var(--fresh)"></div><div id="pGarb" style="background:var(--garb)"></div></div>
    <span id="stats" class="muted"></span>
    <select id="filter">
      <option value="all">Все</option><option value="none">Неразмеченные</option>
      <option value="rotten">Гнилые</option><option value="fresh">Не гнилые</option><option value="garbage">Мусор</option>
    </select>
    <label>Экспорт в</label><input id="exportDir" class="grow">
    <button id="exportBtn">Экспорт датасета</button>
  </div>
  <div class="labbody">
    <div class="viewer">
      <div class="imgwrap" id="wrap"><img id="img" alt=""></div>
      <div class="meta" id="meta"></div>
      <div class="actions">
        <button class="b-rot" id="bRot">Гнилое <kbd>1</kbd></button>
        <button class="b-fresh" id="bFresh">Не гнилое <kbd>2</kbd></button>
        <button class="b-garb" id="bGarb">Мусор <kbd>3</kbd></button>
        <button id="bClear">Снять <kbd>0</kbd></button>
      </div>
    </div>
    <aside id="grid"></aside>
  </div>
  <footer>
    <kbd>1</kbd>/<kbd>R</kbd> гнилое · <kbd>2</kbd>/<kbd>F</kbd> не гнилое · <kbd>3</kbd>/<kbd>G</kbd> мусор (не яблоко, половина, два яблока) ·
    <kbd>0</kbd> снять · <kbd>←</kbd>/<kbd>→</kbd> навигация · <kbd>Ctrl+Z</kbd> отменить · после метки — следующее
  </footer>
</section>
<div class="toast" id="toast"></div>

<script>
const $ = id => document.getElementById(id);
const NAMES = { rotten: "гнилое", fresh: "не гнилое", garbage: "мусор" };
const LABELS = ["rotten", "fresh", "garbage"];
const PARAMS = ["square", "min_size", "max_size", "threshold", "nms_iou"];

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

// ---------------------------------------------------------------- вкладки
let tab = "cut";
function showTab(t) {
  tab = t;
  document.querySelectorAll(".tab").forEach(b => b.classList.toggle("on", b.dataset.tab === t));
  $("tab-cut").hidden = t !== "cut";
  $("tab-label").hidden = t !== "label";
  if (t === "label") loadItems(); else drawStage();
}
document.querySelectorAll(".tab").forEach(b => b.onclick = () => showTab(b.dataset.tab));

// ---------------------------------------------------------------- вырезание
let sources = [], curSrc = null, stageImg = null, stageBoxes = [], previewSeq = 0;

function params() {
  const p = { mode: $("mode").value };
  PARAMS.forEach(k => p[k] = Number($(k).value));
  return p;
}

async function init() {
  const s = await api("/api/state");
  PARAMS.forEach(k => $(k).value = s.defaults[k]);
  $("src").value = s.src; $("out").value = s.out; $("exportDir").value = s.export;
  if (s.out) setSources(s.sources);
  if (s.job.running) pollJob();
}

function setSources(list) {
  sources = list;
  renderSources();
  if (sources.length) selectSource(sources.includes(curSrc) ? curSrc : sources[0]);
  else { curSrc = null; stageImg = null; stageBoxes = []; drawStage(); $("stageInfo").textContent = "В папке нет фото"; }
}

async function openFolders() {
  try {
    const r = await api("/api/open", { src: $("src").value, out: $("out").value });
    $("src").value = r.src; $("out").value = r.out;
    setSources(r.sources);
    toast(`Фото куч: ${r.sources.length}\nЯблоки сохраняются в ${r.out}`);
  } catch (e) { toast(e.message); }
}

document.querySelectorAll("[data-browse]").forEach(b => b.onclick = async () => {
  const input = $(b.dataset.browse);
  b.disabled = true;
  try {
    const r = await api("/api/browse", { initial: input.value });
    if (r.path) {
      input.value = r.path;
      if (b.dataset.browse === "src") { $("out").value = ""; await openFolders(); }
      else if ($("src").value || tab === "label") await openFolders();
    }
  } catch (e) { toast(e.message); }
  b.disabled = false;
});
$("openBtn").onclick = openFolders;
["src", "out"].forEach(id => $(id).addEventListener("keydown", e => { if (e.key === "Enter") openFolders(); }));

function renderSources() {
  const ul = $("srcList"); ul.innerHTML = "";
  sources.forEach(n => {
    const li = document.createElement("li");
    li.textContent = n; li.title = n;
    if (n === curSrc) li.className = "on";
    li.onclick = () => selectSource(n);
    ul.appendChild(li);
  });
}

function selectSource(n) {
  curSrc = n; renderSources();
  stageBoxes = [];
  stageImg = new Image();
  stageImg.onload = drawStage;
  stageImg.src = "/src/" + encodeURIComponent(n) + "?t=" + Date.now();
  preview();
}

async function preview() {
  if (!curSrc) return;
  const seq = ++previewSeq;
  $("stageInfo").textContent = `${curSrc}: ищу яблоки…`;
  try {
    const r = await api("/api/preview", { name: curSrc, ...params() });
    if (seq !== previewSeq) return;
    stageBoxes = r.boxes; drawStage();
    $("stageInfo").textContent = `${curSrc}: найдено ${r.boxes.length} (${r.seconds.toFixed(1)} с)`;
    $("cutsRow").innerHTML = r.cuts.map(u => `<img src="${u}">`).join("");
  } catch (e) { if (seq === previewSeq) $("stageInfo").textContent = e.message; }
}
$("previewBtn").onclick = preview;
[...PARAMS, "mode"].forEach(k => $(k).addEventListener("change", preview));

function drawStage() {
  const c = $("stage"), wrap = c.parentElement;
  const W = wrap.clientWidth, H = wrap.clientHeight;
  c.width = W; c.height = H;
  const g = c.getContext("2d");
  g.clearRect(0, 0, W, H);
  if (!stageImg || !stageImg.naturalWidth) return;
  const k = Math.min(W / stageImg.naturalWidth, H / stageImg.naturalHeight);
  const w = stageImg.naturalWidth * k, h = stageImg.naturalHeight * k, ox = (W - w) / 2, oy = (H - h) / 2;
  g.drawImage(stageImg, ox, oy, w, h);
  g.strokeStyle = "#39ff39"; g.lineWidth = 2;
  for (const [x1, y1, x2, y2] of stageBoxes) g.strokeRect(ox + x1 * k, oy + y1 * k, (x2 - x1) * k, (y2 - y1) * k);
}
window.addEventListener("resize", () => tab === "cut" && drawStage());

async function runAll(confirmed = false) {
  try {
    const r = await api("/api/process", { ...params(), confirm: confirmed });
    if (r.need_confirm) {
      const msg = `В папке для яблок уже есть ${r.existing} вырезок из этих фото` +
        (r.labeled ? `, из них ${r.labeled} размечено` : "") +
        `.\nОни будут удалены и вырезаны заново${r.labeled ? ", их разметка сбросится" : ""}. Продолжить?`;
      if (confirm(msg)) return runAll(true);
      return;
    }
    pollJob();
  } catch (e) { toast(e.message); }
}
$("runBtn").onclick = () => runAll(false);

let pollTimer = null;
function pollJob() {
  clearInterval(pollTimer);
  $("runBtn").disabled = true;
  pollTimer = setInterval(async () => {
    const j = await api("/api/job");
    $("jobBar").style.width = (j.total ? 100 * j.done / j.total : 0) + "%";
    if (j.running) { $("jobText").textContent = `[${j.done}/${j.total}] ${j.current}  яблок: ${j.found}`; return; }
    clearInterval(pollTimer);
    $("runBtn").disabled = false;
    if (j.error) { $("jobText").textContent = "Ошибка: " + j.error; toast("Ошибка: " + j.error); }
    else {
      $("jobText").textContent = `Готово: вырезано ${j.found} яблок из ${j.total} фото`;
      toast(`Готово: вырезано ${j.found} яблок.\nОткройте вкладку «2. Разметка».`);
    }
  }, 400);
}

// ---------------------------------------------------------------- разметка
let items = [], view = [], pos = 0, history = [];

async function loadItems() {
  try {
    const r = await api("/api/items");
    items = r.items;
    applyFilter(true);
  } catch (e) { toast(e.message); }
}

function applyFilter(jumpToFirstUnlabeled) {
  const f = $("filter").value, curName = view[pos]?.name;
  view = items.filter(it => f === "all" || (f === "none" ? !it.label : it.label === f));
  const i = jumpToFirstUnlabeled ? view.findIndex(it => !it.label) : view.findIndex(it => it.name === curName);
  pos = i >= 0 ? i : Math.min(pos, Math.max(view.length - 1, 0));
  renderGrid(); render();
}
$("filter").onchange = () => applyFilter(false);

function imgUrl(it) { return "/img/" + encodeURIComponent(it.name) + "?v=" + it.v; }

function renderGrid() {
  const g = $("grid"); g.innerHTML = "";
  view.forEach((it, i) => {
    const d = document.createElement("div");
    d.className = "thumb"; d.title = it.name;
    d.innerHTML = `<img loading="lazy" src="${imgUrl(it)}">`;
    d.onclick = () => { pos = i; render(); };
    g.appendChild(d);
  });
}

function render() {
  const thumbs = $("grid").children;
  view.forEach((it, i) => {
    if (thumbs[i]) thumbs[i].className = "thumb" + (it.label ? " " + it.label : "") + (i === pos ? " cur" : "");
  });
  thumbs[pos]?.scrollIntoView({ block: "nearest" });

  const total = items.length || 1;
  const n = l => items.filter(x => x.label === l).length;
  const nRot = n("rotten"), nFresh = n("fresh"), nGarb = n("garbage");
  $("pRot").style.width = (100 * nRot / total) + "%";
  $("pFresh").style.width = (100 * nFresh / total) + "%";
  $("pGarb").style.width = (100 * nGarb / total) + "%";
  $("stats").textContent = `${nRot + nFresh + nGarb}/${items.length} · гнилых ${nRot} · не гнилых ${nFresh} · мусор ${nGarb}`;

  const it = view[pos];
  ["bRot", "bFresh", "bGarb"].forEach((id, k) => $(id).classList.toggle("on", it?.label === LABELS[k]));
  if (!it) {
    $("img").removeAttribute("src"); $("wrap").className = "imgwrap";
    $("meta").textContent = items.length ? "Нет картинок в этом фильтре" : "Нет вырезанных яблок: сначала обработайте папку на вкладке «1. Вырезание»";
    return;
  }
  $("img").src = imgUrl(it);
  $("wrap").className = "imgwrap" + (it.label ? " " + it.label : "");
  $("meta").innerHTML = `<b>${pos + 1}/${view.length}</b> · ${it.name}` + (it.source ? ` · из ${it.source}` : "") +
    `<br>метка: <b>${it.label ? NAMES[it.label] : "—"}</b>`;
}

async function setLabel(label, advance = true) {
  const it = view[pos];
  if (!it) return;
  history.push({ name: it.name, prev: it.label ?? null });
  it.label = label;
  if (advance && pos < view.length - 1) pos++;
  render();
  try { await api("/api/label", { name: it.name, label }); }
  catch (e) { toast("Ошибка сохранения: " + e.message); }
}

function undo() {
  const h = history.pop();
  if (!h) return;
  const it = items.find(x => x.name === h.name);
  if (!it) return;
  it.label = h.prev;
  api("/api/label", { name: h.name, label: h.prev }).catch(e => toast(e.message));
  const i = view.findIndex(x => x.name === h.name);
  if (i >= 0) pos = i;
  render();
}

$("bRot").onclick = () => setLabel("rotten");
$("bFresh").onclick = () => setLabel("fresh");
$("bGarb").onclick = () => setLabel("garbage");
$("bClear").onclick = () => setLabel(null, false);
$("exportBtn").onclick = async () => {
  try {
    const r = await api("/api/export", { dir: $("exportDir").value });
    $("exportDir").value = r.dir;
    toast(`Сохранено в ${r.dir}\\train\n` + Object.entries(r.counts).map(([k, v]) => `${k}: ${v}`).join("\n"), 6000);
  } catch (e) { toast(e.message); }
};

document.addEventListener("keydown", e => {
  if (tab !== "label" || ["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
  const k = e.key.toLowerCase();
  if ((e.ctrlKey || e.metaKey) && (k === "z" || k === "я")) { e.preventDefault(); return undo(); }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (k === "1" || k === "r" || k === "к") setLabel("rotten");
  else if (k === "2" || k === "f" || k === "а") setLabel("fresh");
  else if (k === "3" || k === "g" || k === "п") setLabel("garbage");
  else if (k === "0" || k === "delete") setLabel(null, false);
  else if (k === "arrowright" || k === " ") { e.preventDefault(); if (pos < view.length - 1) { pos++; render(); } }
  else if (k === "arrowleft" || k === "backspace") { e.preventDefault(); if (pos > 0) { pos--; render(); } }
  else return;
  document.activeElement?.blur?.();
});

init();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description="HOG-вырезание яблок + разметка fresh / rotten / garbage")
    ap.add_argument("--src", default="", help="папка с фото куч (можно выбрать в приложении)")
    ap.add_argument("--out", default="", help="папка для вырезанных яблок (по умолчанию <src>_apples)")
    ap.add_argument("--export", type=Path, default=ROOT / "dataset_hog", help="куда экспортировать разметку")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    state = State(args.export.resolve())
    if args.src or args.out:
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
