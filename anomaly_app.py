"""Веб-приложение: поиск испорченных яблок на фото.

Конвейер для загруженной фотографии:
  1. YOLO11m-seg находит яблоки и сразу даёт маску каждого (класс COCO "apple");
  2. каждое яблоко вырезается по своей маске и кладётся на белый квадрат —
     так же выглядят фото, на которых обучен PatchCore (main_dataset);
  3. PatchCore (models/patchcore.pt) ставит каждому яблоку скор аномальности;
  4. на исходной фотографии обводятся яблоки: красным — аномалии (скор выше порога),
     зелёным — нормальные. Результат можно скачать.

Запуск (сначала обучить модель: python patchcore.py fit):
    python anomaly_app.py            # откроет браузер на http://127.0.0.1:8766
    python anomaly_app.py --model models/patchcore.pt --yolo models/yolo11m-seg.pt --port 8766

Веса yolo11m-seg.pt скачиваются в models/ автоматически при первом запуске.
Можно указать свою дообученную seg-модель через --yolo и её классы через --classes.
"""

import argparse
import base64
import io
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import torch
from PIL import Image

from patchcore import PatchCore, overlay, preprocess
from yolo_cut import DEFAULT_WEIGHTS, DEFAULTS, Segmenter, cutout_on_white, params_from

ROOT = Path(__file__).resolve().parent
BATCH = 16


def png_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class Analyzer:
    def __init__(self, model_path: Path, yolo_path: Path, class_names: list[str]):
        self.model = PatchCore.load(model_path)
        self.seg = Segmenter(yolo_path, class_names)
        self.lock = threading.Lock()  # одна видеокарта — запросы по очереди
        print(f"PatchCore {model_path}: порог {self.model.threshold:.3f}, устройство {self.model.device}")

    def analyze(self, img_bgr: np.ndarray, p: dict) -> dict:
        size = self.model.image_size
        t0 = time.time()
        dets = self.seg.segment(img_bgr, p)
        t_yolo = time.time() - t0

        apples = []
        for d in dets:
            cut = cutout_on_white(img_bgr, d["mask"], size)
            apples.append({"box": [round(v, 1) for v in d["box"]], "conf": round(d["conf"], 3),
                           "cls": d["cls"], "contour": d["contour"],
                           "cut": Image.fromarray(cv2.cvtColor(cut, cv2.COLOR_BGR2RGB))})

        t1 = time.time()
        with self.lock:
            for i in range(0, len(apples), BATCH):
                part = apples[i:i + BATCH]
                x = torch.stack([preprocess(a["cut"], size, self.model.clahe) for a in part])
                scores, maps = self.model.predict(x)
                for a, s, m in zip(part, scores, maps):
                    a["score"] = round(s.item(), 4)
                    a["heat"] = png_data_url(overlay(a["cut"], m, size, vmax=self.model.threshold * 1.5))
        t_pc = time.time() - t1

        for a in apples:
            a["cut"] = png_data_url(a["cut"])
        return {"width": img_bgr.shape[1], "height": img_bgr.shape[0],
                "threshold": self.model.threshold, "apples": apples,
                "seconds": {"yolo": round(t_yolo, 2), "patchcore": round(t_pc, 2)}}


def make_handler(an: Analyzer):
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

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/api/defaults":
                self._json({**DEFAULTS, "anomaly_threshold": an.model.threshold})
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            url = urlparse(self.path)
            if url.path != "/api/analyze":
                return self._json({"error": "not found"}, 404)
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("Не удалось открыть изображение")
                q = parse_qs(url.query)
                self._json(an.analyze(img, params_from(lambda k, d: q.get(k, [d])[0])))
            except Exception as e:
                self._json({"error": str(e)}, 400)

    return Handler


PAGE = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Яблоки: аномалии</title>
<style>
  :root { --bg:#14161a; --panel:#1d2026; --line:#2c3038; --text:#e6e8eb; --muted:#8a919c;
          --bad:#ff3b30; --ok:#39d353; --accent:#5b9cf5; }
  * { box-sizing:border-box; }
  [hidden] { display:none !important; }
  body { margin:0; background:var(--bg); color:var(--text); font:14px/1.4 system-ui, sans-serif;
         height:100vh; display:flex; flex-direction:column; }
  header { display:flex; gap:12px; align-items:center; padding:8px 16px; background:var(--panel);
           border-bottom:1px solid var(--line); flex-wrap:wrap; }
  header h1 { font-size:16px; margin:0 12px 0 0; }
  input, select, button { background:var(--bg); color:var(--text); border:1px solid var(--line); border-radius:6px;
                          padding:6px 10px; font:inherit; }
  button { cursor:pointer; } button:hover { border-color:var(--accent); }
  button:disabled { opacity:.5; cursor:default; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  input[type=number] { width:80px; }
  .row { display:flex; gap:8px; align-items:center; padding:6px 16px; flex-wrap:wrap; }
  .row label { color:var(--muted); }
  .muted { color:var(--muted); }
  main { flex:1; min-height:0; display:grid; grid-template-columns:1fr 340px; }
  .stagewrap { position:relative; min-height:0; background:#000; margin:0 0 8px 16px; border-radius:8px; overflow:hidden; }
  #stage { position:absolute; inset:0; }
  #drop { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; text-align:center;
          color:var(--muted); font-size:16px; border:2px dashed var(--line); border-radius:8px; padding:20px; }
  .stagewrap.over #drop { border-color:var(--accent); color:var(--text); }
  #info { position:absolute; left:8px; top:8px; background:#000b; padding:4px 10px; border-radius:6px; }
  aside { overflow-y:auto; padding:0 16px 8px 8px; display:flex; flex-direction:column; gap:8px; }
  .card { display:flex; gap:8px; align-items:center; background:var(--panel); border:2px solid var(--line);
          border-radius:8px; padding:6px; cursor:pointer; }
  .card.bad { border-color:var(--bad); } .card.ok { border-color:#2c5a35; }
  .card.cur { outline:2px solid var(--accent); outline-offset:1px; }
  .card img { width:88px; height:88px; border-radius:4px; background:#fff; flex:none; }
  .card .t b { font-size:15px; }
  .verdict.bad { color:var(--bad); } .verdict.ok { color:var(--ok); }
  #thr { width:180px; }
  .toast { position:fixed; right:16px; bottom:16px; background:var(--panel); border:1px solid var(--line);
           padding:10px 14px; border-radius:8px; display:none; white-space:pre-line; max-width:420px; z-index:10; }
  @media (max-width:800px) {
    main { grid-template-columns:1fr; grid-template-rows:60vh auto; }
    .stagewrap { margin:0 16px 8px; }
    aside { padding:0 16px 8px; }
  }
</style>
</head>
<body>
<header>
  <h1>Яблоки: поиск испорченных</h1>
  <input type="file" id="file" accept="image/*" hidden>
  <button id="pick" class="primary">Загрузить фото…</button>
  <button id="rerun" disabled>Пересчитать</button>
  <button id="save" disabled>Скачать результат</button>
  <span id="summary" class="muted"></span>
</header>
<div class="row">
  <label title="минимальная уверенность YOLO: ниже — больше яблок, но и больше ложных">Уверенность YOLO</label><input id="conf" type="number" min="0.01" max="1" step="0.05">
  <label title="рамки, перекрывающиеся сильнее, считаются одним яблоком">NMS</label><input id="iou" type="number" min="0.05" max="0.95" step="0.05">
  <label title="до какого размера YOLO уменьшает фото; больше — лучше видны мелкие яблоки, но медленнее">Размер для YOLO, px</label><input id="imgsz" type="number" min="320" max="4096" step="32">
  <label title="маски меньше этой площади отбрасываются">Мин. площадь, px²</label><input id="min_area" type="number" min="0" step="100">
  <label>Порог аномалии</label><input id="thr" type="range" step="0.01"><span id="thrVal"></span>
  <button id="thrReset" title="порог из модели (p99 по нормальной валидации)">сброс</button>
  <label><input type="checkbox" id="showOk" checked> нормальные</label>
</div>
<main>
  <div class="stagewrap" id="wrap">
    <canvas id="stage"></canvas>
    <div id="drop">Перетащите сюда фото с яблоками<br>или нажмите «Загрузить фото…»</div>
    <div id="info" hidden></div>
  </div>
  <aside id="list"></aside>
</main>
<div class="toast" id="toast"></div>

<script>
const $ = id => document.getElementById(id);
const PARAMS = ["conf", "iou", "imgsz", "min_area"];
let file = null, photo = null, res = null, thr = 0, modelThr = 0, cur = -1;

function toast(text, ms = 4000) {
  const t = $("toast"); t.textContent = text; t.style.display = "block";
  clearTimeout(t._h); t._h = setTimeout(() => t.style.display = "none", ms);
}

async function init() {
  const d = await (await fetch("/api/defaults")).json();
  PARAMS.forEach(k => $(k).value = d[k]);
  modelThr = d.anomaly_threshold;
  setThr(modelThr);
}

function setThr(v) {
  thr = v;
  const s = $("thr");
  s.min = (modelThr * 0.3).toFixed(2); s.max = (modelThr * 2.5).toFixed(2);
  s.value = v; $("thrVal").textContent = v.toFixed(2);
  render();
}
$("thr").oninput = e => setThr(Number(e.target.value));
$("thrReset").onclick = () => setThr(modelThr);
$("showOk").onchange = render;

$("pick").onclick = () => $("file").click();
$("file").onchange = e => e.target.files[0] && load(e.target.files[0]);
const wrap = $("wrap");
wrap.addEventListener("dragover", e => { e.preventDefault(); wrap.classList.add("over"); });
wrap.addEventListener("dragleave", () => wrap.classList.remove("over"));
wrap.addEventListener("drop", e => {
  e.preventDefault(); wrap.classList.remove("over");
  const f = [...e.dataTransfer.files].find(f => f.type.startsWith("image/"));
  if (f) load(f);
});

function load(f) {
  file = f; res = null; cur = -1;
  photo = new Image();
  photo.onload = () => { $("drop").hidden = true; render(); analyze(); };
  photo.src = URL.createObjectURL(f);
}

async function analyze() {
  if (!file) return;
  $("rerun").disabled = $("pick").disabled = $("save").disabled = true;
  const info = $("info"); info.hidden = false; info.textContent = "Ищу яблоки (YOLO) и проверяю PatchCore…";
  const q = new URLSearchParams();
  PARAMS.forEach(k => q.set(k, $(k).value));
  try {
    const r = await fetch("/api/analyze?" + q, { method: "POST", body: file });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    res = j; cur = -1;
    info.textContent = `${file.name}: яблок ${j.apples.length} · YOLO ${j.seconds.yolo} с · PatchCore ${j.seconds.patchcore} с`;
    $("save").disabled = false;
  } catch (e) { info.textContent = "Ошибка: " + e.message; toast(e.message); }
  $("rerun").disabled = $("pick").disabled = false;
  render();
}
$("rerun").onclick = analyze;
PARAMS.forEach(k => $(k).addEventListener("change", analyze));

const isBad = a => a.score > thr;

// рисует фото с обводкой яблок; k — масштаб фото -> холст
function drawResult(g, k, ox, oy, highlight) {
  g.drawImage(photo, ox, oy, photo.naturalWidth * k, photo.naturalHeight * k);
  if (!res) return;
  const lw = Math.max(2, Math.round(Math.max(photo.naturalWidth, photo.naturalHeight) * k / 400));
  const font = Math.max(12, lw * 6);
  res.apples.forEach((a, i) => {
    const bad = isBad(a);
    if (!bad && !$("showOk").checked && i !== highlight) return;
    const color = bad ? "#ff3b30" : "#39d353";
    g.lineWidth = bad ? lw * 1.5 : lw;
    g.strokeStyle = i === highlight ? "#5b9cf5" : color;
    g.beginPath();
    const pts = a.contour.length > 2 ? a.contour : [[a.box[0], a.box[1]], [a.box[2], a.box[1]], [a.box[2], a.box[3]], [a.box[0], a.box[3]]];
    pts.forEach(([x, y], j) => j ? g.lineTo(ox + x * k, oy + y * k) : g.moveTo(ox + x * k, oy + y * k));
    g.closePath(); g.stroke();
    if (bad || i === highlight) {
      const text = `${i + 1}: ${a.score.toFixed(2)}`;
      g.font = `bold ${font}px system-ui, sans-serif`;
      const w = g.measureText(text).width + 8, x = ox + a.box[0] * k, y = Math.max(oy + a.box[1] * k - font - 6, 0);
      g.fillStyle = i === highlight ? "#5b9cf5" : color; g.fillRect(x, y, w, font + 6);
      g.fillStyle = "#fff"; g.fillText(text, x + 4, y + font);
    }
  });
}

function render() {
  const c = $("stage"), W = wrap.clientWidth, H = wrap.clientHeight;
  c.width = W; c.height = H;
  const g = c.getContext("2d");
  g.clearRect(0, 0, W, H);
  if (photo && photo.naturalWidth) {
    const k = Math.min(W / photo.naturalWidth, H / photo.naturalHeight);
    drawResult(g, k, (W - photo.naturalWidth * k) / 2, (H - photo.naturalHeight * k) / 2, cur);
  }
  renderList();
}
window.addEventListener("resize", render);

function renderList() {
  const list = $("list");
  if (!res) { list.innerHTML = ""; $("summary").textContent = ""; return; }
  const nBad = res.apples.filter(isBad).length;
  $("summary").innerHTML = `найдено яблок: <b>${res.apples.length}</b> · испорченных: <b style="color:var(--bad)">${nBad}</b>`;
  const order = res.apples.map((a, i) => i).sort((i, j) => res.apples[j].score - res.apples[i].score);
  list.innerHTML = "";
  if (!order.length) { list.innerHTML = `<div class="muted">Яблоки не найдены — попробуйте снизить уверенность YOLO или увеличить размер для YOLO.</div>`; return; }
  order.forEach(i => {
    const a = res.apples[i], bad = isBad(a);
    const d = document.createElement("div");
    d.className = "card " + (bad ? "bad" : "ok") + (i === cur ? " cur" : "");
    d.innerHTML = `<img src="${a.cut}"><img src="${a.heat}" title="тепловая карта: красное — непохожие на норму участки">
      <div class="t"><b>#${i + 1}</b><br><span class="verdict ${bad ? "bad" : "ok"}">${bad ? "аномалия" : "норма"}</span><br>
      <span class="muted">скор ${a.score.toFixed(2)}<br>YOLO ${a.conf.toFixed(2)}</span></div>`;
    d.onclick = () => { cur = cur === i ? -1 : i; render(); };
    list.appendChild(d);
  });
}

$("save").onclick = () => {
  const c = document.createElement("canvas");
  c.width = photo.naturalWidth; c.height = photo.naturalHeight;
  drawResult(c.getContext("2d"), 1, 0, 0, -1);
  const a = document.createElement("a");
  a.download = file.name.replace(/\.[^.]+$/, "") + "_anomalies.png";
  a.href = c.toDataURL("image/png");
  a.click();
};

init();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description="YOLO11-seg + PatchCore: поиск испорченных яблок на фото")
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "patchcore.pt")
    ap.add_argument("--yolo", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--classes", default="apple", help="классы YOLO через запятую, например apple,orange")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"Нет модели {args.model}. Сначала обучите: python patchcore.py fit")
    analyzer = Analyzer(args.model, args.yolo, args.classes.split(","))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(analyzer))
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
