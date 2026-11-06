"""Веб-приложение: вырезание яблок из папки фото через YOLO11m-seg + отбор плохих вырезок.

  1. Выбрать папку с фото -> «Вырезать все фото»: каждое фото проходит через yolo11m-seg,
     каждое найденное яблоко вырезается по маске и кладётся на белый квадрат.
     Вырезки складываются в рабочую папку cuts_review/<папка>/ (+ cuts.csv: откуда вырезано).
  2. Кликом отметить вырезки, которые не понравились (обрезано, два яблока, не яблоко...).
     Отметки сразу сохраняются (rejected.json), их можно менять и после перезапуска.
  3. «Сохранить в train_dataset»: все НЕ отмеченные вырезки копируются в train_dataset/.

Запуск:
    python cut_app.py                  # откроет браузер на http://127.0.0.1:8767
    python cut_app.py --src raw/apple/fresh --out train_dataset
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

from yolo_cut import DEFAULT_WEIGHTS, DEFAULTS, Segmenter, cutout_on_white, params_from

ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
SQUARE_SIZE = 256
WORK_ROOT = ROOT / "cuts_review"


def imread(path: Path) -> np.ndarray | None:
    # cv2.imread не открывает пути с кириллицей на Windows
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def imwrite(path: Path, img: np.ndarray):
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise ValueError(f"не удалось сохранить {path}")
    buf.tofile(str(path))


def list_sources(src: Path, recursive: bool) -> list[Path]:
    it = src.rglob("*") if recursive else src.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def work_dir_for(src: Path) -> Path:
    # имя папки + короткий хэш пути: у двух разных папок "fresh" не смешаются вырезки
    h = hashlib.md5(str(src).lower().encode()).hexdigest()[:6]
    return WORK_ROOT / f"{src.name}_{h}"


class Review:
    """Вырезки одной исходной папки и отметки «плохая»."""

    def __init__(self, src: Path):
        self.src = src
        self.dir = work_dir_for(src)
        self.lock = threading.Lock()
        self.rejected: set[str] = set()
        f = self.dir / "rejected.json"
        if f.exists():
            self.rejected = set(json.loads(f.read_text(encoding="utf-8")))

    def items(self) -> list[dict]:
        f = self.dir / "cuts.csv"
        if not f.exists():
            return []
        with f.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        # во время вырезания последняя строка может быть недописана — такие пропускаем
        return [{"name": r["crop"], "source": r["source"], "conf": float(r["conf"]),
                 "bad": r["crop"] in self.rejected} for r in rows
                if r.get("y2") and (self.dir / r["crop"]).is_file()]

    def mark(self, names: list[str], bad: bool):
        with self.lock:
            (self.rejected.update if bad else self.rejected.difference_update)(names)
            tmp = self.dir / "rejected.tmp"
            tmp.write_text(json.dumps(sorted(self.rejected), ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.dir / "rejected.json")

    def save(self, out: Path) -> dict:
        out.mkdir(parents=True, exist_ok=True)
        items = self.items()
        good = [it["name"] for it in items if not it["bad"]]
        bad = [it["name"] for it in items if it["bad"]]
        for n in good:
            shutil.copy2(self.dir / n, out / n)
        # вырезки, которые сохранили раньше, а потом отметили плохими, из датасета убираем
        removed = 0
        for n in bad:
            if (out / n).is_file():
                (out / n).unlink()
                removed += 1
        return {"saved": len(good), "skipped": len(bad), "removed": removed, "dir": str(out)}


class State:
    def __init__(self, seg: Segmenter, out: Path):
        self.seg = seg
        self.out = out
        self.review: Review | None = None
        self.job = {"running": False, "done": 0, "total": 0, "current": "", "found": 0, "error": None}

    def open(self, src: str) -> Review:
        p = Path(src.strip()).expanduser()
        if not p.is_dir():
            raise ValueError(f"Папка не найдена: {p}")
        self.review = Review(p.resolve())
        return self.review

    def run(self, review: Review, params: dict, square: int, recursive: bool):
        job = self.job
        try:
            files = list_sources(review.src, recursive)
            job.update(total=len(files))
            # вырезаем заново: старые вырезки и отметки этой папки удаляются
            if review.dir.exists():
                shutil.rmtree(review.dir)
            review.dir.mkdir(parents=True)
            review.rejected = set()
            with (review.dir / "cuts.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["crop", "source", "conf", "x1", "y1", "x2", "y2"])
                for i, path in enumerate(files):
                    rel = path.relative_to(review.src)
                    job.update(done=i, current=rel.as_posix())
                    img = imread(path)
                    if img is None:
                        continue
                    # в подпапках бывают одинаковые имена -> в имя вырезки идёт относительный путь
                    stem = "__".join(rel.with_suffix("").parts)
                    for k, d in enumerate(self.seg.segment(img, params)):
                        name = f"{stem}_{k:03d}.png"
                        imwrite(review.dir / name, cutout_on_white(img, d["mask"], square))
                        w.writerow([name, rel.as_posix(), round(d["conf"], 3), *(round(v) for v in d["box"])])
                        job["found"] += 1
                    fh.flush()
            job.update(done=len(files), current="")
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
        "print(filedialog.askdirectory(initialdir=sys.argv[1] or None, mustexist=True) or '')\n"
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

        def do_GET(self):
            path = urlparse(self.path).path
            r = state.review
            if path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._json({"src": str(r.src) if r else "", "out": str(state.out), "job": state.job,
                            "defaults": {**DEFAULTS, "square": SQUARE_SIZE}})
            elif path == "/api/job":
                self._json(state.job)
            elif path == "/api/items":
                self._json({"items": r.items() if r else []})
            elif path.startswith("/img/"):
                f = r.dir / Path(unquote(path[5:])).name if r else None
                if not f or f.suffix.lower() != ".png" or not f.is_file():
                    return self._send(404, b"not found", "text/plain")
                self._send(200, f.read_bytes(), "image/png")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            path = urlparse(self.path).path
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            try:
                handler = {"/api/browse": self.browse, "/api/open": self.open, "/api/process": self.process,
                           "/api/mark": self.mark, "/api/save": self.save}.get(path)
                if handler is None:
                    return self._json({"error": "not found"}, 404)
                handler(data)
            except Exception as e:
                self._json({"error": str(e)}, 400)

        def browse(self, data):
            self._json({"path": browse_folder(data.get("initial", ""))})

        def open(self, data):
            if state.job["running"]:
                raise ValueError("Идёт вырезание, дождитесь окончания")
            r = state.open(data.get("src", ""))
            n_src = len(list_sources(r.src, bool(data.get("recursive", True))))
            self._json({"src": str(r.src), "sources": n_src, "items": r.items()})

        def process(self, data):
            if state.job["running"]:
                raise ValueError("Вырезание уже идёт")
            r = state.open(data.get("src", ""))
            items = r.items()
            if items and not data.get("confirm"):
                return self._json({"need_confirm": True, "existing": len(items),
                                   "rejected": sum(it["bad"] for it in items)})
            params = params_from(lambda k, d: data.get(k, d))
            square = min(1024, max(32, int(data.get("square", SQUARE_SIZE))))
            state.job.update(running=True, done=0, total=0, current="", found=0, error=None)
            threading.Thread(target=state.run, args=(r, params, square, bool(data.get("recursive", True))),
                             daemon=True).start()
            self._json({"ok": True})

        def mark(self, data):
            if not state.review:
                raise ValueError("Папка не открыта")
            state.review.mark([Path(n).name for n in data.get("names", [])], bool(data.get("bad")))
            self._json({"ok": True})

        def save(self, data):
            if not state.review:
                raise ValueError("Папка не открыта")
            if state.job["running"]:
                raise ValueError("Идёт вырезание, дождитесь окончания")
            if data.get("dir", "").strip():
                state.out = Path(data["dir"].strip()).expanduser().resolve()
            self._json(state.review.save(state.out))

    return Handler


PAGE = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Вырезание яблок</title>
<style>
  :root { --bg:#14161a; --panel:#1d2026; --line:#2c3038; --text:#e6e8eb; --muted:#8a919c;
          --bad:#ff3b30; --ok:#39d353; --accent:#5b9cf5; }
  * { box-sizing:border-box; }
  [hidden] { display:none !important; }
  body { margin:0; background:var(--bg); color:var(--text); font:14px/1.4 system-ui, sans-serif;
         height:100vh; display:flex; flex-direction:column; }
  header { display:flex; gap:8px; align-items:center; padding:8px 16px; background:var(--panel);
           border-bottom:1px solid var(--line); flex-wrap:wrap; }
  header h1 { font-size:16px; margin:0 8px 0 0; }
  input, select, button { background:var(--bg); color:var(--text); border:1px solid var(--line); border-radius:6px;
                          padding:6px 10px; font:inherit; }
  button { cursor:pointer; } button:hover { border-color:var(--accent); }
  button:disabled { opacity:.5; cursor:default; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  button.save { background:#1f7a35; border-color:#1f7a35; color:#fff; }
  input[type=number] { width:80px; }
  .row { display:flex; gap:8px; align-items:center; padding:6px 16px; flex-wrap:wrap; }
  .row label { color:var(--muted); }
  .grow { flex:1; min-width:220px; }
  .muted { color:var(--muted); }
  .bar { flex:1; min-width:120px; height:8px; background:var(--line); border-radius:4px; overflow:hidden; }
  .bar div { height:100%; background:var(--accent); width:0; transition:width .2s; }
  #grid { flex:1; min-height:0; overflow-y:auto; padding:8px 16px; display:grid; gap:8px; align-content:start;
          grid-template-columns:repeat(auto-fill, minmax(var(--cell, 140px), 1fr)); }
  .cut { position:relative; aspect-ratio:1; border:3px solid var(--line); border-radius:8px; overflow:hidden;
         background:#fff; cursor:pointer; user-select:none; }
  .cut img { width:100%; height:100%; object-fit:contain; display:block; }
  .cut:hover { border-color:var(--accent); }
  .cut.bad { border-color:var(--bad); }
  .cut.bad img { opacity:.35; }
  .cut.bad::after { content:"✕"; position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
                    color:var(--bad); font-size:64px; font-weight:700; }
  .cut .cap { position:absolute; left:0; right:0; bottom:0; padding:2px 6px; font-size:11px; color:#fff;
              background:#000a; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  #empty { padding:40px 16px; text-align:center; color:var(--muted); }
  footer { display:flex; gap:8px; align-items:center; padding:8px 16px; background:var(--panel);
           border-top:1px solid var(--line); flex-wrap:wrap; }
  .toast { position:fixed; right:16px; bottom:64px; background:var(--panel); border:1px solid var(--line);
           padding:10px 14px; border-radius:8px; display:none; white-space:pre-line; max-width:440px; z-index:10; }
  @media (max-width:600px) { #grid { --cell:100px !important; } }
</style>
</head>
<body>
<header>
  <h1>Вырезание яблок (YOLO11m-seg)</h1>
  <input id="src" class="grow" placeholder="папка с фотографиями">
  <button id="browse">Обзор…</button>
  <button id="openBtn" title="показать уже вырезанное из этой папки">Открыть</button>
  <button id="runBtn" class="primary">Вырезать все фото</button>
</header>
<div class="row">
  <label title="минимальная уверенность YOLO: ниже — больше яблок, но и больше ложных">Уверенность</label><input id="conf" type="number" min="0.01" max="1" step="0.05">
  <label title="рамки, перекрывающиеся сильнее, считаются одним яблоком">NMS</label><input id="iou" type="number" min="0.05" max="0.95" step="0.05">
  <label title="до какого размера YOLO уменьшает фото; больше — лучше видны мелкие яблоки, но медленнее">Размер для YOLO</label><input id="imgsz" type="number" min="320" max="4096" step="32">
  <label title="маски меньше этой площади (px² исходного фото) отбрасываются">Мин. площадь</label><input id="min_area" type="number" min="0" step="100">
  <label>Квадрат, px</label><input id="square" type="number" min="32" max="1024" step="32">
  <label><input type="checkbox" id="recursive" checked> с подпапками</label>
  <div class="bar"><div id="jobBar"></div></div>
  <span id="jobText" class="muted"></span>
</div>
<div class="row" style="padding-top:0">
  <span id="stats"></span>
  <select id="filter">
    <option value="all">Все</option><option value="good">Хорошие</option><option value="bad">Плохие</option>
  </select>
  <label>Размер</label><input id="zoom" type="range" min="80" max="320" step="10" value="140">
  <button id="allGood" title="снять отметку со всех показанных">Снять отметки</button>
  <span class="muted">Клик — отметить плохую / снять отметку · Shift+клик — диапазон</span>
</div>
<div id="grid"></div>
<div id="empty">Выберите папку с фото и нажмите «Вырезать все фото»</div>
<footer>
  <label class="muted">Сохранить в</label><input id="out" class="grow">
  <button id="saveBtn" class="save" disabled>Сохранить хорошие в train_dataset</button>
</footer>
<div class="toast" id="toast"></div>

<script>
const $ = id => document.getElementById(id);
const PARAMS = ["conf", "iou", "imgsz", "min_area", "square"];
let items = [], view = [], lastClick = -1, pollTimer = null;

async function api(url, body) {
  const r = await fetch(url, body === undefined ? {} : { method: "POST", body: JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
function toast(text, ms = 5000) {
  const t = $("toast"); t.textContent = text; t.style.display = "block";
  clearTimeout(t._h); t._h = setTimeout(() => t.style.display = "none", ms);
}

async function init() {
  const s = await api("/api/state");
  PARAMS.forEach(k => $(k).value = s.defaults[k]);
  $("src").value = s.src; $("out").value = s.out;
  try { const z = localStorage.getItem("cutZoom"); if (z) $("zoom").value = z; } catch (e) {}
  applyZoom();
  if (s.job.running) pollJob(); else if (s.src) loadItems();
}

$("zoom").oninput = () => { applyZoom(); try { localStorage.setItem("cutZoom", $("zoom").value); } catch (e) {} };
function applyZoom() { $("grid").style.setProperty("--cell", $("zoom").value + "px"); }

$("browse").onclick = async () => {
  $("browse").disabled = true;
  try {
    const r = await api("/api/browse", { initial: $("src").value });
    if (r.path) { $("src").value = r.path; await openFolder(); }
  } catch (e) { toast(e.message); }
  $("browse").disabled = false;
};
$("openBtn").onclick = () => openFolder();
$("src").addEventListener("keydown", e => { if (e.key === "Enter") openFolder(); });

async function openFolder() {
  try {
    const r = await api("/api/open", { src: $("src").value, recursive: $("recursive").checked });
    $("src").value = r.src;
    setItems(r.items);
    toast(r.items.length ? `Фото в папке: ${r.sources}. Уже вырезано: ${r.items.length}`
                         : `Фото в папке: ${r.sources}. Нажмите «Вырезать все фото»`);
  } catch (e) { toast(e.message); }
}

async function runAll(confirmed = false) {
  const body = { src: $("src").value, recursive: $("recursive").checked, confirm: confirmed };
  PARAMS.forEach(k => body[k] = Number($(k).value));
  try {
    const r = await api("/api/process", body);
    if (r.need_confirm) {
      const msg = `Из этой папки уже вырезано ${r.existing} яблок` +
        (r.rejected ? `, ${r.rejected} отмечено плохими` : "") + `.\nВырезать заново? Старые вырезки и отметки удалятся.`;
      if (confirm(msg)) return runAll(true);
      return;
    }
    setItems([]);
    pollJob();
  } catch (e) { toast(e.message); }
}
$("runBtn").onclick = () => runAll(false);

function pollJob() {
  clearInterval(pollTimer);
  $("runBtn").disabled = $("openBtn").disabled = $("saveBtn").disabled = true;
  let tick = 0;
  pollTimer = setInterval(async () => {
    const j = await api("/api/job");
    $("jobBar").style.width = (j.total ? 100 * j.done / j.total : 0) + "%";
    if (j.running) {
      $("jobText").textContent = `[${j.done}/${j.total}] ${j.current} · яблок: ${j.found}`;
      if (++tick % 5 === 0) loadItems();   // показываем вырезки по мере появления
      return;
    }
    clearInterval(pollTimer);
    $("runBtn").disabled = $("openBtn").disabled = false;
    if (j.error) { $("jobText").textContent = "Ошибка: " + j.error; toast("Ошибка: " + j.error); }
    else { $("jobText").textContent = `Готово: ${j.found} яблок из ${j.total} фото`; toast(`Готово: вырезано ${j.found} яблок.\nОтметьте плохие вырезки и нажмите «Сохранить».`); }
    loadItems();
  }, 500);
}

async function loadItems() {
  try { setItems((await api("/api/items")).items); } catch (e) { toast(e.message); }
}

function setItems(list) {
  const scroll = $("grid").scrollTop;
  items = list; lastClick = -1;
  render();
  $("grid").scrollTop = scroll;
}

$("filter").onchange = render;

function render() {
  const f = $("filter").value;
  view = items.filter(it => f === "all" || (f === "bad" ? it.bad : !it.bad));
  const g = $("grid");
  g.innerHTML = "";
  view.forEach((it, i) => {
    const d = document.createElement("div");
    d.className = "cut" + (it.bad ? " bad" : "");
    d.title = `${it.source}\nYOLO: ${it.conf.toFixed(2)}`;
    d.innerHTML = `<img loading="lazy" src="/img/${encodeURIComponent(it.name)}"><div class="cap">${it.source}</div>`;
    d.onclick = e => toggle(i, e.shiftKey);
    g.appendChild(d);
  });
  $("empty").hidden = view.length > 0;
  if (!view.length) $("empty").textContent = items.length ? "Нет вырезок в этом фильтре" : "Выберите папку с фото и нажмите «Вырезать все фото»";
  stats();
}

function stats() {
  const nBad = items.filter(it => it.bad).length;
  $("stats").innerHTML = `вырезок: <b>${items.length}</b> · плохих: <b style="color:var(--bad)">${nBad}</b> · ` +
    `пойдут в датасет: <b style="color:var(--ok)">${items.length - nBad}</b>`;
  $("saveBtn").disabled = !items.length || $("runBtn").disabled;  // во время вырезания сохранять нельзя
}

async function setBad(list, bad) {
  list.forEach(it => it.bad = bad);
  const cells = $("grid").children;
  list.forEach(it => { const i = view.indexOf(it); if (cells[i]) cells[i].classList.toggle("bad", bad); });
  stats();
  try { await api("/api/mark", { names: list.map(it => it.name), bad }); }
  catch (e) { toast("Ошибка сохранения отметки: " + e.message); }
}

function toggle(i, range) {
  const it = view[i];
  if (range && lastClick >= 0) {
    // Shift+клик: весь диапазон получает то же состояние, что и последний кликнутый
    const [a, b] = [Math.min(lastClick, i), Math.max(lastClick, i)];
    setBad(view.slice(a, b + 1), view[lastClick].bad);
  } else {
    setBad([it], !it.bad);
  }
  lastClick = i;
}

$("allGood").onclick = () => {
  const list = view.filter(it => it.bad);
  if (list.length && confirm(`Снять отметку «плохая» с ${list.length} вырезок?`)) setBad(list, false);
};

$("saveBtn").onclick = async () => {
  try {
    const r = await api("/api/save", { dir: $("out").value });
    $("out").value = r.dir;
    toast(`Сохранено ${r.saved} вырезок в ${r.dir}` + `\nПропущено плохих: ${r.skipped}` +
          (r.removed ? `\nУдалено из датасета ранее сохранённых, теперь плохих: ${r.removed}` : ""), 7000);
  } catch (e) { toast(e.message); }
};

init();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description="YOLO11m-seg: вырезать яблоки из папки и отобрать хорошие в train_dataset")
    ap.add_argument("--src", default="", help="папка с фото (можно выбрать в приложении)")
    ap.add_argument("--out", type=Path, default=ROOT / "train_dataset", help="куда сохранять хорошие вырезки")
    ap.add_argument("--yolo", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--classes", default="apple", help="классы YOLO через запятую, например apple,orange")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    state = State(Segmenter(args.yolo, args.classes.split(",")), args.out.resolve())
    if args.src:
        state.open(args.src)
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
