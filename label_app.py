"""Разметка кропов на «гнилые» / «не гнилые».

Запуск:
    python label_app.py            # откроет браузер на http://127.0.0.1:8765
    python label_app.py --crops crops --port 8765

Разметка сохраняется сразу после каждого клика в crops/labels.csv.
Кнопка «Экспорт» раскладывает размеченные картинки в dataset_cls/train/{rotten,fresh}.
"""

import argparse
import csv
import json
import shutil
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
LABELS = ("rotten", "fresh")
ROTTEN_HINTS = ("rotten", "moldy", "bruised")


class Store:
    def __init__(self, crops_dir: Path):
        self.crops_dir = crops_dir
        self.csv_path = crops_dir / "labels.csv"
        self.lock = threading.Lock()
        self.labels: dict[str, str] = {}
        if self.csv_path.exists():
            with self.csv_path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("label") in LABELS:
                        self.labels[row["filename"]] = row["label"]

    def images(self) -> list[str]:
        return sorted(p.name for p in self.crops_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    def set(self, name: str, label: str | None):
        with self.lock:
            if label in LABELS:
                self.labels[name] = label
            else:
                self.labels.pop(name, None)
            tmp = self.csv_path.with_suffix(".tmp")
            with tmp.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["filename", "label"])
                for k in sorted(self.labels):
                    w.writerow([k, self.labels[k]])
            tmp.replace(self.csv_path)

    def export(self, out_dir: Path) -> dict:
        with self.lock:
            items = dict(self.labels)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        counts = {}
        for label in LABELS:
            names = sorted(n for n, l in items.items() if l == label and (self.crops_dir / n).exists())
            dst = out_dir / "train" / label
            dst.mkdir(parents=True, exist_ok=True)
            for n in names:
                shutil.copy2(self.crops_dir / n, dst / n)
            counts[f"train/{label}"] = len(names)
        return counts


def hint_for(name: str) -> str:
    return "rotten" if any(h in name.lower() for h in ROTTEN_HINTS) else "fresh"


def make_handler(store: Store, export_dir: Path):
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
            elif path == "/api/items":
                self._json([
                    {"name": n, "label": store.labels.get(n), "hint": hint_for(n)}
                    for n in store.images()
                ])
            elif path.startswith("/img/"):
                name = Path(unquote(path[len("/img/"):])).name
                file = store.crops_dir / name
                if file.suffix.lower() not in IMAGE_EXTS or not file.is_file():
                    return self._send(404, b"not found", "text/plain")
                ctype = "image/png" if file.suffix.lower() == ".png" else "image/jpeg"
                self._send(200, file.read_bytes(), ctype)
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(length) or b"{}")
            if path == "/api/label":
                name = Path(data["name"]).name
                if not (store.crops_dir / name).is_file():
                    return self._json({"error": "no such file"}, 404)
                store.set(name, data.get("label"))
                self._json({"ok": True})
            elif path == "/api/export":
                counts = store.export(export_dir)
                self._json({"ok": True, "dir": str(export_dir), "counts": counts})
            else:
                self._json({"error": "not found"}, 404)

    return Handler


PAGE = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Разметка яблок</title>
<style>
  :root { --bg:#14161a; --panel:#1d2026; --line:#2c3038; --text:#e6e8eb; --muted:#8a919c;
          --rot:#d9534f; --fresh:#4caf50; --accent:#5b9cf5; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:14px/1.4 system-ui, sans-serif;
         height:100vh; display:grid; grid-template-rows:auto 1fr auto; }
  header { display:flex; gap:16px; align-items:center; padding:10px 16px; background:var(--panel);
           border-bottom:1px solid var(--line); flex-wrap:wrap; }
  header h1 { font-size:16px; margin:0; }
  .progress { flex:1; min-width:160px; height:8px; background:var(--line); border-radius:4px; overflow:hidden; display:flex; }
  .progress div { height:100%; }
  .stats { color:var(--muted); white-space:nowrap; }
  select, button { background:var(--bg); color:var(--text); border:1px solid var(--line); border-radius:6px;
                   padding:6px 10px; font:inherit; cursor:pointer; }
  button:hover { border-color:var(--accent); }
  main { display:grid; grid-template-columns:1fr 300px; min-height:0; }
  .viewer { display:flex; flex-direction:column; align-items:center; justify-content:center; gap:14px; padding:16px; min-height:0; }
  .imgwrap { flex:1; min-height:0; width:100%; display:flex; align-items:center; justify-content:center;
             border:4px solid transparent; border-radius:10px; }
  .imgwrap.rotten { border-color:var(--rot); }
  .imgwrap.fresh  { border-color:var(--fresh); }
  .imgwrap img { width:100%; height:100%; object-fit:contain; }
  .meta { color:var(--muted); text-align:center; }
  .meta b { color:var(--text); }
  .actions { display:flex; gap:12px; flex-wrap:wrap; justify-content:center; }
  .actions button { padding:12px 22px; font-size:16px; border-width:2px; }
  .btn-rot { border-color:var(--rot); } .btn-rot.on { background:var(--rot); }
  .btn-fresh { border-color:var(--fresh); } .btn-fresh.on { background:var(--fresh); }
  kbd { background:var(--line); border-radius:4px; padding:1px 6px; font-size:12px; }
  aside { border-left:1px solid var(--line); background:var(--panel); overflow-y:auto; padding:8px;
          display:grid; grid-template-columns:repeat(auto-fill, minmax(84px, 1fr)); grid-auto-rows:84px; gap:6px; align-content:start; }
  .thumb { position:relative; border:3px solid var(--line); border-radius:6px; overflow:hidden; cursor:pointer; background:#000; }
  .thumb img { width:100%; height:100%; object-fit:cover; }
  .thumb.rotten { border-color:var(--rot); } .thumb.fresh { border-color:var(--fresh); }
  .thumb.cur { outline:3px solid var(--accent); outline-offset:1px; }
  footer { padding:6px 16px; color:var(--muted); background:var(--panel); border-top:1px solid var(--line); font-size:12px; }
  .toast { position:fixed; right:16px; bottom:40px; background:var(--panel); border:1px solid var(--line);
           padding:10px 14px; border-radius:8px; display:none; white-space:pre; }
  @media (max-width: 760px) { main { grid-template-columns:1fr; grid-template-rows:1fr 160px; }
                               aside { border-left:none; border-top:1px solid var(--line); } }
</style>
</head>
<body>
<header>
  <h1>Гнилые / не гнилые</h1>
  <div class="progress"><div id="pRot" style="background:var(--rot)"></div><div id="pFresh" style="background:var(--fresh)"></div></div>
  <span class="stats" id="stats"></span>
  <select id="filter">
    <option value="all">Все</option>
    <option value="none">Неразмеченные</option>
    <option value="rotten">Гнилые</option>
    <option value="fresh">Не гнилые</option>
  </select>
  <button id="exportBtn">Экспорт датасета</button>
</header>
<main>
  <section class="viewer">
    <div class="imgwrap" id="wrap"><img id="img" alt=""></div>
    <div class="meta" id="meta"></div>
    <div class="actions">
      <button class="btn-rot" id="bRot">Гнилое <kbd>1</kbd> <kbd>R</kbd></button>
      <button class="btn-fresh" id="bFresh">Не гнилое <kbd>2</kbd> <kbd>F</kbd></button>
      <button id="bClear">Снять метку <kbd>0</kbd></button>
      <button id="bHint">Как у детектора <kbd>Enter</kbd></button>
    </div>
  </section>
  <aside id="grid"></aside>
</main>
<footer>
  <kbd>1</kbd>/<kbd>R</kbd> гнилое · <kbd>2</kbd>/<kbd>F</kbd> не гнилое · <kbd>Enter</kbd> согласиться с детектором ·
  <kbd>0</kbd> снять · <kbd>←</kbd>/<kbd>→</kbd> навигация · <kbd>Ctrl+Z</kbd> отменить · после метки — переход к следующей
</footer>
<div class="toast" id="toast"></div>
<script>
const NAMES = { rotten: "гнилое", fresh: "не гнилое" };
let items = [], view = [], pos = 0, history = [];
const $ = id => document.getElementById(id);

async function load() {
  items = await (await fetch("/api/items")).json();
  applyFilter(true);
}

function applyFilter(jumpToFirstUnlabeled) {
  const f = $("filter").value;
  const curName = view[pos]?.name;
  view = items.filter(it => f === "all" || (f === "none" ? !it.label : it.label === f));
  if (jumpToFirstUnlabeled) {
    const i = view.findIndex(it => !it.label);
    pos = i >= 0 ? i : 0;
  } else {
    const i = view.findIndex(it => it.name === curName);
    pos = i >= 0 ? i : Math.min(pos, Math.max(view.length - 1, 0));
  }
  renderGrid();
  render();
}

function renderGrid() {
  const g = $("grid");
  g.innerHTML = "";
  view.forEach((it, i) => {
    const d = document.createElement("div");
    d.className = "thumb";
    d.title = it.name;
    d.innerHTML = `<img loading="lazy" src="/img/${encodeURIComponent(it.name)}">`;
    d.onclick = () => { pos = i; render(); };
    g.appendChild(d);
  });
}

function render() {
  const thumbs = $("grid").children;
  view.forEach((it, i) => {
    const t = thumbs[i];
    if (!t) return;
    t.className = "thumb" + (it.label ? " " + it.label : "") + (i === pos ? " cur" : "");
  });
  thumbs[pos]?.scrollIntoView({ block: "nearest" });

  const it = view[pos];
  const total = items.length;
  const nRot = items.filter(x => x.label === "rotten").length;
  const nFresh = items.filter(x => x.label === "fresh").length;
  $("pRot").style.width = (100 * nRot / total) + "%";
  $("pFresh").style.width = (100 * nFresh / total) + "%";
  $("stats").textContent = `${nRot + nFresh}/${total} · гнилых ${nRot} · не гнилых ${nFresh}`;

  if (!it) {
    $("img").removeAttribute("src");
    $("wrap").className = "imgwrap";
    $("meta").textContent = "Нет картинок в этом фильтре";
    return;
  }
  $("img").src = "/img/" + encodeURIComponent(it.name);
  $("wrap").className = "imgwrap" + (it.label ? " " + it.label : "");
  $("meta").innerHTML = `<b>${pos + 1}/${view.length}</b> · ${it.name}<br>` +
    `Детектор считает: <b>${NAMES[it.hint]}</b> · метка: <b>${it.label ? NAMES[it.label] : "—"}</b>`;
  $("bRot").classList.toggle("on", it.label === "rotten");
  $("bFresh").classList.toggle("on", it.label === "fresh");
}

async function setLabel(label, advance = true, record = true) {
  const it = view[pos];
  if (!it) return;
  if (record) history.push({ name: it.name, prev: it.label ?? null });
  it.label = label;
  if (advance && pos < view.length - 1) pos++;
  render();
  const r = await fetch("/api/label", { method: "POST", body: JSON.stringify({ name: it.name, label }) });
  if (!r.ok) toast("Ошибка сохранения: " + it.name);
}

function undo() {
  const h = history.pop();
  if (!h) return;
  const it = items.find(x => x.name === h.name);
  it.label = h.prev;
  fetch("/api/label", { method: "POST", body: JSON.stringify({ name: h.name, label: h.prev }) });
  const i = view.findIndex(x => x.name === h.name);
  if (i >= 0) pos = i;
  render();
}

function toast(text) {
  const t = $("toast");
  t.textContent = text;
  t.style.display = "block";
  clearTimeout(t._h);
  t._h = setTimeout(() => t.style.display = "none", 4000);
}

$("bRot").onclick = () => setLabel("rotten");
$("bFresh").onclick = () => setLabel("fresh");
$("bClear").onclick = () => setLabel(null, false);
$("bHint").onclick = () => view[pos] && setLabel(view[pos].hint);
$("filter").onchange = () => applyFilter(false);
$("exportBtn").onclick = async () => {
  const r = await (await fetch("/api/export", { method: "POST", body: "{}" })).json();
  toast("Сохранено в " + r.dir + "\n" + Object.entries(r.counts).map(([k, v]) => `${k}: ${v}`).join("\n"));
};

document.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT") return;
  const k = e.key.toLowerCase();
  if ((e.ctrlKey || e.metaKey) && (k === "z" || k === "я")) { e.preventDefault(); return undo(); }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (k === "1" || k === "r" || k === "к") setLabel("rotten");
  else if (k === "2" || k === "f" || k === "а") setLabel("fresh");
  else if (k === "0" || k === "delete") setLabel(null, false);
  else if (k === "enter") { e.preventDefault(); view[pos] && setLabel(view[pos].hint); }
  else if (k === "arrowright" || k === " ") { e.preventDefault(); if (pos < view.length - 1) { pos++; render(); } }
  else if (k === "arrowleft" || k === "backspace") { e.preventDefault(); if (pos > 0) { pos--; render(); } }
});

load();
</script>
</body>
</html>
"""


def main():
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Разметка кропов на гнилые / не гнилые")
    ap.add_argument("--crops", type=Path, default=root / "crops")
    ap.add_argument("--export", type=Path, default=root / "dataset_cls")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    store = Store(args.crops.resolve())
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store, args.export.resolve()))
    url = f"http://127.0.0.1:{args.port}"
    print(f"Картинок: {len(store.images())}, уже размечено: {len(store.labels)}")
    print(f"Разметка: {store.csv_path}")
    print(f"Открой {url}  (Ctrl+C — выход)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
