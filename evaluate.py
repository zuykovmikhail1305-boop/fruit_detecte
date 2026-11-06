"""Проверка качества PatchCore на размеченных фото: precision, recall, F1.

Метка берётся из имени папки: .../fresh/... = норма (0), .../rotten/... = аномалия (1).
«Положительный» класс — гнилое яблоко: recall = какую долю гнилых нашли,
precision = какая доля помеченных «гнилыми» действительно гнилые.

Фото проходят тот же путь, что в anomaly_app.py: yolo11m-seg находит яблоки, каждое
вырезается на белый фон, скор фото = максимум скоров его яблок (одно гнилое — фото гнилое).
Если YOLO ничего не нашёл, скорится фото целиком.

    python evaluate.py                                  # raw/, models/patchcore.pt
    python evaluate.py --mode whole                     # без YOLO, фото целиком
    python evaluate.py --threshold 1.7 --conf 0.1
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from patchcore import PatchCore, list_images, preprocess
from yolo_cut import DEFAULT_WEIGHTS, DEFAULTS, Segmenter, cutout_on_white, params_from

ROOT = Path(__file__).resolve().parent
LABELS = {"fresh": 0, "good": 0, "normal": 0, "rotten": 1, "bad": 1, "anomaly": 1}
BATCH = 16


def label_of(path: Path, root: Path):
    for part in reversed(path.relative_to(root).parts[:-1]):
        if part.lower() in LABELS:
            return LABELS[part.lower()]
    return None


def read_bgr(path: Path) -> np.ndarray:
    # cv2.imread не открывает пути с кириллицей на Windows
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def metrics(y, s, thr):
    pred = s > thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "accuracy": (tp + tn) / len(y), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def auroc(y, s):
    """Вероятность, что случайное гнилое получит скор выше случайного свежего (ранги, без sklearn)."""
    order = s.argsort()
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):                      # одинаковым скорам — средний ранг
        ranks[s == v] = ranks[s == v].mean()
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def print_metrics(title, m):
    print(f"{title}\n  precision = {m['precision']:.3f}   recall = {m['recall']:.3f}   "
          f"F1 = {m['f1']:.3f}   accuracy = {m['accuracy']:.3f}")
    print(f"  гнилые: найдено {m['tp']}, пропущено {m['fn']} | "
          f"свежие: ложных тревог {m['fp']}, верно {m['tn']}")


def main():
    ap = argparse.ArgumentParser(description="precision / recall / F1 PatchCore на размеченных фото")
    ap.add_argument("--data", type=Path, default=ROOT / "raw", help="папка с подпапками fresh/ и rotten/")
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "patchcore.pt")
    ap.add_argument("--threshold", type=float, default=None, help="по умолчанию — порог из модели")
    ap.add_argument("--mode", choices=["yolo", "whole"], default="yolo",
                    help="yolo: вырезать яблоки как в приложении; whole: фото целиком")
    ap.add_argument("--yolo", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--classes", default="apple", help="классы YOLO через запятую, напр. apple,orange")
    ap.add_argument("--conf", type=float, default=DEFAULTS["conf"])
    ap.add_argument("--imgsz", type=int, default=DEFAULTS["imgsz"])
    ap.add_argument("--out", type=Path, default=ROOT / "results_eval" / "scores.csv")
    args = ap.parse_args()

    paths = [p for p in list_images(args.data) if label_of(p, args.data) is not None]
    if not paths:
        raise SystemExit(f"В {args.data} нет фото в папках {sorted(LABELS)}")
    model = PatchCore.load(args.model)
    thr = model.threshold if args.threshold is None else args.threshold
    size = model.image_size
    seg = Segmenter(args.yolo, args.classes.split(",")) if args.mode == "yolo" else None
    p = params_from(lambda k, d: {"conf": args.conf, "imgsz": args.imgsz}.get(k, d))
    print(f"Модель {args.model.name}: порог {thr:.3f}, CLAHE {'да' if model.clahe else 'нет'} | "
          f"фото: {len(paths)} | режим: {args.mode}")

    rows, no_det = [], 0
    for n, path in enumerate(paths, 1):
        bgr = read_bgr(path)
        dets = seg.segment(bgr, p) if seg else []
        if dets:
            cuts = [cutout_on_white(bgr, d["mask"], size) for d in dets]
        else:
            no_det += seg is not None
            cuts = [bgr]
        imgs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in cuts]
        scores = torch.cat([model.predict(torch.stack([preprocess(im, size, model.clahe)
                                                       for im in imgs[i:i + BATCH]]))[0]
                            for i in range(0, len(imgs), BATCH)])
        rows.append([path.relative_to(args.data).as_posix(), label_of(path, args.data),
                     float(scores.max()), len(dets)])
        print(f"\r  {n}/{len(paths)}", end="", flush=True)
    print()

    y = np.array([r[1] for r in rows])
    s = np.array([r[2] for r in rows])
    print(f"Свежих: {(y == 0).sum()}, гнилых: {(y == 1).sum()}"
          + (f" | YOLO не нашёл яблок на {no_det} фото (скор по фото целиком)" if seg else ""))
    for lab, name in [(0, "свежие"), (1, "гнилые")]:
        v = s[y == lab]
        print(f"  скор {name:7s}: mean={v.mean():.3f} median={np.median(v):.3f} "
              f"min={v.min():.3f} max={v.max():.3f}")

    print_metrics(f"\nПри пороге модели {thr:.3f}:", metrics(y, s, thr))
    print(f"\nAUROC = {auroc(y, s):.3f}  (качество разделения без привязки к порогу; 0.5 = случайно, 1 = идеально)")
    best = max(np.unique(s), key=lambda t: metrics(y, s, t)["f1"])
    print_metrics(f"Лучший F1 при пороге {best:.3f} (подобран на этих же фото — оценка оптимистичная):",
                  metrics(y, s, best))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "label", "score", "apples", "pred"])
        w.writerows(r + [int(r[2] > thr)] for r in rows)
    print(f"\nСкоры по каждому фото: {args.out}")


if __name__ == "__main__":
    main()
