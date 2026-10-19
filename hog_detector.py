"""HOG: поиск яблок на фото куч и вырезание каждого яблока на белый квадрат. Без обучения.

Как это работает:
  1. HOG (гистограммы направлений градиентов) описывает форму в окне 64x64:
     в каких направлениях идут края в каждом кусочке окна.
  2. Эталон — HOG нарисованного круга: у круглого яблока края идут по окружности.
     Эталон строится прямо в коде, никаких данных и обучения не нужно.
  3. Окно скользит по фото на разных масштабах (от мин. до макс. размера яблока),
     для каждого окна считается похожесть его HOG на HOG круга (от -1 до 1).
     Окна с похожестью выше порога — кандидаты в яблоки.
  4. Пересекающиеся рамки прореживаются (NMS): остаётся самая похожая на круг.
  5. Каждое яблоко отделяется от соседей (GrabCut внутри рамки или эллипс)
     и кладётся по центру белого квадрата.

Обработка папки без веб-приложения:
    python hog_detector.py used_data out_apples --square 128 --min-size 34 --max-size 80
"""

import argparse
import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

WIN = 64                      # окно детектора, px
CELL = 8                      # ячейка HOG, px
BINS = 9                      # направления 0..180°
WIN_CELLS = WIN // CELL       # 8
WIN_BLOCKS = WIN_CELLS - 1    # 7 (блок 2x2 ячейки, шаг 1 ячейка)
CIRCLE_RADIUS = 0.42          # радиус эталонного круга в долях окна


# ---------------------------------------------------------------- HOG

def cell_histograms(img: np.ndarray) -> np.ndarray:
    """Гистограммы направлений градиентов по ячейкам 8x8 -> [Hc, Wc, BINS]."""
    f = img.astype(np.float32) / 255.0
    if f.ndim == 2:
        f = f[..., None]
    kx = np.array([[-1, 0, 1]], np.float32)
    gx = cv2.filter2D(f, -1, kx, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.filter2D(f, -1, kx.T, borderType=cv2.BORDER_REPLICATE)
    if gx.ndim == 2:
        gx, gy = gx[..., None], gy[..., None]
    mag = np.sqrt(gx * gx + gy * gy)
    # для цветного изображения берём канал с самым сильным градиентом
    ch = mag.argmax(axis=2)[..., None]
    mag = np.take_along_axis(mag, ch, 2)[..., 0]
    gx = np.take_along_axis(gx, ch, 2)[..., 0]
    gy = np.take_along_axis(gy, ch, 2)[..., 0]

    hc, wc = mag.shape[0] // CELL, mag.shape[1] // CELL
    mag, gx, gy = (a[:hc * CELL, :wc * CELL] for a in (mag, gx, gy))
    ang = np.arctan2(gy, gx) % np.pi
    pos = ang / (np.pi / BINS) - 0.5
    b0 = np.floor(pos).astype(np.int64)
    frac = pos - b0
    b0 %= BINS
    b1 = (b0 + 1) % BINS

    cy = np.arange(hc * CELL) // CELL
    cx = np.arange(wc * CELL) // CELL
    cell_idx = (cy[:, None] * wc + cx[None, :]) * BINS
    n = hc * wc * BINS
    hist = np.bincount((cell_idx + b0).ravel(), (mag * (1 - frac)).ravel(), n)
    hist += np.bincount((cell_idx + b1).ravel(), (mag * frac).ravel(), n)
    return hist.reshape(hc, wc, BINS).astype(np.float32)


def block_features(hist: np.ndarray) -> np.ndarray:
    """Блоки 2x2 ячейки с нормировкой L2-Hys -> [Hc-1, Wc-1, 4*BINS]."""
    b = np.concatenate([hist[:-1, :-1], hist[:-1, 1:], hist[1:, :-1], hist[1:, 1:]], axis=2)
    b = b / np.sqrt((b * b).sum(2, keepdims=True) + 1e-6)
    b = np.minimum(b, 0.2)
    return b / np.sqrt((b * b).sum(2, keepdims=True) + 1e-6)


# ---------------------------------------------------------------- детектор

@dataclass
class DetectParams:
    min_size: int = 28        # минимальный размер яблока, px исходного фото
    max_size: int = 0         # 0 = половина меньшей стороны фото
    scale_step: float = 1.15  # шаг пирамиды масштабов
    threshold: float = 0.2    # мин. похожесть на круг (ниже — больше яблок, но и больше мусора)
    nms_iou: float = 0.2      # рамки, перекрывающиеся сильнее, считаются одним яблоком


@lru_cache(maxsize=1)
def circle_template() -> np.ndarray:
    """HOG нарисованного круга -> [7, 7, 36], центрирован и нормирован.

    Из каждого блока вычитается среднее по направлениям, поэтому однородная текстура
    (одинаково во все стороны) даёт 0, а важна только форма краёв."""
    full = WIN + 2 * CELL
    img = np.zeros((full, full), np.uint8)
    cv2.circle(img, (full // 2, full // 2), round(WIN * CIRCLE_RADIUS), 255, -1, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 2.0)
    hist = cell_histograms(img)[1:1 + WIN_CELLS, 1:1 + WIN_CELLS]
    t = block_features(hist).reshape(WIN_BLOCKS, WIN_BLOCKS, 4, BINS)
    t = t - t.mean(axis=3, keepdims=True)
    t = t.reshape(WIN_BLOCKS, WIN_BLOCKS, 4 * BINS)
    return (t / np.linalg.norm(t)).astype(np.float32)


def score_map(img: np.ndarray) -> np.ndarray:
    """Похожесть HOG каждого окна (шаг — одна ячейка) на HOG круга -> [Hc-7, Wc-7]."""
    t = circle_template()
    blocks = block_features(cell_histograms(img))
    hb, wb = blocks.shape[:2]
    oh, ow = hb - WIN_BLOCKS + 1, wb - WIN_BLOCKS + 1
    if oh <= 0 or ow <= 0:
        return np.zeros((0, 0), np.float32)
    dot = np.zeros((oh, ow), np.float32)
    norm = np.zeros((oh, ow), np.float32)
    sq = (blocks * blocks).sum(2)
    for dy in range(WIN_BLOCKS):
        for dx in range(WIN_BLOCKS):
            dot += blocks[dy:dy + oh, dx:dx + ow] @ t[dy, dx]
            norm += sq[dy:dy + oh, dx:dx + ow]
    return dot / np.sqrt(norm + 1e-6)


def detect(img: np.ndarray, p: DetectParams = DetectParams()) -> list[tuple[float, float, float, float, float]]:
    """Находит яблоки. Возвращает [(x1, y1, x2, y2, похожесть)] в координатах исходного фото."""
    h, w = img.shape[:2]
    max_size = p.max_size or max(p.min_size + 1, min(h, w) // 2)
    boxes = []
    size = float(p.min_size)
    while size <= max_size:
        scale = WIN / size
        resized = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                             interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        sm = score_map(resized)
        ys, xs = np.nonzero(sm > p.threshold)
        for y, x in zip(ys, xs):
            x1, y1 = x * CELL / scale, y * CELL / scale
            boxes.append((x1, y1, x1 + size, y1 + size, float(sm[y, x])))
        size *= p.scale_step
    return nms(boxes, p.nms_iou)


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, thr: float):
    if not boxes:
        return []
    arr = np.array(boxes, np.float32)
    order = arr[:, 4].argsort()[::-1]
    x1, y1, x2, y2 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    area = (x2 - x1) * (y2 - y1)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        ix = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        iy = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = ix * iy
        ov = inter / (area[i] + area[rest] - inter)
        # маленькая рамка почти целиком внутри большой — тоже дубль
        contain = inter / np.minimum(area[i], area[rest])
        order = rest[(ov <= thr) & (contain <= 0.7)]
    return [tuple(float(v) for v in arr[i]) for i in keep]


# ---------------------------------------------------------------- вырезание на белый фон

def fruit_mask(img: np.ndarray, box, mode: str = "grabcut"):
    """Маска фрукта внутри рамки box.

    mode="grabcut" — граница уточняется GrabCut (внутри эллипса, вписанного в рамку);
    mode="ellipse" — просто эллипс, вписанный в рамку (быстрее, но захватывает углы соседей);
    mode="box"     — вся рамка как есть.

    Возвращает (mask, x0, y0): mask — uint8 0/255 размером с окрестность рамки,
    (x0, y0) — где её левый верхний угол на исходном фото.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box[:4]
    bw, bh = x2 - x1, y2 - y1
    pad = 0.15
    cx1, cy1 = max(0, int(x1 - pad * bw)), max(0, int(y1 - pad * bh))
    cx2, cy2 = min(w, int(np.ceil(x2 + pad * bw))), min(h, int(np.ceil(y2 + pad * bh)))
    crop = img[cy1:cy2, cx1:cx2]
    ch, cw = crop.shape[:2]
    ecx, ecy = (x1 + x2) / 2 - cx1, (y1 + y2) / 2 - cy1

    if mode == "box":
        mask = np.zeros((ch, cw), np.uint8)
        mask[max(0, int(y1 - cy1)):int(y2 - cy1), max(0, int(x1 - cx1)):int(x2 - cx1)] = 255
    else:
        ellipse = np.zeros((ch, cw), np.uint8)
        cv2.ellipse(ellipse, (round(ecx), round(ecy)), (round(bw * 0.5), round(bh * 0.5)), 0, 0, 360, 255, -1)
        mask = ellipse
        if mode == "grabcut" and min(ch, cw) >= 12:
            gc = np.full((ch, cw), cv2.GC_BGD, np.uint8)
            outer = np.zeros_like(gc)
            cv2.ellipse(outer, (round(ecx), round(ecy)), (round(bw * 0.58), round(bh * 0.58)), 0, 0, 360, 1, -1)
            gc[outer > 0] = cv2.GC_PR_BGD
            gc[ellipse > 0] = cv2.GC_PR_FGD
            core = np.zeros_like(gc)
            cv2.ellipse(core, (round(ecx), round(ecy)), (round(bw * 0.25), round(bh * 0.25)), 0, 0, 360, 1, -1)
            gc[core > 0] = cv2.GC_FGD
            try:
                bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
                cv2.grabCut(crop, gc, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
                fg = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8) * 255
                fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
                n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
                if n > 1:
                    # оставляем кусок, содержащий центр (или самый большой)
                    c = lab[min(ch - 1, round(ecy)), min(cw - 1, round(ecx))]
                    if c == 0:
                        c = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
                    fg = (lab == c).astype(np.uint8) * 255
                    # заполняем дырки (тёмные пятна гнили не должны стать белыми)
                    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(fg, cnts, -1, 255, -1)
                    if fg.sum() > 0.35 * ellipse.sum():
                        mask = fg
            except cv2.error:
                pass
    return mask, cx1, cy1


def label_map(img: np.ndarray, dets, mode: str = "grabcut", min_area: int = 30) -> tuple[np.ndarray, list[dict]]:
    """Маски всех найденных фруктов одной картой uint16: 0 — фон, 1..N — номер маски.

    Маски не пересекаются: спорный пиксель достаётся более уверенной рамке (dets идут по убыванию оценки).
    Возвращает (карта, [{"id", "score", "bbox": [x, y, w, h]}])."""
    labels = np.zeros(img.shape[:2], np.uint16)
    info = []
    for box in dets:
        mask, x0, y0 = fruit_mask(img, box, mode)
        region = labels[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]]
        free = (mask > 0) & (region == 0)
        if free.sum() < min_area:
            continue
        mid = len(info) + 1
        region[free] = mid
        ys, xs = np.nonzero(free)
        info.append({"id": mid, "score": round(float(box[4]), 3),
                     "bbox": [int(x0 + xs.min()), int(y0 + ys.min()), int(np.ptp(xs) + 1), int(np.ptp(ys) + 1)]})
    return labels, info


def cutout_on_white(img: np.ndarray, box, square: int, mode: str = "grabcut", margin: float = 0.05) -> np.ndarray:
    """Вырезает фрукт из рамки box (см. fruit_mask) и кладёт по центру белого квадрата square x square."""
    mask, cx1, cy1 = fruit_mask(img, box, mode)
    crop = img[cy1:cy1 + mask.shape[0], cx1:cx1 + mask.shape[1]]

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.full((square, square, 3), 255, np.uint8)
    mx1, mx2, my1, my2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    obj, m = crop[my1:my2, mx1:mx2], mask[my1:my2, mx1:mx2]

    inner = max(1, round(square * (1 - 2 * margin)))
    k = inner / max(obj.shape[:2])
    nw, nh = max(1, round(obj.shape[1] * k)), max(1, round(obj.shape[0] * k))
    interp = cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC
    obj = cv2.resize(obj, (nw, nh), interpolation=interp)
    m = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_LINEAR)
    alpha = cv2.GaussianBlur(m, (3, 3), 0).astype(np.float32)[..., None] / 255.0

    canvas = np.full((square, square, 3), 255, np.uint8)
    ox, oy = (square - nw) // 2, (square - nh) // 2
    roi = canvas[oy:oy + nh, ox:ox + nw].astype(np.float32)
    canvas[oy:oy + nh, ox:ox + nw] = (obj * alpha + roi * (1 - alpha)).astype(np.uint8)
    return canvas


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def imread(path: Path) -> np.ndarray:
    # cv2.imread не открывает пути с кириллицей на Windows
    data = np.fromfile(str(path), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"не удалось открыть {path}")
    return img


def imwrite(path: Path, img: np.ndarray):
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise ValueError(f"не удалось сохранить {path}")
    buf.tofile(str(path))


def process_folder(src: Path, dst: Path, square: int, params: DetectParams,
                   mode: str = "grabcut", on_progress=None) -> int:
    """Обрабатывает все фото папки. Пишет dst/<фото>_<NNN>.png и dst/detections.csv."""
    dst.mkdir(parents=True, exist_ok=True)
    files = list_images(src)
    total = 0
    with (dst / "detections.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["crop", "source", "x1", "y1", "x2", "y2", "score"])
        for i, path in enumerate(files):
            img = imread(path)
            dets = detect(img, params)
            for k, box in enumerate(dets):
                name = f"{path.stem}_{k:03d}.png"
                imwrite(dst / name, cutout_on_white(img, box, square, mode))
                wr.writerow([name, path.name, *(round(v) for v in box[:4]), round(box[4], 3)])
            total += len(dets)
            if on_progress:
                on_progress(i + 1, len(files), path.name, len(dets))
    return total


def main():
    ap = argparse.ArgumentParser(description="HOG: вырезать яблоки из всех фото папки на белые квадраты")
    ap.add_argument("src", type=Path, help="папка с фото куч")
    ap.add_argument("dst", type=Path, help="куда сохранить яблоки")
    ap.add_argument("--square", type=int, default=128, help="сторона белого квадрата, px")
    ap.add_argument("--mode", default="grabcut", choices=["grabcut", "ellipse", "box"])
    ap.add_argument("--min-size", type=int, default=DetectParams.min_size)
    ap.add_argument("--max-size", type=int, default=DetectParams.max_size)
    ap.add_argument("--threshold", type=float, default=DetectParams.threshold)
    ap.add_argument("--nms", type=float, default=DetectParams.nms_iou)
    args = ap.parse_args()

    params = DetectParams(args.min_size, args.max_size, threshold=args.threshold, nms_iou=args.nms)
    n = process_folder(args.src, args.dst, args.square, params, args.mode,
                       lambda i, k, name, c: print(f"[{i}/{k}] {name}: {c}"))
    print(f"Готово: {n} яблок -> {args.dst}")


if __name__ == "__main__":
    main()
