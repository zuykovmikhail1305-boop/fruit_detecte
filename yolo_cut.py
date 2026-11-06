"""YOLO11-seg: поиск яблок на фото и вырезание каждого по маске на белый квадрат.

Используется в anomaly_app.py и cut_app.py.
Веса yolo11m-seg.pt (COCO, есть класс "apple") скачиваются в models/ автоматически.
"""

import threading
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "models" / "yolo11m-seg.pt"
# Яблоки в main_dataset почти касаются краёв кадра, поэтому поля у белого квадрата маленькие
CUT_MARGIN = 0.03
DEFAULTS = {"conf": 0.25, "iou": 0.5, "imgsz": 1024, "min_area": 400}


def params_from(get) -> dict:
    """get(key, default) -> параметры YOLO в допустимых пределах."""
    val = lambda k: float(get(k, DEFAULTS[k]))
    return {
        "conf": min(1.0, max(0.01, val("conf"))),
        "iou": min(0.95, max(0.05, val("iou"))),
        # YOLO требует размер, кратный 32
        "imgsz": int(min(4096, max(320, round(val("imgsz") / 32) * 32))),
        "min_area": int(max(0, val("min_area"))),
    }


def cutout_on_white(img: np.ndarray, mask: np.ndarray, square: int, margin: float = CUT_MARGIN) -> np.ndarray:
    """Вырезает объект по маске (uint8 0/255 размером с фото) и кладёт по центру белого квадрата."""
    ys, xs = np.nonzero(mask)
    x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    obj, m = img[y1:y2, x1:x2], mask[y1:y2, x1:x2]

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


class Segmenter:
    def __init__(self, weights: Path = DEFAULT_WEIGHTS, class_names=("apple",)):
        self.yolo = YOLO(str(weights))  # стандартные веса скачиваются сами, если файла нет
        wanted = {n.strip().lower() for n in class_names if n.strip()}
        self.classes = [i for i, n in self.yolo.names.items() if n.lower() in wanted]
        if not self.classes:
            raise SystemExit(f"В {weights} нет классов {sorted(wanted)}. Есть: {list(self.yolo.names.values())}")
        self.lock = threading.Lock()  # одна видеокарта — запросы по очереди
        print(f"YOLO {weights}: классы {[self.yolo.names[i] for i in self.classes]}")

    def segment(self, img_bgr: np.ndarray, p: dict) -> list[dict]:
        """-> [{"box", "conf", "cls", "mask", "contour"}], маски uint8 0/255 размером с фото."""
        with self.lock:
            r = self.yolo.predict(img_bgr, conf=p["conf"], iou=p["iou"], imgsz=p["imgsz"],
                                  classes=self.classes, retina_masks=True, verbose=False)[0]
        if r.masks is None:
            return []
        masks = (r.masks.data.cpu().numpy() > 0.5).astype(np.uint8) * 255
        boxes, confs, cls = r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()
        out = []
        for m, b, c, k in zip(masks, boxes, confs, cls):
            # от маски оставляем самый большой кусок: мелкие обрывки — шум сегментации
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
            if n > 2:
                m = (lab == 1 + stats[1:, cv2.CC_STAT_AREA].argmax()).astype(np.uint8) * 255
            if cv2.countNonZero(m) < max(1, p["min_area"]):
                continue
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = cv2.approxPolyDP(max(cnts, key=cv2.contourArea), 1.5, True).reshape(-1, 2)
            out.append({"box": b, "conf": c, "cls": self.yolo.names[int(k)], "mask": m,
                        "contour": contour.tolist()})
        return out
