"""
PatchCore с нуля на PyTorch (Roth et al., "Towards Total Recall in Industrial
Anomaly Detection", CVPR 2022).

Обучение только на нормальных (хороших) яблоках, модель -> models/patchcore.pt.
В memory bank идут ВСЕ фото из main_dataset/ со всеми подпапками (data1, data2, train_dataset, ...):
    python patchcore.py fit

Порог подбирается по 10% этих же фото (--val_ratio) «без себя»: каждое фото сравнивается
только с патчами ДРУГИХ фото, иначе оно найдёт в банке само себя и порог выйдет заниженным.
Если есть отдельная папка с нормальными фото для порога — порог считается по ней:
    python patchcore.py fit --val_dir some/good_val

Инференс (скор + тепловая карта для каждого фото, папка обходится рекурсивно):
    python patchcore.py predict --model models/patchcore.pt --images path/to/test --out_dir results

Зависимости: torch, torchvision, pillow, numpy (faiss не нужен).
"""
import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms.functional import gaussian_blur
import cv2

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# --------------------------------------------------------------------------- #
# Данные
# --------------------------------------------------------------------------- #
def list_images(root):
    paths = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXT)
    if not paths:
        raise FileNotFoundError(f"Нет изображений в {root}")
    return paths


def resize_image(img, image_size):
    # Вырезанные яблоки бывают не квадратными, а CenterCrop отрезал бы края плода
    # (где тоже бывают дефекты), поэтому просто приводим к квадрату.
    return img.resize((image_size, image_size), Image.BILINEAR)


_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet
])


_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def apply_clahe(img):
    """PIL RGB -> PIL RGB: выравнивание яркости (канал L в LAB), цвет (a, b) не меняется."""
    l, a, b = cv2.split(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2LAB))
    return Image.fromarray(cv2.cvtColor(cv2.merge([_clahe.apply(l), a, b]), cv2.COLOR_LAB2RGB))


def preprocess(img, image_size=224, clahe=True):
    """PIL RGB -> нормированный тензор [3, S, S], как при обучении.
    clahe должен совпадать с тем, с каким обучена модель (PatchCore.clahe)."""
    if clahe:
        img = apply_clahe(img)
    return _to_tensor(resize_image(img, image_size))


class ImageList(Dataset):
    """Список картинок без меток."""

    def __init__(self, paths, image_size=224, clahe=True):
        self.paths = [str(p) for p in paths]
        self.image_size = image_size
        self.clahe = clahe

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return preprocess(img, self.image_size, self.clahe), self.paths[i]


# --------------------------------------------------------------------------- #
# 1. Экстрактор признаков: замороженный WideResNet-50, слои layer2 + layer3
# --------------------------------------------------------------------------- #
class FeatureExtractor(torch.nn.Module):
    def __init__(self, layers=("layer2", "layer3")):
        super().__init__()
        bb = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        bb.eval()
        for p in bb.parameters():
            p.requires_grad_(False)
        self.backbone = bb
        self.layers = layers

    @torch.no_grad()
    def forward(self, x):
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        feats = {}
        x = b.layer1(x)
        x = b.layer2(x); feats["layer2"] = x   # [B, 512,  H/8,  W/8]
        x = b.layer3(x); feats["layer3"] = x   # [B, 1024, H/16, W/16]
        return [feats[l] for l in self.layers]


# --------------------------------------------------------------------------- #
# 2. Локально-осведомлённые патч-признаки
# --------------------------------------------------------------------------- #
def patch_embeddings(feature_maps, target_dim=1024, patch_size=3):
    """
    feature_maps: список карт признаков разных уровней.
    Возвращает: [B, H*W, target_dim] и размер сетки (H, W).
    """
    # (a) локальная агрегация: каждый патч "видит" окрестность 3x3
    agg = [F.avg_pool2d(f, kernel_size=patch_size, stride=1, padding=patch_size // 2)
           for f in feature_maps]
    # (b) приводим глубокий слой к разрешению самого "высокого" (layer2)
    h, w = agg[0].shape[-2:]
    agg = [agg[0]] + [F.interpolate(f, size=(h, w), mode="bilinear", align_corners=False)
                      for f in agg[1:]]
    x = torch.cat(agg, dim=1)                         # [B, 1536, h, w]
    B, C = x.shape[:2]
    x = x.permute(0, 2, 3, 1).reshape(B * h * w, 1, C)
    # (c) снижение размерности усреднением по каналам: 1536 -> 1024
    x = F.adaptive_avg_pool1d(x, target_dim).squeeze(1)
    return x.reshape(B, h * w, target_dim), (h, w)


# --------------------------------------------------------------------------- #
# 3. Coreset: жадный k-center отбор (с random projection для скорости)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def greedy_coreset(features, n, device, proj_dim=128, seed=0, chunk=65536):
    """
    Выбирает n патчей, которые максимально покрывают всё множество
    (минимизирует максимальное расстояние от любого патча до ближайшего выбранного).
    features может лежать на CPU (в fp16) — на device переносится только проекция.
    """
    N, D = features.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    # Johnson–Lindenstrauss: расстояния почти сохраняются, считать в 8 раз дешевле
    R = (torch.randn(D, proj_dim, generator=g) / math.sqrt(proj_dim)).to(device)
    z = torch.cat([f.to(device).float() @ R for f in features.split(chunk)])

    selected = [int(torch.randint(N, (1,), generator=g))]
    min_dist = torch.linalg.norm(z - z[selected[0]], dim=1)
    t0 = time.time()
    for i in range(1, n):
        idx = int(torch.argmax(min_dist))             # самый "непокрытый" патч
        selected.append(idx)
        min_dist = torch.minimum(min_dist, torch.linalg.norm(z - z[idx], dim=1))
        if i % 5000 == 0:
            el = time.time() - t0
            print(f"  coreset {i:,}/{n:,}  ({el:.0f} c, осталось ~{el / i * (n - i):.0f} c)")
    return torch.tensor(selected)


# --------------------------------------------------------------------------- #
# 4. Поиск ближайших соседей (батчами, без faiss)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def knn(query, bank, k=1, chunk=None, query_owner=None, bank_owner=None):
    """query_owner / bank_owner — номера фото: патчи банка из того же фото, что и запрос, пропускаются."""
    chunk = chunk or max(64, int(2e8 // bank.shape[0]))  # матрица расстояний ~800 МБ на чанк
    owners = query_owner.split(chunk) if query_owner is not None else [None] * len(query.split(chunk))
    dists, idxs = [], []
    for q, own in zip(query.split(chunk), owners):
        d = torch.cdist(q, bank)                      # [chunk, M]
        if own is not None:
            d.masked_fill_(own[:, None] == bank_owner[None, :], float("inf"))
        v, i = d.topk(k, dim=1, largest=False)
        dists.append(v); idxs.append(i)
    return torch.cat(dists), torch.cat(idxs)


# --------------------------------------------------------------------------- #
# Модель
# --------------------------------------------------------------------------- #
class PatchCore:
    def __init__(self, device=None, coreset_ratio=0.1, num_neighbors=9, target_dim=1024,
                 image_size=224, patch_ratio=1.0, clahe=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = FeatureExtractor().to(self.device)
        self.coreset_ratio = coreset_ratio
        self.patch_ratio = patch_ratio
        self.num_neighbors = num_neighbors
        self.target_dim = target_dim
        self.image_size = image_size
        self.clahe = clahe
        self.memory_bank = None
        self.bank_owner = None   # номер фото для каждого патча банка (только для порога, не сохраняется)
        self.threshold = None

    def _embed(self, x):
        return patch_embeddings(self.extractor(x.to(self.device)), self.target_dim)

    @torch.no_grad()
    def fit(self, loader):
        """Обучение = собрать патчи всех нормальных картинок и сжать их coreset'ом."""
        all_patches = []
        n_img = 0
        torch.manual_seed(0)
        for x, _ in loader:
            emb, _ = self._embed(x)
            B, P, D = emb.shape
            k = max(1, int(P * self.patch_ratio))
            if k < P:
                # из каждого фото берём случайную часть патчей: соседние патчи почти
                # одинаковые, а память и время coreset растут с их общим числом
                keep = torch.rand(B, P, device=emb.device).argsort(dim=1)[:, :k]
                emb = emb.gather(1, keep[..., None].expand(-1, -1, D))
            all_patches.append(emb.reshape(-1, D).half().cpu())  # fp16: вдвое меньше RAM
            n_img += x.shape[0]
            print(f"\r  признаки: {n_img}/{len(loader.dataset)}", end="", flush=True)
        print()
        bank = torch.cat(all_patches)
        del all_patches
        n = max(1, int(bank.shape[0] * self.coreset_ratio))
        print(f"Всего патчей: {bank.shape[0]:,} -> coreset {n:,}")
        idx = greedy_coreset(bank, n, self.device)
        self.memory_bank = bank[idx].float().contiguous().to(self.device)
        # с каждого фото подряд взято ровно k патчей -> строка r банка принадлежит фото r // k
        self.bank_owner = (idx // k).to(self.device)
        print(f"Memory bank после coreset: {self.memory_bank.shape[0]:,} x {self.target_dim}")

    @torch.no_grad()
    def predict(self, x, sigma=4, exclude=None):
        """Возвращает (image_scores [B], anomaly_maps [B, H, W]).
        exclude [B] — номера этих фото в обучении: их собственные патчи в банке не учитываются."""
        emb, (h, w) = self._embed(x)
        B, P, D = emb.shape
        flat = emb.reshape(-1, D)
        rep = exclude.repeat_interleave(P) if exclude is not None else None
        patch_scores, nn_idx = knn(flat, self.memory_bank, k=1, query_owner=rep, bank_owner=self.bank_owner)
        patch_scores = patch_scores.reshape(B, P)
        nn_idx = nn_idx.reshape(B, P)

        # --- скор изображения с перевзвешиванием (eq. 7 в статье) ---
        image_scores = []
        for b in range(B):
            p_star = int(patch_scores[b].argmax())           # самый аномальный патч
            s_star = patch_scores[b, p_star]
            m_test = emb[b, p_star]                          # его признак
            m_star = self.memory_bank[nn_idx[b, p_star]]     # его ближайший сосед в банке
            # b ближайших к m* элементов банка (окрестность "нормы")
            _, support = knn(m_star[None], self.memory_bank, k=self.num_neighbors,
                             query_owner=exclude[b:b + 1] if exclude is not None else None,
                             bank_owner=self.bank_owner)
            d = torch.linalg.norm(self.memory_bank[support[0]] - m_test, dim=1)
            weight = 1 - torch.softmax(d, dim=0)[0]          # если m* в "редком" районе — скор выше
            image_scores.append(weight * s_star)
        image_scores = torch.stack(image_scores)

        # --- карта аномалий: сетка патчей -> размер входа + гауссово сглаживание ---
        amap = patch_scores.reshape(B, 1, h, w)
        amap = F.interpolate(amap, size=x.shape[-2:], mode="bilinear", align_corners=False)
        ks = 2 * int(4 * sigma + 0.5) + 1
        amap = gaussian_blur(amap, kernel_size=ks, sigma=sigma).squeeze(1)
        return image_scores.cpu(), amap.cpu()

    @torch.no_grad()
    def calibrate(self, loader, percentile=99.0, owners=None):
        """Порог по нормальной валидации (дефектов нет — берём верхний перцентиль).
        owners — номера фото loader'а в обучении, если валидация взята из train (оценка «без себя»)."""
        scores, i = [], 0
        for x, _ in loader:
            ex = owners[i:i + len(x)].to(self.device) if owners is not None else None
            scores.append(self.predict(x, exclude=ex)[0])
            i += len(x)
        scores = torch.cat(scores).numpy()
        self.threshold = float(np.percentile(scores, percentile))
        print(f"Валидация ({len(scores)} фото): mean={scores.mean():.3f} std={scores.std():.3f} "
              f"max={scores.max():.3f} -> порог (p{percentile:g}) = {self.threshold:.3f}")

    def save(self, path):
        torch.save({"memory_bank": self.memory_bank.cpu(), "threshold": self.threshold,
                    "num_neighbors": self.num_neighbors, "target_dim": self.target_dim,
                    "image_size": self.image_size, "clahe": self.clahe}, path)

    @classmethod
    def load(cls, path, device=None):
        ckpt = torch.load(path, map_location="cpu")
        m = cls(device=device, num_neighbors=ckpt["num_neighbors"], target_dim=ckpt["target_dim"],
                image_size=ckpt.get("image_size", 224),
                clahe=ckpt.get("clahe", False))  # модели до CLAHE обучены без него
        m.memory_bank = ckpt["memory_bank"].to(m.device)
        m.threshold = ckpt["threshold"]
        return m


# --------------------------------------------------------------------------- #
# Визуализация
# --------------------------------------------------------------------------- #
def overlay(img, amap, image_size=224, vmax=None):
    """PIL RGB + карта аномалий -> PIL RGB с наложенной тепловой картой."""
    img = resize_image(img, image_size)
    a = amap.numpy()
    a = np.clip(a / (vmax or a.max() + 1e-8), 0, 1)
    heat = np.zeros((*a.shape, 3), dtype=np.float32)      # простая карта "синий -> красный"
    heat[..., 0] = a
    heat[..., 2] = 1 - a
    out = 0.55 * np.asarray(img, dtype=np.float32) / 255 + 0.45 * heat
    return Image.fromarray((out * 255).astype(np.uint8))


def save_overlay(img_path, amap, out_path, image_size=224, vmax=None):
    overlay(Image.open(img_path).convert("RGB"), amap, image_size, vmax).save(out_path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def split_train_val(args):
    """train = все фото data_dir (со всеми подпапками); val = --val_dir или None."""
    paths = list_images(args.data_dir)
    # перемешиваем, т.к. файлы отсортированы по папкам и сортам яблок (важно для --max_train)
    random.Random(args.seed).shuffle(paths)
    if args.max_train and len(paths) > args.max_train:
        paths = paths[:args.max_train]
    return paths, (list_images(args.val_dir) if args.val_dir else None)


def folder_counts(paths, root):
    """{"data1": 2220, ...} — сколько фото из каждой верхней подпапки root."""
    counts = {}
    for p in paths:
        parts = Path(p).relative_to(root).parts
        top = parts[0] if len(parts) > 1 else "."
        counts[top] = counts.get(top, 0) + 1
    return dict(sorted(counts.items()))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit")
    f.add_argument("--data_dir", default="main_dataset",
                   help="папка с НОРМАЛЬНЫМИ фото; все подпапки, все фото идут в обучение")
    f.add_argument("--val_dir", default=None,
                   help="отдельные НОРМАЛЬНЫЕ фото для порога; если нет — часть train «без себя»")
    f.add_argument("--val_ratio", type=float, default=0.1,
                   help="какая доля train проверяется «без себя» для порога (без --val_dir)")
    f.add_argument("--max_train", type=int, default=0,
                   help="ограничить число фото для обучения (0 = все)")
    f.add_argument("--patch_ratio", type=float, default=0.25,
                   help="доля патчей с каждого фото (из 784); 1.0 = все, но нужно ~15 ГБ RAM на весь датасет")
    f.add_argument("--out", default="models/patchcore.pt")
    f.add_argument("--coreset_ratio", type=float, default=0.1)
    f.add_argument("--image_size", type=int, default=224)
    f.add_argument("--no_clahe", action="store_true", help="обучать без выравнивания яркости CLAHE")
    f.add_argument("--batch_size", type=int, default=16)
    f.add_argument("--workers", type=int, default=0)
    f.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("predict")
    p.add_argument("--model", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--out_dir", default="results")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--workers", type=int, default=0)

    args = ap.parse_args()

    if args.cmd == "fit":
        train, val = split_train_val(args)
        model = PatchCore(coreset_ratio=args.coreset_ratio, image_size=args.image_size,
                          patch_ratio=args.patch_ratio, clahe=not args.no_clahe)
        print(f"Устройство: {model.device} | CLAHE: {'да' if model.clahe else 'нет'} | train: {len(train)} фото "
              + ", ".join(f"{k}={v}" for k, v in folder_counts(train, args.data_dir).items()))
        if model.device == "cpu":
            print("Внимание: CUDA недоступна, на CPU coreset и признаки считаются долго.")
        dl = dict(batch_size=args.batch_size, num_workers=args.workers)
        # порядок фото в fit важен: по нему bank_owner узнаёт, из какого фото патч
        model.fit(DataLoader(ImageList(train, args.image_size, model.clahe), **dl))
        if val:
            model.calibrate(DataLoader(ImageList(val, args.image_size, model.clahe), **dl))
        else:
            n_cal = max(1, int(len(train) * args.val_ratio))
            cal = sorted(random.Random(args.seed).sample(range(len(train)), n_cal))
            model.calibrate(DataLoader(ImageList([train[i] for i in cal], args.image_size, model.clahe), **dl),
                            owners=torch.tensor(cal))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        model.save(args.out)
        print(f"Сохранено в {args.out}")

    else:
        model = PatchCore.load(args.model)
        out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
        root = Path(args.images)
        paths = list_images(root)
        loader = DataLoader(ImageList(paths, model.image_size, model.clahe), batch_size=args.batch_size,
                            num_workers=args.workers)
        rows = []
        for x, batch_paths in loader:
            scores, maps = model.predict(x)
            for path, s, m in zip(batch_paths, scores, maps):
                is_anom = s.item() > model.threshold
                rows.append([path, f"{s.item():.4f}", int(is_anom)])
                # в разных подпапках одинаковые имена файлов -> имя из относительного пути
                rel = Path(path).relative_to(root).with_suffix("") if root.is_dir() else Path(Path(path).stem)
                name = "__".join(rel.parts) + "_heatmap.png"
                save_overlay(path, m, out / name, image_size=model.image_size,
                             vmax=model.threshold * 1.5)
                print(f"{'АНОМАЛИЯ' if is_anom else 'норма   '}  {s.item():7.3f}  {path}")
        with open(out / "scores.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["path", "score", "is_anomaly"])
            w.writerows(rows)
        n_anom = sum(r[2] for r in rows)
        print(f"Готово: {len(rows)} фото, аномалий {n_anom}. Результаты в {out}")


if __name__ == "__main__":
    main()
