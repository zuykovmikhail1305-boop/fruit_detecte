"""Распаковка датасета Hugging Face из parquet-файлов в папки с картинками.

В каждом parquet колонки: image {bytes, path}, label (0 = fresh, 1 = rotten), fruit_type.
Результат:
    raw/<fruit_type>/<fresh|rotten>/<имя файла>
    raw/labels.csv  — путь, фрукт, метка, из какого parquet

Запуск:
    python unpack_parquet.py                          # все train-*.parquet из папки проекта -> raw/
    python unpack_parquet.py --src data --out raw --pattern "*.parquet"
"""

import argparse
import csv
import json
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent


def label_names(pf: pq.ParquetFile) -> list[str]:
    """Имена классов из метаданных Hugging Face (ClassLabel), иначе ["0", "1", ...]."""
    meta = (pf.schema_arrow.metadata or {}).get(b"huggingface")
    if meta:
        feat = json.loads(meta).get("info", {}).get("features", {}).get("label", {})
        if feat.get("names"):
            return feat["names"]
    return []


def main():
    ap = argparse.ArgumentParser(description="Распаковать parquet (Hugging Face) в папки с картинками")
    ap.add_argument("--src", type=Path, default=ROOT)
    ap.add_argument("--pattern", default="train-*.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "raw")
    args = ap.parse_args()

    files = sorted(args.src.glob(args.pattern))
    if not files:
        raise SystemExit(f"Нет файлов {args.pattern} в {args.src}")
    args.out.mkdir(parents=True, exist_ok=True)

    rows, counts = [], {}
    for f in files:
        pf = pq.ParquetFile(f)
        names = label_names(pf)
        # читаем по группам строк: целиком файл в 1 ГБ в память не тянем
        for g in range(pf.metadata.num_row_groups):
            for r in pf.read_row_group(g).to_pylist():
                label = names[r["label"]] if r["label"] < len(names) else str(r["label"])
                fruit = r.get("fruit_type") or "unknown"
                dst = args.out / fruit / label / Path(r["image"]["path"]).name
                dst.parent.mkdir(parents=True, exist_ok=True)
                # одинаковые имена в разных parquet не должны затирать друг друга
                k = 1
                while dst.exists():
                    dst = dst.with_name(f"{dst.stem.rsplit('__', 1)[0]}__{k}{dst.suffix}")
                    k += 1
                dst.write_bytes(r["image"]["bytes"])
                rows.append([dst.relative_to(args.out).as_posix(), fruit, label, f.name])
                counts[(fruit, label)] = counts.get((fruit, label), 0) + 1
        print(f"{f.name}: готово, всего {len(rows)} фото")

    with (args.out / "labels.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "fruit_type", "label", "source"])
        w.writerows(rows)

    for (fruit, label), n in sorted(counts.items()):
        print(f"  {fruit:12s} {label:7s} {n}")
    print(f"Распаковано {len(rows)} фото -> {args.out}")


if __name__ == "__main__":
    main()
