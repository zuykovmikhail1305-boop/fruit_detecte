"""Отбор всех фото с яблоками из augmented/ в dataset_main/.

Разметка лежит в augmented/annotations.xml (формат CVAT for images): у каждого <image>
есть фигуры (<mask>, <box>, <polygon>...) с атрибутом label. Берём фото, на которых есть
хотя бы одна фигура с меткой Apple, копируем их и сохраняем XML только с этими фото.

Результат:
    dataset_main/images/*.jpg
    dataset_main/annotations.xml

Запуск:
    python make_dataset_main.py
    python make_dataset_main.py --label Apple --src augmented --out dataset_main
"""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description="Собрать dataset_main из фото с яблоками")
    ap.add_argument("--src", type=Path, default=ROOT / "augmented")
    ap.add_argument("--xml", type=Path, default=None, help="по умолчанию <src>/annotations.xml")
    ap.add_argument("--images", type=Path, default=None, help="по умолчанию <src>/ext_data")
    ap.add_argument("--out", type=Path, default=ROOT / "dataset_main")
    ap.add_argument("--label", default="Apple", help="какую метку отбирать (без учёта регистра)")
    args = ap.parse_args()

    xml_path = args.xml or args.src / "annotations.xml"
    img_dir = args.images or args.src / "ext_data"
    label = args.label.lower()

    tree = ET.parse(xml_path)
    root = tree.getroot()

    out_img_dir = args.out / "images"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    kept, dropped, missing = 0, 0, []
    for image in root.findall("image"):
        has_label = any((shape.get("label") or "").lower() == label for shape in image)
        if not has_label:
            root.remove(image)
            dropped += 1
            continue
        name = image.get("name")
        src_file = img_dir / name
        if not src_file.is_file():
            missing.append(name)
            root.remove(image)
            continue
        dst_file = out_img_dir / Path(name).name
        shutil.copy2(src_file, dst_file)
        kept += 1

    ET.indent(tree, space="  ")
    tree.write(args.out / "annotations.xml", encoding="utf-8", xml_declaration=True)

    print(f"Метка: {args.label}")
    print(f"Скопировано фото: {kept} -> {out_img_dir}")
    print(f"Пропущено (без метки): {dropped}")
    if missing:
        print(f"Нет файла для {len(missing)} записей из XML, например: {missing[:5]}")
    print(f"Разметка: {args.out / 'annotations.xml'}")


if __name__ == "__main__":
    main()
