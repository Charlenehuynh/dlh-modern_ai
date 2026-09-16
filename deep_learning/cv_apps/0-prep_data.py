#!/usr/bin/env python3
"""
Prepare a YOLOv8-format detection dataset from Pascal VOC 2012, keeping
only the classes: person, car, bicycle.

"""

import argparse
import os
import random
import shutil
import sys
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

VOC_URL = (
    "https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/" "VOCtrainval_11-May-2012.tar"
)
TAR_NAME = "VOCtrainval_11-May-2012.tar"
KEPT_CLASSES = ["person", "car", "bicycle"]  # index = YOLO class id
CLASS_TO_ID = {name: idx for idx, name in enumerate(KEPT_CLASSES)}
VAL_FRACTION = 0.2  # only used for the fallback random split
RANDOM_SEED = 42
SKIP_DIFFICULT = True  # set via --include-difficult


def download_voc(dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    tar_path = dest_dir / TAR_NAME
    if tar_path.exists():
        print(f"[download] {tar_path} already exists, skipping download.")
        return tar_path

    print(
        f"[download] Downloading {VOC_URL} -> {tar_path} "
        "(this is ~1.9 GB, may take a while)"
    )

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            mb_done = downloaded / 1e6
            mb_total = total_size / 1e6
            sys.stdout.write(
                f"\r  {pct:5.1f}% ({mb_done:8.1f} MB / {mb_total:8.1f} MB)"
            )
        else:
            sys.stdout.write(f"\r  {downloaded / 1e6:8.1f} MB downloaded")
        sys.stdout.flush()

    urllib.request.urlretrieve(VOC_URL, tar_path, reporthook=_progress)
    print()
    return tar_path


def extract_voc(tar_path: Path, extract_dir: Path) -> Path:
    """Extract only Annotations/ and JPEGImages/ (and ImageSets/Main for
    the fallback split) from the VOC tar.

    Returns the path to VOCdevkit/VOC2012.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    voc_root = extract_dir / "VOCdevkit" / "VOC2012"
    if (
        voc_root.exists()
        and (voc_root / "Annotations").exists()
        and (voc_root / "JPEGImages").exists()
    ):
        print(f"[extract] {voc_root} already extracted, skipping.")
        return voc_root

    print(
        "[extract] Extracting Annotations/, JPEGImages/, "
        f"ImageSets/Main from {tar_path} ..."
    )
    wanted_prefixes = (
        "VOCdevkit/VOC2012/Annotations/",
        "VOCdevkit/VOC2012/JPEGImages/",
        "VOCdevkit/VOC2012/ImageSets/Main/",
    )
    with tarfile.open(tar_path, "r") as tf:
        members = [m for m in tf.getmembers() if m.name.startswith(wanted_prefixes)]
        tf.extractall(path=extract_dir, members=members)

    print(f"[extract] Done -> {voc_root}")
    return voc_root


def read_sample_list(path: Path) -> set:
    names = set()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # normalize: strip extension if present, we re-add .jpg later
            stem = Path(line).stem
            names.add(stem)
    return names


def parse_voc_annotation(xml_path: Path):
    """Parse one VOC XML annotation.

    Returns a list of (class_name, xmin, ymin, xmax, ymax, is_difficult)
    tuples plus the (img_w, img_h) image size.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        difficult = obj.find("difficult")
        is_difficult = difficult is not None and int(difficult.text) == 1
        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)
        boxes.append((name, xmin, ymin, xmax, ymax, is_difficult))
    return boxes, img_w, img_h


def voc_box_to_yolo_line(cls_id, xmin, ymin, xmax, ymax, img_w, img_h):
    xmin = max(0.0, xmin)
    ymin = max(0.0, ymin)
    xmax = min(float(img_w), xmax)
    ymax = min(float(img_h), ymax)

    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"


def build_fallback_split(voc_root: Path):
    """When train_samples.txt/val_samples.txt are not provided, build a
    split from VOC's own train/val sets, filtered to images that contain
    at least one of the kept classes."""
    main_dir = voc_root / "ImageSets" / "Main"
    ann_dir = voc_root / "Annotations"

    def _filter(names):
        keep = []
        for name in names:
            xml_path = ann_dir / f"{name}.xml"
            if not xml_path.exists():
                continue
            boxes, _, _ = parse_voc_annotation(xml_path)
            if any(
                b[0] in CLASS_TO_ID and not (b[5] and SKIP_DIFFICULT) for b in boxes
            ):
                keep.append(name)
        return keep

    train_txt = main_dir / "train.txt"
    val_txt = main_dir / "val.txt"
    if train_txt.exists() and val_txt.exists():
        with open(train_txt) as f:
            train_names = [line.strip() for line in f if line.strip()]
        with open(val_txt) as f:
            val_names = [line.strip() for line in f if line.strip()]
        train_names = _filter(train_names)
        val_names = _filter(val_names)
        print(
            "[split] Using VOC's own train/val split, filtered to "
            f"kept classes: {len(train_names)} train, "
            f"{len(val_names)} val."
        )
        return set(train_names), set(val_names)

    # last resort: build our own split from all annotated images
    all_names = [p.stem for p in ann_dir.glob("*.xml")]
    all_names = _filter(all_names)
    random.Random(RANDOM_SEED).shuffle(all_names)
    n_val = int(len(all_names) * VAL_FRACTION)
    val_names = set(all_names[:n_val])
    train_names = set(all_names[n_val:])
    print(
        f"[split] No ImageSets found; built random split: "
        f"{len(train_names)} train, {len(val_names)} val."
    )
    return train_names, val_names


def convert_split(names, voc_root: Path, images_out: Path, labels_out: Path):
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    ann_dir = voc_root / "Annotations"
    img_dir = voc_root / "JPEGImages"

    kept = 0
    skipped_no_class = 0
    skipped_missing = 0

    for name in sorted(names):
        xml_path = ann_dir / f"{name}.xml"
        img_path = img_dir / f"{name}.jpg"
        if not xml_path.exists() or not img_path.exists():
            skipped_missing += 1
            continue

        boxes, img_w, img_h = parse_voc_annotation(xml_path)
        yolo_lines = [
            voc_box_to_yolo_line(CLASS_TO_ID[cls], xmin, ymin, xmax, ymax, img_w, img_h)
            for (cls, xmin, ymin, xmax, ymax, is_difficult) in boxes
            if cls in CLASS_TO_ID and not (is_difficult and SKIP_DIFFICULT)
        ]
        if not yolo_lines:
            skipped_no_class += 1
            continue

        shutil.copy2(img_path, images_out / f"{name}.jpg")
        with open(labels_out / f"{name}.txt", "w") as f:
            f.write("\n".join(yolo_lines) + "\n")
        kept += 1

    print(
        f"  kept={kept}  "
        f"skipped(no target class)={skipped_no_class}  "
        f"skipped(missing file)={skipped_missing}"
    )
    return kept


def write_data_yaml(dataset_root: Path):
    content = (
        "path: datasets/detection/\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "nc: 3\n"
        f"names: {KEPT_CLASSES}\n"
    )
    (dataset_root / "data.yaml").write_text(content)
    print(f"[data.yaml] Written to {dataset_root / 'data.yaml'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-dir",
        default="cv_apps",
        help=(
            "Project root that will contain datasets/detection/ " "(default: cv_apps)"
        ),
    )
    parser.add_argument(
        "--work-dir",
        default="voc_work",
        help=(
            "Scratch directory for the downloaded tar and extracted "
            "VOC files (default: voc_work)"
        ),
    )
    parser.add_argument(
        "--train-samples",
        default="train_samples.txt",
        help="Path to a file listing image names for train (optional)",
    )
    parser.add_argument(
        "--val-samples",
        default="val_samples.txt",
        help="Path to a file listing image names for val (optional)",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help=(
            "Keep the downloaded tar and extracted VOCdevkit folder "
            "instead of deleting them"
        ),
    )
    parser.add_argument(
        "--voc-root",
        default=None,
        help=(
            "Path to an already-extracted VOC2012 folder (containing "
            "Annotations/, JPEGImages/, ImageSets/), e.g. one a "
            "teammate already downloaded. If set, skips "
            "downloading/extracting entirely."
        ),
    )
    parser.add_argument(
        "--include-difficult",
        action="store_true",
        help=(
            "Include objects marked difficult=1 in VOC annotations "
            "(excluded by default)"
        ),
    )
    args = parser.parse_args()

    global SKIP_DIFFICULT
    SKIP_DIFFICULT = not args.include_difficult

    project_dir = Path(args.project_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    dataset_root = project_dir / "datasets" / "detection"

    if args.voc_root:
        voc_root = Path(args.voc_root).resolve()
        if (
            not (voc_root / "Annotations").exists()
            or not (voc_root / "JPEGImages").exists()
        ):
            print(
                f"Error: --voc-root '{voc_root}' does not look like a "
                "VOC2012 folder (missing Annotations/ or JPEGImages/)."
            )
            sys.exit(1)
        print(
            f"[voc] Using existing local VOC data at {voc_root}, "
            "skipping download/extract."
        )
    else:
        # 1. download
        tar_path = download_voc(work_dir)
        # 2. extract
        voc_root = extract_voc(tar_path, work_dir)

    # 3. determine split
    train_file = Path(args.train_samples)
    val_file = Path(args.val_samples)
    if train_file.exists() and val_file.exists():
        train_names = read_sample_list(train_file)
        val_names = read_sample_list(val_file)
        print(
            "[split] Using provided sample lists: "
            f"{len(train_names)} train, {len(val_names)} val."
        )
    else:
        print(
            f"[split] '{train_file}' / '{val_file}' not found, "
            "falling back to VOC's own split."
        )
        train_names, val_names = build_fallback_split(voc_root)

    # 4-5. convert + copy
    print("[convert] Building train split ...")
    n_train = convert_split(
        train_names,
        voc_root,
        dataset_root / "images" / "train",
        dataset_root / "labels" / "train",
    )
    print("[convert] Building val split ...")
    n_val = convert_split(
        val_names,
        voc_root,
        dataset_root / "images" / "val",
        dataset_root / "labels" / "val",
    )

    # 6. data.yaml
    write_data_yaml(dataset_root)

    # 7. cleanup
    if args.voc_root:
        print(
            "[cleanup] --voc-root was used, nothing downloaded/"
            "extracted to clean up."
        )
    elif not args.keep_raw:
        print(f"[cleanup] Removing raw VOC data in {work_dir} ...")
        shutil.rmtree(work_dir, ignore_errors=True)
    else:
        print(f"[cleanup] --keep-raw set, leaving {work_dir} in place.")

    print("\nDone.")
    print(f"  Train images/labels: {n_train}")
    print(f"  Val images/labels:   {n_val}")
    print(f"  Dataset root:        {dataset_root}")


if __name__ == "__main__":
    main()
