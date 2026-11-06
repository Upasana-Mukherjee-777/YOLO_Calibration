#!/usr/bin/env python3
"""
COCO val2017 Dataset Downloader & Setup Utility
For: Reliable Calibration Transfer for Lightweight YOLO Detectors
Author: Upasana Mukherjee
"""

import os
import sys
import json
import zipfile
import argparse
import urllib.request
from pathlib import Path

DATA_DIR = Path("./data/coco")
VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\rProgress: {percent}% [{count * block_size // (1024*1024)} MB / {total_size // (1024*1024)} MB]")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, str(dest_path), reporthook=reporthook)
    print("\nDownload complete!")

def _write_simple_bmp(filepath, width, height, color):
    """Writes a simple uncompressed 24-bit BMP image without any third-party dependencies."""
    r, g, b = color
    row_bytes = width * 3
    padding = (4 - (row_bytes % 4)) % 4
    stride = row_bytes + padding
    pixel_data_size = stride * height
    file_size = 54 + pixel_data_size

    header = bytearray([
        0x42, 0x4D,                         # 'BM'
        file_size & 0xFF, (file_size >> 8) & 0xFF, (file_size >> 16) & 0xFF, (file_size >> 24) & 0xFF,
        0, 0, 0, 0,                         # Reserved
        54, 0, 0, 0,                        # Offset to pixel data
        40, 0, 0, 0,                        # DIB header size
        width & 0xFF, (width >> 8) & 0xFF, (width >> 16) & 0xFF, (width >> 24) & 0xFF,
        height & 0xFF, (height >> 8) & 0xFF, (height >> 16) & 0xFF, (height >> 24) & 0xFF,
        1, 0,                               # Planes
        24, 0,                              # Bits per pixel
        0, 0, 0, 0,                         # Compression (None)
        pixel_data_size & 0xFF, (pixel_data_size >> 8) & 0xFF, (pixel_data_size >> 16) & 0xFF, (pixel_data_size >> 24) & 0xFF,
        0x13, 0x0B, 0, 0,                   # Horizontal resolution (2835 px/m)
        0x13, 0x0B, 0, 0,                   # Vertical resolution (2835 px/m)
        0, 0, 0, 0,                         # Colors in palette
        0, 0, 0, 0                          # Important colors
    ])

    row = bytearray()
    for _ in range(width):
        row.extend([b, g, r])
    row.extend([0] * padding)

    with open(filepath, "wb") as f:
        f.write(header)
        for _ in range(height):
            f.write(row)

def setup_quick_subset(target_dir: Path, n_images=50):
    """
    Sets up a lightweight synthetic / sample subset for quick verification
    without downloading 1GB of COCO images (Zero dependencies required).
    """
    print(f"\n[Quick Setup] Setting up quick verification dataset ({n_images} images) in {target_dir}...")
    img_dir = target_dir / "val2017"
    ann_dir = target_dir / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    coco_classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    ]
    categories = [{"id": i, "name": name, "supercategory": "object"} for i, name in enumerate(coco_classes)]

    images = []
    annotations = []
    ann_id = 1

    try:
        import cv2
        import numpy as np
        has_cv2 = True
    except ImportError:
        has_cv2 = False

    for i in range(1, n_images + 1):
        fname = f"sample_{i:06d}.jpg"
        fpath = img_dir / fname
        w, h = 640, 480
        color = (40 + (i * 7) % 180, 50 + (i * 11) % 180, 70 + (i * 13) % 180)

        if has_cv2:
            img = np.full((h, w, 3), color, dtype=np.uint8)
            cv2.line(img, (0, h), (w, int(h * 0.4)), (200, 200, 200), 2)
            cv2.rectangle(img, (100, 150), (250, 300), (0, 180, 0), -1)
            cv2.putText(img, f"Test {i}", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imwrite(str(fpath), img)
        else:
            # Fallback pure python bmp saved as jpg
            _write_simple_bmp(fpath, w, h, color)

        images.append({"id": i, "file_name": fname, "width": w, "height": h})
        annotations.append({
            "id": ann_id,
            "image_id": i,
            "category_id": 2, # car
            "bbox": [100, 150, 150, 150],
            "area": 22500,
            "iscrowd": 0
        })
        ann_id += 1

    ann_json = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    ann_file = ann_dir / "instances_val2017.json"
    with open(ann_file, "w") as f:
        json.dump(ann_json, f, indent=2)

    print(f"\n[SUCCESS] Quick dataset generated at {target_dir.resolve()}!")
    print(f"  - Images: {len(images)} in {img_dir}")
    print(f"  - Annotation file: {ann_file}")
    print("\nNext step: Run calibration benchmark:")
    print(f"  python run_calibration.py --data_dir {target_dir} --n_images {n_images}")

def download_full_coco(target_dir: Path):
    """Downloads official COCO val2017 images and annotations (~1.2 GB)."""
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_img = target_dir / "val2017.zip"
    zip_ann = target_dir / "annotations_trainval2017.zip"

    # 1. Download Annotations
    ann_file = target_dir / "annotations" / "instances_val2017.json"
    if not ann_file.exists():
        if not zip_ann.exists():
            download_file(ANNOTATIONS_URL, zip_ann)
        print("Extracting annotations...")
        with zipfile.ZipFile(zip_ann, "r") as z:
            z.extractall(target_dir)
        print("Annotations extracted!")
    else:
        print(f"Annotations already present at {ann_file}")

    # 2. Download Images
    img_dir = target_dir / "val2017"
    if not img_dir.exists() or len(list(img_dir.glob("*.jpg"))) < 5000:
        if not zip_img.exists():
            download_file(VAL_IMAGES_URL, zip_img)
        print("Extracting val2017 images (~5,000 images, ~1 GB)...")
        with zipfile.ZipFile(zip_img, "r") as z:
            z.extractall(target_dir)
        print("Images extracted!")
    else:
        print(f"Images already present at {img_dir}")

    print("\n[SUCCESS] Full COCO val2017 dataset is ready!")
    print(f"Path: {target_dir.resolve()}")
    print("\nRun the full calibration benchmark with:")
    print(f"  python run_calibration.py --data_dir {target_dir} --n_images 5000")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download & Setup COCO val2017 for YOLO Calibration Benchmark")
    parser.add_argument("--full", action="store_true", help="Download official full COCO val2017 dataset (~1.2 GB)")
    parser.add_argument("--quick", action="store_true", help="Generate a fast verification dataset (50 images) in 2 seconds")
    parser.add_argument("--dest", type=str, default="./data/coco", help="Destination directory (default: ./data/coco)")
    parser.add_argument("--n_images", type=int, default=50, help="Number of images for quick setup (default: 50)")

    args = parser.parse_args()
    dest = Path(args.dest)

    if args.full:
        download_full_coco(dest)
    else:
        setup_quick_subset(dest, n_images=args.n_images)
