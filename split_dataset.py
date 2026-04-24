#!/usr/bin/env python3
"""Split image dataset into train/val/test (80/10/10) preserving folder structure.

Usage:
    python split_dataset.py --dataset dataset --output dataset_split

Options:
    --move    Move files instead of copying
    --seed N  Random seed (default 1337)
"""

from pathlib import Path
import random
import shutil
import argparse
import sys

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def find_class_dirs(dataset_dir: Path):
    class_dirs = []
    for root, dirs, files in __import__("os").walk(dataset_dir):
        p = Path(root)
        # ignore hidden folders
        if p.name.startswith("."):
            continue
        imgs = [f for f in files if Path(f).suffix.lower() in IMAGE_EXTS]
        if imgs:
            class_dirs.append(p)
    return class_dirs


def split_list(items, train_ratio=0.8, val_ratio=0.1):
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    # adjust in case of rounding issues
    if n_test < 0:
        n_test = 0
    return items[:n_train], items[n_train : n_train + n_val], items[n_train + n_val :]


def copy_or_move(src: Path, dst: Path, move: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def main():
    parser = argparse.ArgumentParser(description="Split image dataset into train/val/test (80/10/10)")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"), help="Path to source dataset")
    parser.add_argument("--output", type=Path, default=Path("dataset_split"), help="Output base directory")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    args = parser.parse_args()

    dataset_dir = args.dataset
    out_dir = args.output
    move_files = args.move

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        print(f"Dataset folder not found: {dataset_dir}")
        sys.exit(1)

    random.seed(args.seed)

    class_dirs = find_class_dirs(dataset_dir)
    if not class_dirs:
        print(f"No image files found under {dataset_dir}")
        sys.exit(1)

    summary = {"train": 0, "val": 0, "test": 0}

    for class_dir in class_dirs:
        rel = class_dir.relative_to(dataset_dir)
        parts = rel.parts
        # Determine top-level category (e.g., 'binary' or 'multiclass') and the class subpath
        if len(parts) >= 2:
            category = parts[0]
            rel_class = Path(*parts[1:])
        else:
            category = parts[0] if parts else "unknown"
            rel_class = Path()

        images = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        images = sorted(images)
        random.shuffle(images)
        train, val, test = split_list(images)

        for p in train:
            dest = out_dir / category / "train" / rel_class / p.name
            copy_or_move(p, dest, move_files)
        for p in val:
            dest = out_dir / category / "val" / rel_class / p.name
            copy_or_move(p, dest, move_files)
        for p in test:
            dest = out_dir / category / "test" / rel_class / p.name
            copy_or_move(p, dest, move_files)

        summary["train"] += len(train)
        summary["val"] += len(val)
        summary["test"] += len(test)

    print("Split complete:")
    print(f"  train: {summary['train']}")
    print(f"  val:   {summary['val']}")
    print(f"  test:  {summary['test']}")
    print(f"Output base: {out_dir}")


if __name__ == "__main__":
    main()
