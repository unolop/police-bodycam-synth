"""Dataset packaging: train/val/test split and final assembly."""

import json
import random
from pathlib import Path


def package_dataset(
    run_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """Split COCO annotations into train/val/test and write split files."""
    coco_path = run_dir / "annotations" / "coco.json"
    if not coco_path.exists():
        raise FileNotFoundError(f"COCO annotations not found: {coco_path}")

    coco = json.loads(coco_path.read_text())
    images = coco["images"]

    rng = random.Random(seed)
    rng.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    # Build image_id sets for each split
    split_ids = {name: {img["id"] for img in imgs} for name, imgs in splits.items()}

    for split_name, split_images in splits.items():
        ids = split_ids[split_name]
        split_anns = [a for a in coco["annotations"] if a["image_id"] in ids]

        split_coco = {
            "info": coco["info"],
            "licenses": coco["licenses"],
            "images": split_images,
            "annotations": split_anns,
            "categories": coco["categories"],
        }

        split_path = run_dir / "annotations" / f"coco_{split_name}.json"
        with open(split_path, "w") as f:
            json.dump(split_coco, f, indent=2, ensure_ascii=False)

        print(f"  {split_name}: {len(split_images)} images, {len(split_anns)} annotations")

    # Write metadata
    metadata = {
        "total_images": n,
        "splits": {name: len(imgs) for name, imgs in splits.items()},
        "categories": len(coco["categories"]),
        "total_annotations": len(coco["annotations"]),
    }
    meta_path = run_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset packaged: {meta_path}")
    return metadata
