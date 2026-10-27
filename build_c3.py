"""Phase 5: Build C3 dry-tile dataset with prompt variants.

Selects verified dry tiles (water_fraction == 0.0, nodata_fraction <= 0.50),
samples with capped tiles per parent chip (seed 42), and configures the C3
prompt variants designed to test model susceptibility to leading prompts:
  - direct: "Is there flooding in this satellite image? Answer only Yes or No."
  - evidence: "Using only visible evidence in this satellite image, is flooding present? Answer only Yes or No."
  - conservative: "Do not infer flooding from rivers, vegetation, or terrain alone. Is there visible evidence of flooding? Answer only Yes or No."

Outputs:
  data/c3/orig/<tile_id>.png
  data/c3/metadata.json
  data/c3/c3.zip
"""

import argparse
import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

import config

C3_DIR = config.DATA_DIR / "c3"
PROCESSED_META = config.PROCESSED_128_DIR / "metadata.json"
MAX_NODATA = 0.5
MAX_PER_PARENT = 3
TARGET_DRY_TILES = 150

PROMPTS = {
    "leading_damage": "Describe the flood damage in this image.",
    "direct": "Is there flooding in this satellite image? Answer only Yes or No.",
    "evidence": "Using only visible evidence in this satellite image, is flooding present? Answer only Yes or No.",
    "conservative": "Do not infer flooding from rivers, vegetation, or terrain alone. Is there visible evidence of flooding? Answer only Yes or No.",
}


def build_c3(target_count=TARGET_DRY_TILES, seed=config.SEED):
    print("=== Phase 5: Building C3 Dry-Tile Dataset ===")
    if not PROCESSED_META.exists():
        raise FileNotFoundError(f"Missing {PROCESSED_META}. Run oversample_data.py first.")

    with open(PROCESSED_META, encoding="utf-8") as f:
        meta_128 = json.load(f)

    # 1. Filter dry tiles
    by_parent = defaultdict(list)
    for tid, info in meta_128.items():
        stats = info.get("stats", {})
        water_frac = stats.get("water_fraction", 1.0)
        nodata_frac = stats.get("nodata_fraction", 1.0)
        if water_frac == 0.0 and nodata_frac <= MAX_NODATA:
            parent = tid.split("_r")[0]
            by_parent[parent].append(tid)

    print(f"Candidate dry parent chips: {len(by_parent)}")
    total_candidates = sum(len(ts) for ts in by_parent.values())
    print(f"Total candidate dry tiles: {total_candidates}")

    # 2. Sample tiles (<= MAX_PER_PARENT per chip)
    rng = np.random.default_rng(seed)
    sampled = []
    parents = sorted(by_parent.keys())
    rng.shuffle(parents)

    for p in parents:
        ts = sorted(by_parent[p])
        rng.shuffle(ts)
        chosen = ts[:MAX_PER_PARENT]
        sampled.extend(chosen)
        if len(sampled) >= target_count:
            break

    sampled = sampled[:target_count]
    print(f"Selected {len(sampled)} dry tiles across {len({t.split('_r')[0] for t in sampled})} parents.")

    # 3. Setup directories & copy images
    orig_dir = C3_DIR / "orig"
    orig_dir.mkdir(parents=True, exist_ok=True)

    c3_metadata = {
        "meta": {
            "seed": seed,
            "max_nodata": MAX_NODATA,
            "max_per_parent": MAX_PER_PARENT,
            "target_count": target_count,
            "rule": "water_fraction == 0.0 and nodata_fraction <= 0.50",
            "prompts": PROMPTS,
        },
        "tiles": {},
    }

    for tid in sampled:
        src_png = config.PROCESSED_128_DIR / f"{tid}.png"
        dst_png = orig_dir / f"{tid}.png"
        if not dst_png.exists():
            shutil.copy2(src_png, dst_png)

        parent = tid.split("_r")[0]
        c3_metadata["tiles"][tid] = {
            "tile_id": tid,
            "parent_chip_id": parent,
            "image_path": f"orig/{tid}.png",
            "stats": meta_128[tid].get("stats", {}),
            "prompts": PROMPTS,
        }

    meta_out = C3_DIR / "metadata.json"
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(c3_metadata, f, indent=2)
    print(f"Saved metadata: {meta_out}")

    # 4. Package into data/c3/c3.zip
    out_zip = C3_DIR / "c3.zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(meta_out, "metadata.json")
        for img in orig_dir.glob("*.png"):
            z.write(img, f"orig/{img.name}")

    size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"Packaged archive: {out_zip} ({size_mb:.2f} MB, {len(sampled) + 1} files)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build C3 dry-tile dataset with prompt variants")
    parser.add_argument("--count", type=int, default=TARGET_DRY_TILES, help="Target number of dry tiles")
    args = parser.parse_args()
    build_c3(args.count)
