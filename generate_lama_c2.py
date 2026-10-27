"""Phase 2: Generate LaMa inpainting images and area-matched C1 dry-region controls.

Replaces the baseline speckled dry-pixel fill with deep inpainting (LaMa), and constructs
an area-matched control that inprints an identical-shape patch over valid dry land while
leaving the permanent water and flood water intact.

Outputs:
  data/c2/lama/flood_only_removed_lama/<tile_id>.png
  data/c2/lama/c1_area_matched_lama/<tile_id>.png
  data/c2/lama/manifest.json
  results/figures/lama_c1_examples.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image
from simple_lama_inpainting import SimpleLama
from tqdm import tqdm

import config
from data import load_pair

C2_DIR = config.DATA_DIR / "c2"
META = C2_DIR / "metadata.json"
OUT = C2_DIR / "lama"
JRC_DIR = config.SEN1FLOODS_DIR / "data" / "JRCWaterHand"


def inpaint(lama, rgb, mask):
    """Inpaints the regions where mask is True (white mask pixels)."""
    image = Image.fromarray(rgb.astype(np.uint8))
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255))
    result = lama(image, mask_image)
    return np.asarray(result.convert("RGB"))


def find_control_mask(flood_mask, dry_mask, seed=42):
    """Translates the flood-mask shape to a fully labelled dry region.

    Ensures the control region has exactly identical shape, area, and orientation,
    placed exclusively on validated dry pixels (dry_mask == True).
    """
    h, w = flood_mask.shape
    ys, xs = np.where(flood_mask)
    if len(ys) == 0:
        return None

    min_dy, max_dy = -int(ys.min()), int(h - 1 - ys.max())
    min_dx, max_dx = -int(xs.min()), int(w - 1 - xs.max())
    if min_dy > max_dy or min_dx > max_dx:
        return None

    offsets = [(dy, dx) for dy in range(min_dy, max_dy + 1)
               for dx in range(min_dx, max_dx + 1)
               if (dy, dx) != (0, 0)]
    if not offsets:
        return None

    rng = np.random.default_rng(seed)
    rng.shuffle(offsets)

    for dy, dx in offsets:
        ny, nx = ys + dy, xs + dx
        if dry_mask[ny, nx].all():
            control = np.zeros((h, w), dtype=bool)
            control[ny, nx] = True
            return control
    return None


def save_comparison_figure(sample_tiles, out_path):
    """Saves a multi-row visual comparison figure of baseline vs LaMa vs C1."""
    if not sample_tiles:
        return
    n = len(sample_tiles)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))
    if n == 1:
        axes = [axes]

    for row_idx, info in enumerate(sample_tiles):
        tile_id = info["tile_id"]
        orig_img = Image.open(C2_DIR / "orig" / f"{tile_id}.png")
        camo_img = Image.open(C2_DIR / "flood_only_removed" / f"{tile_id}.png")
        lama_img = Image.open(OUT / "flood_only_removed_lama" / f"{tile_id}.png")
        
        c1_path = OUT / "c1_area_matched_lama" / f"{tile_id}.png"
        c1_img = Image.open(c1_path) if c1_path.exists() else None

        axes[row_idx][0].imshow(orig_img)
        axes[row_idx][0].set_title(f"{tile_id}\nOriginal (River + Flood)", fontsize=10)

        axes[row_idx][1].imshow(camo_img)
        axes[row_idx][1].set_title("Baseline Camouflage\n(Speckled Fill)", fontsize=10)

        axes[row_idx][2].imshow(lama_img)
        axes[row_idx][2].set_title("LaMa Inpainting\n(Flood Removed)", fontsize=10)

        if c1_img is not None:
            axes[row_idx][3].imshow(c1_img)
            axes[row_idx][3].set_title("C1 Control (LaMa)\n(Dry Land Inpainted)", fontsize=10)
        else:
            axes[row_idx][3].text(0.5, 0.5, "No Valid Dry\nControl Location", ha="center", va="center")
            axes[row_idx][3].set_title("C1 Control (N/A)", fontsize=10)

        for ax in axes[row_idx]:
            ax.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved qualitative comparison figure: {out_path}")


def main(max_tiles=None):
    print("=== Phase 2: Generating LaMa Inpainting & C1 Area-Matched Controls ===")
    if not META.exists():
        raise FileNotFoundError(f"Missing {META}. Run build_c2.py first.")

    with open(META, encoding="utf-8") as f:
        meta_data = json.load(f)
        tiles = meta_data["tiles"]

    print("Loading SimpleLama model...")
    lama = SimpleLama()

    flood_out = OUT / "flood_only_removed_lama"
    control_out = OUT / "c1_area_matched_lama"
    flood_out.mkdir(parents=True, exist_ok=True)
    control_out.mkdir(parents=True, exist_ok=True)

    mixed_tiles = [(tid, t) for tid, t in tiles.items() if t["group"] == "mixed"]
    if max_tiles is not None:
        mixed_tiles = mixed_tiles[:max_tiles]
        print(f"Testing on first {max_tiles} mixed tiles.")
    else:
        print(f"Processing {len(mixed_tiles)} mixed tiles.")

    manifest = []
    parent_cache = {}  # Cache parent chip labels to minimize disk I/O

    for idx, (tile_id, t) in enumerate(tqdm(mixed_tiles, desc="Generating LaMa & C1")):
        parent = t["parent_chip_id"]
        r0, c0, r1, c1 = t["tile_bbox_in_parent"]

        if parent not in parent_cache:
            _, full_label = load_pair(parent)
            jrc = tifffile.imread(JRC_DIR / f"{parent}_JRCWaterHand.tif")
            parent_cache[parent] = (full_label, jrc)
        else:
            full_label, jrc = parent_cache[parent]

        label = full_label[r0:r1, c0:c1]
        permanent = jrc[r0:r1, c0:c1] == 1

        # flood = hand-labelled water & not permanent & valid
        flood = (label == 1) & ~permanent & (label != -1)
        dry = label == 0

        if not flood.any():
            manifest.append({
                "tile_id": tile_id,
                "status": "skipped_no_flood_pixels",
                "flood_pixels": 0,
                "control_status": "skipped"
            })
            continue

        orig_path = C2_DIR / t["image_path"]
        rgb = np.asarray(Image.open(orig_path).convert("RGB"))

        # 1. Generate LaMa flood-removed image
        lama_flood_img = inpaint(lama, rgb, flood)
        Image.fromarray(lama_flood_img).save(flood_out / f"{tile_id}.png")

        # 2. Find C1 area-matched dry control mask
        # Use seed seeded by index for diversity across tiles while maintaining reproducibility
        control = find_control_mask(flood, dry, seed=42 + idx)
        rec = {
            "tile_id": tile_id,
            "parent_chip_id": parent,
            "status": "ok",
            "flood_pixels": int(flood.sum()),
            "flood_fraction": float(flood.sum() / flood.size)
        }

        if control is not None:
            lama_control_img = inpaint(lama, rgb, control)
            Image.fromarray(lama_control_img).save(control_out / f"{tile_id}.png")
            rec["control_status"] = "ok"
            rec["control_pixels"] = int(control.sum())
        else:
            rec["control_status"] = "no_valid_dry_location"
            rec["control_pixels"] = 0

        manifest.append(rec)

    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    ok_flood = sum(r["status"] == "ok" for r in manifest)
    ok_control = sum(r.get("control_status") == "ok" for r in manifest)
    print("\n=== Generation Summary ===")
    print(f"Total tiles processed: {len(manifest)}")
    print(f"LaMa flood-removed generated: {ok_flood}")
    print(f"Valid C1 area-matched controls: {ok_control} ({ok_control / max(ok_flood, 1):.1%})")
    print(f"Manifest written to: {manifest_path}")

    # Generate visual figure with a small sample
    sample_recs = [r for r in manifest if r.get("status") == "ok" and r.get("control_status") == "ok"][:4]
    fig_path = config.FIGURES_DIR / "lama_c1_examples.png"
    save_comparison_figure(sample_recs, fig_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LaMa inpainting and C1 controls for C2")
    parser.add_argument("--max-tiles", type=int, default=None, help="Limit number of mixed tiles")
    args = parser.parse_args()
    main(args.max_tiles)
