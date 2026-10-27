"""Phase 3: Package the new experiment data (baseline images + LaMa outputs) into c2_lama.zip.

Packages:
  - metadata.json
  - orig/*.png
  - lama/flood_only_removed_lama/*.png
  - lama/c1_area_matched_lama/*.png
  - lama_manifest.json

Archive paths match the condition-aware inference resolver (Phase 4).
"""

import json
import zipfile
from pathlib import Path

ROOT = Path("data/c2")
LAMA = ROOT / "lama"
OUT = ROOT / "c2_lama.zip"


def package_lama_dataset():
    if not (LAMA / "manifest.json").exists():
        raise FileNotFoundError(f"Missing {LAMA / 'manifest.json'}. Run generate_lama_c2.py first.")

    with open(LAMA / "manifest.json", encoding="utf-8") as f:
        records = json.load(f)

    included = [r for r in records if r.get("status") == "ok"]
    valid_c1 = [r for r in included if r.get("control_status") == "ok"]
    print("=== Packaging C2 LaMa Dataset ===")
    print("Total tiles in manifest:", len(records))
    print("Valid LaMa flood-removal images:", len(included))
    print("Valid C1 controls:", len(valid_c1))

    total_files = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        # 1. Metadata
        z.write(ROOT / "metadata.json", "metadata.json")
        total_files += 1

        # 2. Original images
        orig_images = list((ROOT / "orig").glob("*.png"))
        for p in orig_images:
            z.write(p, f"orig/{p.name}")
            total_files += 1

        # 3. Baseline flood_only_removed images (for direct comparison in same notebook)
        camo_dir = ROOT / "flood_only_removed"
        if camo_dir.exists():
            for p in camo_dir.glob("*.png"):
                z.write(p, f"flood_only_removed/{p.name}")
                total_files += 1

        # 4. LaMa flood-removed images
        for r in included:
            tile_id = r["tile_id"]
            p = LAMA / "flood_only_removed_lama" / f"{tile_id}.png"
            if p.exists():
                z.write(p, f"lama/flood_only_removed_lama/{p.name}")
                total_files += 1

            if r.get("control_status") == "ok":
                cp = LAMA / "c1_area_matched_lama" / f"{tile_id}.png"
                if cp.exists():
                    z.write(cp, f"lama/c1_area_matched_lama/{p.name}")
                    total_files += 1

        # 5. Manifest
        z.write(LAMA / "manifest.json", "lama_manifest.json")
        total_files += 1

    size_mb = OUT.stat().st_size / (1024 * 1024)
    print(f"\nCreated: {OUT} ({size_mb:.2f} MB, {total_files} files)")


if __name__ == "__main__":
    package_lama_dataset()
