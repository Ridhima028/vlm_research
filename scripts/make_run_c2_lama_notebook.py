"""Writes notebooks/run_c2_lama.ipynb (C2 LaMa inpainting and C1 control Yes/No inference on Colab).

Expands the baseline C2 experiment to include:
  - orig
  - flood_only_removed (baseline camo)
  - flood_only_removed_lama (deep inpainting)
  - c1_area_matched_lama (dry region inpainting control)

Outputs written to results/c2_lama/<model>_lama.jsonl without overwriting baseline outputs.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "run_c2_lama.ipynb"

CELLS = [
    ("markdown", r"""# C2 Extended: LaMa Inpainting & Area-Matched C1 Control (Yes/No)

Evaluates VLM sensitivity to deep inpainting (LaMa) vs speckled camouflage, and runs
the area-matched dry control (C1) where identical-area dry land is inpainted while floodwater is kept.

| group | conditions | prompts |
| --- | --- | --- |
| permanent_only | orig | yn_flood, yn_water |
| mixed | orig, flood_only_removed, flood_only_removed_lama, c1_area_matched_lama* | yn_flood, yn_water |

*\* c1_area_matched_lama is included only for tiles where a valid in-bounds dry control placement exists.*

**Before running:**
1. Upload `data/c2/c2_lama.zip` to `MyDrive/vlm_flood_research/data/c2_lama.zip`.
2. Select a GPU runtime (T4 is sufficient).
3. Pick `MODEL` ("qwen" or "llava") → Run all cells.
4. Outputs are saved to `MyDrive/vlm_flood_research/results/c2_lama/<model>_lama.jsonl`.
"""),

    ("code", """MODEL = "qwen"  # @param ["qwen", "llava"]
MODEL = MODEL.strip().lower()
assert MODEL in ("qwen", "llava"), MODEL"""),

    ("code", """# Install dependencies based on selected model
if MODEL == "qwen":
    !pip install -q -U transformers accelerate bitsandbytes qwen-vl-utils
else:
    !pip install -q "transformers==4.46.3" "accelerate>=0.26.0" "bitsandbytes>=0.42.0" "huggingface-hub>=0.25.0\""""),

    ("code", """import datetime
import json
import os
import time
import zipfile
from pathlib import Path

from google.colab import drive

drive.mount("/content/drive")

DRIVE_ROOT = "/content/drive/MyDrive/vlm_flood_research"
os.environ["HF_HOME"] = "/content/hf_cache"
ZIP_PATH = f"{DRIVE_ROOT}/data/c2_lama.zip"
C2_DIR = Path("/content/c2")
RESULTS_DIR = Path(f"{DRIVE_ROOT}/results/c2_lama")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = RESULTS_DIR / f"{MODEL}_lama.jsonl"

if not (C2_DIR / "metadata.json").exists():
    print(f"Extracting {ZIP_PATH} -> {C2_DIR}")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(C2_DIR)

with open(C2_DIR / "metadata.json") as f:
    tiles = json.load(f)["tiles"]

# Load LaMa manifest for C1 valid controls
manifest_file = C2_DIR / "lama_manifest.json"
valid_c1_tiles = set()
if manifest_file.exists():
    with open(manifest_file) as f:
        manifest_data = json.load(f)
        for rec in manifest_data:
            if rec.get("control_status") == "ok":
                valid_c1_tiles.add(rec["tile_id"])

print(f"Loaded {len(tiles)} tiles.")
print(f"Valid C1 control tiles: {len(valid_c1_tiles)}")
print(f"Results destination: {OUT_FILE}")"""),

    ("code", """PROMPTS = {
    "yn_flood": "Is there flooding in this satellite image? Answer only Yes or No.",
    "yn_water": "Is there any water (river, lake, or flood) in this satellite image? Answer only Yes or No.",
}

CONDITIONS = {
    "permanent_only": ["orig"],
    "mixed": [
        "orig",
        "flood_only_removed",
        "flood_only_removed_lama",
        "c1_area_matched_lama",
    ],
}


def image_for_condition(tile, condition, c2_dir):
    c2_dir = Path(c2_dir)
    if condition == "orig":
        return c2_dir / tile["image_path"]
    if condition == "flood_only_removed":
        return c2_dir / tile["flood_only_removed_path"]
    if condition == "flood_only_removed_lama":
        return c2_dir / "lama" / "flood_only_removed_lama" / f"{tile['tile_id']}.png"
    if condition == "c1_area_matched_lama":
        return c2_dir / "lama" / "c1_area_matched_lama" / f"{tile['tile_id']}.png"
    raise ValueError(f"Unknown condition: {condition}")


def build_jobs(tiles, valid_c1_set):
    jobs = []
    for tid in sorted(tiles):
        t = tiles[tid]
        for cond in CONDITIONS[t["group"]]:
            # Only include C1 condition if a valid dry control mask exists
            if cond == "c1_area_matched_lama" and tid not in valid_c1_set:
                continue

            img_path = image_for_condition(t, cond, C2_DIR)
            for pid in PROMPTS:
                jobs.append({
                    "job_id": f"{tid}|{cond}|{pid}",
                    "tile_id": tid,
                    "parent_chip_id": t["parent_chip_id"],
                    "group": t["group"],
                    "condition": cond,
                    "prompt_id": pid,
                    "image": img_path,
                })
    return jobs


def load_done(path):
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    try:
                        done.add(json.loads(line)["job_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return done


jobs = build_jobs(tiles, valid_c1_tiles)
done = load_done(OUT_FILE)
todo = [j for j in jobs if j["job_id"] not in done]
missing = [j for j in jobs if not j["image"].exists()]
print(f"{len(jobs)} total jobs | {len(done)} already done | {len(todo)} to run | {len(missing)} missing images")"""),

    ("code", """import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig

MAX_NEW_TOKENS = 5

torch.cuda.empty_cache()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

if MODEL == "qwen":
    from qwen_vl_utils import process_vision_info
    from transformers import Qwen2_5_VLForConditionalGeneration

    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map="cuda", quantization_config=bnb_config, torch_dtype=torch.float16
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    def infer(image_path, prompt):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                {"type": "text", "text": prompt},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        new_tokens = out[0][inputs.input_ids.shape[1]:]
        return processor.decode(new_tokens, skip_special_tokens=True).strip()

else:
    import transformers
    from transformers import LlavaForConditionalGeneration

    transformers.utils.is_bitsandbytes_multi_backend_available = lambda: False

    MODEL_ID = "llava-hf/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map="cuda", quantization_config=bnb_config, torch_dtype=torch.float16
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.patch_size = 14
    processor.vision_feature_select_strategy = "default"

    def infer(image_path, prompt):
        image = Image.open(image_path).convert("RGB")
        text = f"USER: <image>\\n{prompt}\\nASSISTANT:"
        inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        new_tokens = out[0][inputs.input_ids.shape[1]:]
        return processor.decode(new_tokens, skip_special_tokens=True).strip()

run_info = {
    "model": MODEL,
    "model_id": MODEL_ID,
    "transformers": transformers.__version__,
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "decoding": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS},
    "quantization": "4-bit NF4, double quant, fp16 compute",
    "total_jobs": len(jobs),
}
with open(RESULTS_DIR / f"{MODEL}_run_info.json", "w") as f:
    json.dump(run_info, f, indent=2)
print(json.dumps(run_info, indent=2))"""),

    ("code", """# Smoke test on first job
j = jobs[0]
print(j["job_id"], "->", repr(infer(j["image"], PROMPTS[j["prompt_id"]])))"""),

    ("code", """done = load_done(OUT_FILE)
todo = [j for j in jobs if j["job_id"] not in done]
print(f"Running {len(todo)} of {len(jobs)} jobs")

start = time.time()
with open(OUT_FILE, "a", buffering=1) as out_f:
    for i, j in enumerate(todo, 1):
        try:
            answer = infer(j["image"], PROMPTS[j["prompt_id"]])
        except Exception as e:
            print(f"ERROR {j['job_id']}: {e}")
            continue
        out_f.write(json.dumps({
            "job_id": j["job_id"],
            "tile_id": j["tile_id"],
            "parent_chip_id": j["parent_chip_id"],
            "group": j["group"],
            "condition": j["condition"],
            "prompt_id": j["prompt_id"],
            "answer": answer,
            "model_id": MODEL_ID,
        }) + "\\n")
        if i % 50 == 0 or i == len(todo):
            rate = i / (time.time() - start)
            print(f"[{i}/{len(todo)}] {rate:.2f} jobs/s | last: {j['job_id']} -> {answer!r}")
print("Done.")"""),

    ("code", """# Quick summary breakdown
from collections import defaultdict

counts = defaultdict(lambda: [0, 0, 0])
with open(OUT_FILE) as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        first = r["answer"].strip().lower().lstrip("*\\"' ").split(" ")[0].strip(".,!")
        k = 0 if first == "yes" else 1 if first == "no" else 2
        counts[(r["group"], r["condition"], r["prompt_id"])][k] += 1

print(f"{'group':15s} {'condition':26s} {'prompt':9s} {'n':>4s} {'%Yes':>6s} {'other':>5s}")
for key in sorted(counts):
    yes, no, other = counts[key]
    n = yes + no + other
    print(f"{key[0]:15s} {key[1]:26s} {key[2]:9s} {n:4d} {100 * yes / n:6.1f} {other:5d}")"""),
]


def to_source(text):
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def main():
    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [],
    }
    for kind, src in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": to_source(src)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb["cells"].append(cell)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote {OUT} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
