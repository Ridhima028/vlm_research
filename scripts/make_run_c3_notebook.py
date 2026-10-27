"""Writes notebooks/run_c3.ipynb (C3 prompt variants on dry tiles on Colab).

Evaluates whether prompt framing ("direct" vs "evidence" vs "conservative") reduces
false alarms on verified dry satellite tiles.
Outputs written to results/c3/<model>_c3.jsonl.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "run_c3.ipynb"

CELLS = [
    ("markdown", r"""# C3: Prompt Variants on Dry Tiles (False-Alarm Mitigation)

Tests whether leading prompts or explicit evidence constraints reduce false positive flood detections on 150 verified dry tiles (water = 0%).

| Prompt ID | Text |
| --- | --- |
| `leading_damage` | "Describe the flood damage in this image." |
| `direct` | "Is there flooding in this satellite image? Answer only Yes or No." |
| `evidence` | "Using only visible evidence in this satellite image, is flooding present? Answer only Yes or No." |
| `conservative` | "Do not infer flooding from rivers, vegetation, or terrain alone. Is there visible evidence of flooding? Answer only Yes or No." |

**Before running:**
1. Upload `data/c3/c3.zip` to `MyDrive/vlm_flood_research/data/c3.zip`.
2. Select a GPU runtime (T4 is sufficient).
3. Pick `MODEL` ("qwen" or "llava") → Run all cells.
4. Outputs are saved to `MyDrive/vlm_flood_research/results/c3/<model>_c3.jsonl`.
"""),

    ("code", """MODEL = "qwen"  # @param ["qwen", "llava"]
MODEL = MODEL.strip().lower()
assert MODEL in ("qwen", "llava"), MODEL"""),

    ("code", """if MODEL == "qwen":
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
ZIP_PATH = f"{DRIVE_ROOT}/data/c3.zip"
C3_DIR = Path("/content/c3")
RESULTS_DIR = Path(f"{DRIVE_ROOT}/results/c3")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = RESULTS_DIR / f"{MODEL}_c3.jsonl"

if not (C3_DIR / "metadata.json").exists():
    print(f"Extracting {ZIP_PATH} -> {C3_DIR}")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(C3_DIR)

with open(C3_DIR / "metadata.json") as f:
    meta_obj = json.load(f)
    tiles = meta_obj["tiles"]
    prompts = meta_obj["meta"]["prompts"]

print(f"Loaded {len(tiles)} dry tiles.")
print(f"Prompts to test: {list(prompts.keys())}")
print(f"Results destination: {OUT_FILE}")"""),

    ("code", """def build_jobs(tiles, prompts):
    jobs = []
    for tid in sorted(tiles):
        t = tiles[tid]
        img_path = C3_DIR / t["image_path"]
        for pid, prompt_text in prompts.items():
            jobs.append({
                "job_id": f"{tid}|{pid}",
                "tile_id": tid,
                "parent_chip_id": t["parent_chip_id"],
                "prompt_id": pid,
                "prompt": prompt_text,
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


jobs = build_jobs(tiles, prompts)
done = load_done(OUT_FILE)
todo = [j for j in jobs if j["job_id"] not in done]
print(f"{len(jobs)} total jobs | {len(done)} already done | {len(todo)} to run")"""),

    ("code", """import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig

MAX_NEW_TOKENS = 64

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

if MODEL == "qwen":
    from qwen_vl_utils import process_vision_info
    from transformers import Qwen2_5_VLForConditionalGeneration

    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map="auto", quantization_config=bnb_config
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
        MODEL_ID, device_map="auto", quantization_config=bnb_config
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
    "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "decoding": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS},
    "total_jobs": len(jobs),
}
with open(RESULTS_DIR / f"{MODEL}_run_info.json", "w") as f:
    json.dump(run_info, f, indent=2)
print(json.dumps(run_info, indent=2))"""),

    ("code", """done = load_done(OUT_FILE)
todo = [j for j in jobs if j["job_id"] not in done]
print(f"Running {len(todo)} of {len(jobs)} jobs")

start = time.time()
with open(OUT_FILE, "a", buffering=1) as out_f:
    for i, j in enumerate(todo, 1):
        try:
            answer = infer(j["image"], j["prompt"])
        except Exception as e:
            print(f"ERROR {j['job_id']}: {e}")
            continue
        out_f.write(json.dumps({
            "job_id": j["job_id"],
            "tile_id": j["tile_id"],
            "parent_chip_id": j["parent_chip_id"],
            "prompt_id": j["prompt_id"],
            "answer": answer,
            "model_id": MODEL_ID,
        }) + "\\n")
        if i % 50 == 0 or i == len(todo):
            rate = i / (time.time() - start)
            print(f"[{i}/{len(todo)}] {rate:.2f} jobs/s | last: {j['job_id']} -> {answer!r}")
print("Done.")"""),

    ("code", """# Breakdown: False-Alarm Rate (% Yes on Dry Tiles) by Prompt
from collections import defaultdict

counts = defaultdict(lambda: [0, 0, 0])
with open(OUT_FILE) as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        first = r["answer"].strip().lower().lstrip("*\\"' ").split(" ")[0].strip(".,!")
        k = 0 if first == "yes" else 1 if first == "no" else 2
        counts[r["prompt_id"]][k] += 1

print(f"{'prompt_id':15s} {'n':>4s} {'%Yes (False Alarm)':>20s} {'other':>6s}")
for pid in sorted(counts):
    yes, no, other = counts[pid]
    n = yes + no + other
    print(f"{pid:15s} {n:4d} {100 * yes / n:19.1f}% {other:6d}")"""),
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
