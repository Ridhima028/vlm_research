"""Writes notebooks/run_c2.ipynb (C2 river-vs-flood Yes/No inference on Colab).

Model loading follows vlm_benchmark_colab.ipynb (4-bit NF4 Qwen2.5-VL / LLaVA-1.5).
Edit the cell sources here and re-run this script instead of hand-editing the JSON.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "run_c2.ipynb"

CELLS = [
    ("markdown", """# C2: River vs. flood (Yes/No)

Runs one model on the C2 set built by `build_c2.py`:

| group | conditions | prompts |
| --- | --- | --- |
| permanent_only | orig | yn_flood, yn_water |
| mixed | orig, flood_only_removed | yn_flood, yn_water |

**Before running:** upload `data/c2.zip` to `MyDrive/vlm_flood_research/data/c2.zip` and select a GPU runtime (T4 is enough).

**Run order:** pick `MODEL` → run all cells top to bottom. Results are appended to
`MyDrive/vlm_flood_research/results/c2/<model>.jsonl`; re-running skips finished jobs, so a disconnect only costs the current job.

**Switching model:** `Runtime → Restart session`, change `MODEL`, run from the top (the two models need different `transformers` versions)."""),

    ("code", """MODEL = "qwen"  # @param ["qwen", "llava"]
MODEL = MODEL.strip().lower()
assert MODEL in ("qwen", "llava"), MODEL"""),

    ("code", """# Qwen2.5-VL needs a recent transformers; the pilot LLaVA run used 4.46.3 (see vlm_benchmark_colab.ipynb)
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
os.environ["HF_HOME"] = "/content/hf_cache"  # Colab local disk: weights re-download each session, Drive stays free
ZIP_PATH = f"{DRIVE_ROOT}/data/c2.zip"
C2_DIR = Path("/content/c2")
RESULTS_DIR = Path(f"{DRIVE_ROOT}/results/c2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = RESULTS_DIR / f"{MODEL}.jsonl"

if not (C2_DIR / "metadata.json").exists():
    print(f"Extracting {ZIP_PATH} -> {C2_DIR}")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(C2_DIR)

with open(C2_DIR / "metadata.json") as f:
    tiles = json.load(f)["tiles"]
groups = {}
for t in tiles.values():
    groups[t["group"]] = groups.get(t["group"], 0) + 1
print(f"Loaded {len(tiles)} tiles: {groups}")
print(f"Results file: {OUT_FILE}")"""),

    ("code", """PROMPTS = {
    "yn_flood": "Is there flooding in this satellite image? Answer only Yes or No.",
    "yn_water": "Is there any water (river, lake, or flood) in this satellite image? Answer only Yes or No.",
}
CONDITIONS = {"permanent_only": ["orig"], "mixed": ["orig", "flood_only_removed"]}
IMAGE_KEY = {"orig": "image_path", "flood_only_removed": "flood_only_removed_path"}


def build_jobs(tiles):
    jobs = []
    for tid in sorted(tiles):
        t = tiles[tid]
        for cond in CONDITIONS[t["group"]]:
            for pid in PROMPTS:
                jobs.append({
                    "job_id": f"{tid}|{cond}|{pid}",
                    "tile_id": tid,
                    "group": t["group"],
                    "condition": cond,
                    "prompt_id": pid,
                    "image": C2_DIR / t[IMAGE_KEY[cond]],
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
                        pass  # partial last line after a disconnect
    return done


jobs = build_jobs(tiles)
done = load_done(OUT_FILE)
todo = [j for j in jobs if j["job_id"] not in done]
missing = [j for j in jobs if not j["image"].exists()]
print(f"{len(jobs)} jobs | {len(done)} already done | {len(todo)} to run | {len(missing)} missing images")"""),

    ("code", """import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig

MAX_NEW_TOKENS = 5

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
        new_tokens = out[0][inputs.input_ids.shape[1]:]  # decode only the generated part
        return processor.decode(new_tokens, skip_special_tokens=True).strip()

else:
    import transformers
    from transformers import LlavaForConditionalGeneration

    # Workaround carried over from vlm_benchmark_colab.ipynb (bitsandbytes multi-backend check bug)
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
        new_tokens = out[0][inputs.input_ids.shape[1]:]  # decode only the generated part
        return processor.decode(new_tokens, skip_special_tokens=True).strip()

import transformers
run_info = {
    "model": MODEL,
    "model_id": MODEL_ID,
    "transformers": transformers.__version__,
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "decoding": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS},
    "quantization": "4-bit NF4, double quant, fp16 compute",
}
with open(RESULTS_DIR / f"{MODEL}_run_info.json", "w") as f:
    json.dump(run_info, f, indent=2)
print(json.dumps(run_info, indent=2))"""),

    ("code", """# Smoke test on one job before the full loop
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
        except Exception as e:  # skip, it will be retried on the next run
            print(f"ERROR {j['job_id']}: {e}")
            continue
        out_f.write(json.dumps({
            "job_id": j["job_id"],
            "tile_id": j["tile_id"],
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

    ("code", """# Sanity check: % answering Yes per group / condition / prompt (not the final analysis)
from collections import defaultdict

counts = defaultdict(lambda: [0, 0, 0])  # yes, no, other
with open(OUT_FILE) as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        first = r["answer"].strip().lower().lstrip("*\\"' ").split(" ")[0].strip(".,!")
        k = 0 if first == "yes" else 1 if first == "no" else 2
        counts[(r["group"], r["condition"], r["prompt_id"])][k] += 1

print(f"{'group':15s} {'condition':19s} {'prompt':9s} {'n':>4s} {'%Yes':>6s} {'other':>5s}")
for key in sorted(counts):
    yes, no, other = counts[key]
    n = yes + no + other
    print(f"{key[0]:15s} {key[1]:19s} {key[2]:9s} {n:4d} {100 * yes / n:6.1f} {other:5d}")"""),
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
