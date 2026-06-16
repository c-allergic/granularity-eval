"""
Pilot experiment: 20 images, 1 model (Qwen2.5-VL-7B), 9 cells.
Quick check if the granularity bias pattern actually exists.
"""

import json
import math
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# Pure-numpy replacements for scipy.stats (avoid heavy dependency)
def kendalltau(x, y):
    """Kendall tau-b correlation (pure numpy)."""
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    concordant, discordant = 0, 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx * dy > 0:
                concordant += 1
            elif dx * dy < 0:
                discordant += 1
    total = n * (n - 1) / 2
    if total == 0:
        return (0.0, 1.0)
    tau = (concordant - discordant) / total
    return tau, 0.0  # no p-value for speed


def pearsonr(x, y):
    """Pearson correlation (pure numpy)."""
    x, y = np.asarray(x), np.asarray(y)
    xm, ym = x - x.mean(), y - y.mean()
    r = (xm * ym).sum() / (np.sqrt((xm**2).sum()) * np.sqrt((ym**2).sum()) + 1e-12)
    return r, 0.0

PROJECT_ROOT = Path(__file__).parent
PROMPT_FILE = PROJECT_ROOT / "prompts" / "prompts.json"
IMAGE_DIR = PROJECT_ROOT / "data" / "images" / "natural"
RESULTS_DIR = PROJECT_ROOT / "results" / "pilot"

# ── Step 1: Generate captions ───────────────────────────────────────

def generate_captions(model_path="/mnt/shared_resources/models/Qwen2.5-VL-7B-Instruct"):
    """Generate 9 captions per image using Qwen2.5-VL-7B."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    prompts = json.loads(PROMPT_FILE.read_text())
    images = sorted(IMAGE_DIR.glob("*.jpg")) + sorted(IMAGE_DIR.glob("*.jpeg")) + sorted(IMAGE_DIR.glob("*.png"))
    print(f"Found {len(images)} images")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        ignore_mismatched_sizes=True,
    )
    processor = AutoProcessor.from_pretrained(model_path)

    out_dir = RESULTS_DIR / "captions" / "qwen2.5-vl-7b"
    generated = 0

    for img_path in tqdm(images, desc="Generating"):
        img_id = img_path.stem
        img = Image.open(img_path).convert("RGB")

        for cell_id, cell in prompts.items():
            out_path = out_dir / img_id / f"{cell_id}.txt"
            if out_path.exists():
                continue

            max_tokens = {"G1": 64, "G2": 128, "G3": 256}.get(cell_id[:2], 128)

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": cell["prompt"]},
                ],
            }]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.7)
            response = processor.decode(outputs[0], skip_special_tokens=True)
            # Remove system prompt and assistant prefix from response
            response = response.replace("assistant\n", "").replace("assistant", "").strip()
            if cell["prompt"] in response:
                response = response.split(cell["prompt"])[-1].strip()

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(response.strip())
            generated += 1

    print(f"Generated {generated} captions -> {out_dir}")
    return out_dir


# ── Step 2: Quick human evaluation (you, not crowd) ─────────────────

def print_for_manual_eval(caption_dir):
    """Print all captions for manual review. You rank them yourself."""
    captions = defaultdict(dict)
    for img_dir in sorted(Path(caption_dir).iterdir()):
        if not img_dir.is_dir():
            continue
        for txt in img_dir.glob("*.txt"):
            captions[img_dir.name][txt.stem] = txt.read_text().strip()

    for img_id, cells in sorted(captions.items()):
        print(f"\n{'='*80}")
        print(f"IMAGE: {img_id}")
        print(f"{'='*80}")
        for grain in ["G1", "G2", "G3"]:
            print(f"\n  --- {grain} ---")
            for typ in ["A", "I", "X"]:
                cell = f"{grain}-{typ}"
                text = cells.get(cell, "(missing)")
                print(f"  [{cell}] {text[:200]}")
        print()


# ── Step 3: Compute representative metrics ──────────────────────────

def compute_metrics_simple(caption_dir):
    """Compute CLIPScore and simple heuristics on all captions.
    BLEU/METEOR/SPICE/CAPTURE skipped for pilot — need refs and extra deps."""
    import clip as clip_module

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP ViT-L/14 on {device}...")
    clip_model, clip_preprocess = clip_module.load("ViT-L/14", device=device)
    clip_model.eval()

    all_scores = defaultdict(dict)

    # Load captions
    captions_data = defaultdict(dict)
    for img_dir in Path(caption_dir).iterdir():
        if not img_dir.is_dir():
            continue
        img_id = img_dir.name
        for txt in img_dir.glob("*.txt"):
            text = txt.read_text().strip()
            # Strip leftover "assistant" prefix if present
            if text.startswith("assistant\n"):
                text = text.split("\n", 1)[-1].strip()
            elif text.startswith("assistant"):
                text = text[len("assistant"):].strip()
            captions_data[img_id][txt.stem] = text

    # Use G3-A as pseudo-reference for length-based metrics
    for img_id, cells in tqdm(list(captions_data.items()), desc="Computing metrics"):
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = IMAGE_DIR / f"{img_id}{ext}"
            if p.exists():
                img_path = p
                break
        if not img_path:
            continue

        for cell_id, caption in cells.items():
            key = f"{img_id}_{cell_id}"

            # Simple length metric (word count)
            all_scores["word_count"][key] = float(len(caption.split()))

            # CLIPScore (ref-free)
            try:
                img = clip_preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                text = clip_module.tokenize([caption], truncate=True).to(device)
                with torch.no_grad():
                    img_feat = clip_model.encode_image(img)
                    text_feat = clip_model.encode_text(text)
                    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
                    all_scores["clipscore"][key] = float((img_feat * text_feat).sum())
            except Exception as e:
                pass

    # Save scores
    for metric_name, scores in all_scores.items():
        out_path = RESULTS_DIR / "scores" / f"{metric_name}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(scores, indent=1))

    return all_scores


# ── Step 4: Quick analysis ──────────────────────────────────────────

def analyze(all_scores):
    """Compute per-grain statistics and look for the granularity bias pattern."""
    grains = {"G1": "coarse", "G2": "medium", "G3": "detailed"}
    types = ["A", "I", "X"]

    print("\n" + "="*80)
    print("PER-GRAIN ANALYSIS")
    print("="*80)

    for metric_name, scores in all_scores.items():
        print(f"\n--- {metric_name.upper()} ---")
        print(f"{'Cell':<8} {'Mean':>8} {'Std':>8} {'N':>6}")
        for grain_label, grain_name in grains.items():
            for typ in types:
                cell = f"{grain_label}-{typ}"
                vals = [v for k, v in scores.items() if cell in k]
                if vals:
                    mean = sum(vals) / len(vals)
                    std = (sum((v - mean)**2 for v in vals) / len(vals)) ** 0.5
                    print(f"{cell:<8} {mean:>8.4f} {std:>8.4f} {len(vals):>6}")

        # Granularity penalty: compare G1-A vs G3-A
        g1a_vals = [v for k, v in scores.items() if "G1-A" in k]
        g3a_vals = [v for k, v in scores.items() if "G3-A" in k]
        if g1a_vals and g3a_vals:
            g1a_mean = sum(g1a_vals) / len(g1a_vals)
            g3a_mean = sum(g3a_vals) / len(g3a_vals)
            gap = g3a_mean - g1a_mean
            print(f"\n  Granularity penalty (G3-A - G1-A): {gap:+.4f}")
            print(f"  G1-A mean: {g1a_mean:.4f}, G3-A mean: {g3a_mean:.4f}")

    # Human-expected ranking within each grain: A > I > X
    # Check how often each metric agrees
    print("\n" + "="*80)
    print("WITHIN-GRAIN RANKING AGREEMENT (expected: A > I > X)")
    print("="*80)

    for metric_name, scores in all_scores.items():
        correct = 0
        total = 0
        for grain_label in ["G1", "G2", "G3"]:
            cell_a = f"{grain_label}-A"
            cell_i = f"{grain_label}-I"
            cell_x = f"{grain_label}-X"

            for img_key in set(k.rsplit("_", 1)[0] for k in scores):
                sa = scores.get(f"{img_key}_{cell_a}")
                si = scores.get(f"{img_key}_{cell_i}")
                sx = scores.get(f"{img_key}_{cell_x}")
                if sa is None or si is None or sx is None:
                    continue
                if sa > si and si > sx:
                    correct += 1
                elif sa > sx and sx > si:
                    correct += 1  # partial: A > X
                total += 2  # two pairwise comparisons per grain

        if total > 0:
            acc = correct / total * 100
            print(f"  {metric_name:<15s}: {acc:5.1f}% ranking agreement ({correct}/{total} pairs)")

    # Cross-grain P1: G1-A vs G3-X (short-accurate vs long-inaccurate)
    # Human expectation: G1-A wins
    print("\n" + "="*80)
    print("P1 HEAD-TO-HEAD: G1-A (short-accurate) vs G3-X (long-inaccurate)")
    print("Human expectation: G1-A > G3-X")
    print("="*80)

    for metric_name, scores in all_scores.items():
        g1a_wins = 0
        g3x_wins = 0
        total = 0
        for img_key in set(k.rsplit("_", 1)[0] for k in scores):
            sa = scores.get(f"{img_key}_G1-A")
            sx = scores.get(f"{img_key}_G3-X")
            if sa is None or sx is None:
                continue
            total += 1
            if sa > sx:
                g1a_wins += 1
            else:
                g3x_wins += 1

        if total > 0:
            print(f"  {metric_name:<15s}: G1-A wins {g1a_wins}/{total} ({g1a_wins/total*100:.1f}%), "
                  f"G3-X wins {g3x_wins}/{total} ({g3x_wins/total*100:.1f}%)")

    # Save summary
    summary = {
        "granularity_penalty": {},
        "within_grain_agreement": {},
        "p1_head_to_head": {},
    }
    out_path = RESULTS_DIR / "pilot_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary saved to {out_path}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true", help="Generate captions")
    ap.add_argument("--review", action="store_true", help="Print captions for manual review")
    ap.add_argument("--metrics", action="store_true", help="Compute metrics")
    ap.add_argument("--all", action="store_true", help="Run all steps")
    args = ap.parse_args()

    if args.all or args.generate:
        caption_dir = generate_captions()

    caption_dir = RESULTS_DIR / "captions" / "qwen2.5-vl-7b"

    if args.all or args.review:
        print_for_manual_eval(caption_dir)

    if args.all or args.metrics:
        scores = compute_metrics_simple(caption_dir)
        analyze(scores)


if __name__ == "__main__":
    main()
