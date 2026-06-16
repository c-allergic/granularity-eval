"""
Build human evaluation HITs for Prolific/MTurk.

Phase A: Within-grain ranking — for each (image, grain), show 3 captions (A,I,X) and ask annotator to rank.
Phase B: Cross-grain pairwise — for each (image, pair_type), show 2 captions and ask which is better.
"""

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
CAPTION_DIR = PROJECT_ROOT / "results" / "captions"
OUTPUT_DIR = PROJECT_ROOT / "results" / "human"
METADATA_FILE = PROJECT_ROOT / "data" / "metadata.csv"

PAIRS = {
    "P1": ("G1-A", "G3-X"),   # short-accurate vs long-inaccurate
    "P2": ("G1-A", "G3-A"),   # short-accurate vs long-accurate
    "P3": ("G1-X", "G3-X"),   # short-inaccurate vs long-inaccurate
    "P4": ("G2-A", "G3-X"),   # medium-accurate vs long-inaccurate
}


def load_caption(image_id, model, cell_id):
    """Load caption text for a given image/model/cell combination."""
    path = CAPTION_DIR / model / image_id / f"{cell_id}.txt"
    if path.exists():
        return path.read_text().strip()
    return None


def get_all_image_ids():
    """Get all image IDs from the captions directory."""
    # Use internvl2-8b as reference for available images
    model_dir = CAPTION_DIR / "internvl2-8b"
    if model_dir.exists():
        return sorted(d.name for d in model_dir.iterdir() if d.is_dir())
    if CAPTION_DIR.exists():
        # Try gpt4o
        model_dir = CAPTION_DIR / "gpt4o"
        if model_dir.exists():
            return sorted(d.name for d in model_dir.iterdir() if d.is_dir())
    return []


def build_phase_a(image_ids, model="internvl2-8b"):
    """Build within-grain ranking HITs."""
    rows = []
    grains = {"G1": ["G1-A", "G1-I", "G1-X"],
              "G2": ["G2-A", "G2-I", "G2-X"],
              "G3": ["G3-A", "G3-I", "G3-X"]}

    for img_id in image_ids:
        for grain_label, cells in grains.items():
            captions = {}
            for cell in cells:
                cap = load_caption(img_id, model, cell)
                if cap:
                    captions[cell] = cap

            if len(captions) < 3:
                continue

            rows.append({
                "image_id": img_id,
                "phase": "A",
                "granularity": grain_label,
                "caption_A": captions[cells[0]],
                "caption_I": captions[cells[1]],
                "caption_X": captions[cells[2]],
                "cell_A": cells[0],
                "cell_I": cells[1],
                "cell_X": cells[2],
            })

    return rows


def build_phase_b(image_ids, model="internvl2-8b"):
    """Build cross-grain pairwise HITs."""
    rows = []

    for img_id in image_ids:
        for pair_id, (cell_a, cell_b) in PAIRS.items():
            cap_a = load_caption(img_id, model, cell_a)
            cap_b = load_caption(img_id, model, cell_b)
            if not cap_a or not cap_b:
                continue

            rows.append({
                "image_id": img_id,
                "phase": "B",
                "pair_id": pair_id,
                "caption_A": cap_a,
                "caption_B": cap_b,
                "cell_A": cell_a,
                "cell_B": cell_b,
            })

    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_ids = get_all_image_ids()
    if not image_ids:
        print("No captions found. Run generate.py first.")
        sys.exit(1)

    print(f"Found {len(image_ids)} images with captions")

    # Build Phase A
    phase_a = build_phase_a(image_ids)
    print(f"Phase A (ranking): {len(phase_a)} HITs")

    # Build Phase B
    phase_b = build_phase_b(image_ids)
    print(f"Phase B (pairwise): {len(phase_b)} HITs")

    # Save as CSV for Prolific/MTurk
    with open(OUTPUT_DIR / "phase_a_ranking.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=phase_a[0].keys())
        writer.writeheader()
        writer.writerows(phase_a)

    with open(OUTPUT_DIR / "phase_b_pairwise.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=phase_b[0].keys())
        writer.writeheader()
        writer.writerows(phase_b)

    # Also save metadata about the structure
    meta = {
        "total_images": len(image_ids),
        "phase_a_hits": len(phase_a),
        "phase_b_hits": len(phase_b),
        "grains": ["G1 (coarse)", "G2 (medium)", "G3 (detailed)"],
        "cells": ["A (Accurate)", "I (Incomplete)", "X (Inaccurate)"],
        "pairs": PAIRS,
        "models_available": [d.name for d in CAPTION_DIR.iterdir() if d.is_dir()],
    }
    with open(OUTPUT_DIR / "hit_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nFiles written to {OUTPUT_DIR}/")
    print("  phase_a_ranking.csv")
    print("  phase_b_pairwise.csv")
    print("  hit_metadata.json")


if __name__ == "__main__":
    main()
