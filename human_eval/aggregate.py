"""
Aggregate raw human annotation results into standardized format.

Input:  results/human/phase_a_raw.csv, results/human/phase_b_raw.csv
Output: results/human/aggregated.json
"""

import csv
import json
from pathlib import Path
from collections import defaultdict, Counter

PROJECT_ROOT = Path(__file__).parent.parent
HUMAN_DIR = PROJECT_ROOT / "results" / "human"


def aggregate_phase_a(csv_path):
    """Aggregate within-grain ranking results.

    Expected CSV columns: image_id, granularity, annotator_id, rank_A, rank_I, rank_X
    """
    if not csv_path.exists():
        print(f"Phase A raw data not found: {csv_path}")
        return {}

    # Collect per-image per-grain per-annotator rankings
    raw = defaultdict(lambda: defaultdict(list))

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row["image_id"]
            grain = row["granularity"]
            # If ranks are 1-3 (1=best), convert to ordering
            ranking = sorted(
                [("A", int(row.get("rank_A", 0))),
                 ("I", int(row.get("rank_I", 0))),
                 ("X", int(row.get("rank_X", 0)))],
                key=lambda x: x[1]
            )
            # Normalize: lower rank = better
            ordered = [c for c, _ in ranking]
            raw[img_id][grain].append(ordered)

    # Aggregate: majority vote for ranking
    results = {}
    for img_id, grains in raw.items():
        results[img_id] = {}
        for grain, rankings in grains.items():
            # Use Borda count or simple majority
            # For simplicity: count first-place votes per cell
            first_votes = Counter(r[0] for r in rankings)
            second_votes = Counter(r[1] for r in rankings)
            third_votes = Counter(r[2] for r in rankings)

            # Determine consensus ranking
            # The cell with most first-place votes wins first
            cells = ["G1-A", "G1-I", "G1-X"] if grain.startswith("G1") else \
                    ["G2-A", "G2-I", "G2-X"] if grain.startswith("G2") else \
                    ["G3-A", "G3-I", "G3-X"]

            # Map from (A,I,X) to cell names
            cell_map = {"A": cells[0], "I": cells[1], "X": cells[2]}

            consensus = sorted(cells, key=lambda c: (
                -first_votes.get(c.split("-")[1], 0),
                second_votes.get(c.split("-")[1], 0),
            ))

            results[img_id][grain] = {
                "ranking": consensus,
                "num_annotators": len(rankings),
                "first_votes": {cell_map[k]: v for k, v in first_votes.items()},
            }

    return results


def aggregate_phase_b(csv_path):
    """Aggregate cross-grain pairwise results.

    Expected CSV columns: image_id, pair_id, annotator_id, preferred (A or B)
    """
    if not csv_path.exists():
        print(f"Phase B raw data not found: {csv_path}")
        return {}

    raw = defaultdict(lambda: defaultdict(list))

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row["image_id"]
            pair_id = row["pair_id"]
            pref = row["preferred"]  # "A" or "B"
            raw[img_id][pair_id].append(pref)

    results = {}
    for img_id, pairs in raw.items():
        results[img_id] = {}
        for pair_id, prefs in pairs.items():
            counts = Counter(prefs)
            majority = counts.most_common(1)[0][0]
            results[img_id][pair_id] = {
                "preferred": majority,
                "A_votes": counts.get("A", 0),
                "B_votes": counts.get("B", 0),
                "num_annotators": len(prefs),
            }

    return results


def main():
    # Load build metadata for cell/pair definitions
    meta_path = HUMAN_DIR / "hit_metadata.json"
    if not meta_path.exists():
        print("hit_metadata.json not found, run build_hits.py first")
        return

    # Aggregate
    phase_a = aggregate_phase_a(HUMAN_DIR / "phase_a_raw.csv")
    phase_b = aggregate_phase_b(HUMAN_DIR / "phase_b_raw.csv")

    output = {
        "within_grain": phase_a,
        "cross_grain": phase_b,
    }

    out_path = HUMAN_DIR / "aggregated.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Aggregated results saved to {out_path}")
    print(f"  Phase A: {len(phase_a)} images with within-grain rankings")
    print(f"  Phase B: {len(phase_b)} images with cross-grain pair preferences")


if __name__ == "__main__":
    main()
