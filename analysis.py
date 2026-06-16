"""
Meta-evaluation analysis:
  - Per-grain Kendall τ (Figure 1)
  - Cross-grain pairwise accuracy (Figure 2)
  - Granularity penalty (Figure 3)
  - Williams test for statistical significance

Input:  results/scores/ + results/human/aggregated.json
Output: analysis figures + tables
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import kendalltau, pearsonr
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent
SCORE_DIR = PROJECT_ROOT / "results" / "scores"
HUMAN_DIR = PROJECT_ROOT / "results" / "human"

METRIC_NAMES = ["bleu4", "meteor", "spice", "clipscore",
                "capture", "comprescore", "capsbench", "llava_critic"]

GRAINS = ["coarse", "medium", "detailed"]
GRAIN_PREFIX = {"coarse": "G1", "medium": "G2", "detailed": "G3"}


# ── data loading ─────────────────────────────────────────────────────

def load_human_data():
    """Load aggregated human judgments.

    Expected format (results/human/aggregated.json):
    {
      "within_grain": {
        "image_001": {
          "coarse": {"ranking": ["G1-A", "G1-I", "G1-X"], "scores": {"G1-A": 5, "G1-I": 2, "G1-X": 1}},
          "medium": {...},
          "detailed": {...}
        }, ...
      },
      "cross_grain": {
        "P1": {"image_001": {"preferred": "A", "A": "G1-A_internvl2-8b", "B": "G3-X_internvl2-8b"}}, ...
      }
    }
    """
    path = HUMAN_DIR / "aggregated.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def load_metric_scores(metric_name):
    """Load all scores for a metric. Returns dict: key -> score."""
    metric_dir = SCORE_DIR / metric_name
    if not metric_dir.exists():
        return {}
    scores = {}
    for f in metric_dir.glob("*.json"):
        data = json.loads(f.read_text())
        # key format: image_id_model_cell
        scores[f.stem] = data["score"]
    return scores


def parse_key(key):
    """Parse 'image_id_model_cell' into components."""
    # Key format: {image_id}_{model}_{cell}
    # model is one of: internvl2-8b, gpt4o
    # cell is one of: G1-A, G1-I, G1-X, G2-A, ...
    parts = key.rsplit("_", 2)
    if len(parts) < 3:
        # Try splitting differently: image_id can contain underscores
        for model in ["internvl2-8b", "gpt4o"]:
            if f"_{model}_" in key:
                idx = key.index(f"_{model}_")
                image_id = key[:idx]
                rest = key[idx + len(model) + 1:]
                cell = rest
                grain = rest[:2]  # G1, G2, G3
                grain_map = {"G1": "coarse", "G2": "medium", "G3": "detailed"}
                return image_id, model, cell, grain_map.get(grain, "unknown")
        return key, "unknown", "unknown", "unknown"
    return parts[0], parts[1], parts[2], None


# ── Figure 1: Per-grain Kendall τ ────────────────────────────────────

def compute_per_grain_kendall(metric_scores, human_data):
    """For each grain, compute Kendall τ between metric ranking and human ranking."""
    results = {m: {} for m in metric_scores}
    results["human"] = {}

    if not human_data or "within_grain" not in human_data:
        return results

    within = human_data["within_grain"]

    for metric_name, scores in metric_scores.items():
        for grain in GRAINS:
            grain_prefix = GRAIN_PREFIX[grain]
            tau_values = []

            for image_id, grain_data in within.items():
                if grain not in grain_data:
                    continue
                human_ranking = grain_data[grain].get("ranking", [])
                if not human_ranking:
                    continue

                # Get metric scores for the 3 cells in this grain
                cell_scores = {}
                for cell_id in human_ranking:
                    for model in ["internvl2-8b", "gpt4o"]:
                        key = f"{image_id}_{model}_{cell_id}"
                        if key in scores:
                            cell_scores[cell_id] = scores[key]
                            break  # take first model's score

                if len(cell_scores) < 2:
                    continue

                # Compute Kendall τ between metric ranking and human ranking
                # Convert to rank ordering
                cells = list(cell_scores.keys())
                metric_order = sorted(cells, key=lambda c: cell_scores[c], reverse=True)
                human_order = human_ranking

                # Map to numbers for tau
                human_rank_map = {c: i for i, c in enumerate(human_order)}
                metric_rank_map = {c: i for i, c in enumerate(metric_order)}

                all_cells = list(set(human_order) & set(metric_order))
                if len(all_cells) < 2:
                    continue
                x = [human_rank_map[c] for c in all_cells]
                y = [metric_rank_map[c] for c in all_cells]

                tau, _ = kendalltau(x, y)
                if not np.isnan(tau):
                    tau_values.append(tau)

            results[metric_name][grain] = np.mean(tau_values) if tau_values else float("nan")

    return results


# ── Figure 2: Cross-grain pairwise accuracy ──────────────────────────

def compute_pairwise_accuracy(metric_scores, human_data):
    """Compute how often each metric agrees with humans on P1-P4 pairs."""
    if not human_data or "cross_grain" not in human_data:
        return {}

    cross = human_data["cross_grain"]
    results = {}

    for pair_id, pairs in cross.items():
        for metric_name, scores in metric_scores.items():
            correct = 0
            total = 0
            for image_id, pair in pairs.items():
                preferred = pair["preferred"]
                a_key = pair["A"]  # e.g., "G1-A_internvl2-8b"
                b_key = pair["B"]

                # Resolve the actual score keys
                for model in ["internvl2-8b", "gpt4o"]:
                    a_score_key = f"{image_id}_{model}_{a_key.split('_', 1)[1] if '_' in a_key else a_key}"
                    b_score_key = f"{image_id}_{model}_{b_key.split('_', 1)[1] if '_' in b_key else b_key}"
                    if a_score_key in scores and b_score_key in scores:
                        metric_pref = "A" if scores[a_score_key] > scores[b_score_key] else "B"
                        if metric_pref == preferred:
                            correct += 1
                        total += 1
                        break

            acc = correct / total if total > 0 else float("nan")
            results.setdefault(pair_id, {})[metric_name] = acc

    return results


# ── Figure 3: Granularity penalty ────────────────────────────────────

def compute_granularity_penalty(metric_scores):
    """Compare scores for accurate captions across granularities (G1-A, G2-A, G3-A)."""
    results = {m: {} for m in metric_scores}

    for metric_name, scores in metric_scores.items():
        for grain, prefix in GRAIN_PREFIX.items():
            cell_id = f"{prefix}-A"
            vals = []
            for key, score in scores.items():
                if cell_id in key:
                    vals.append(score)
            results[metric_name][grain] = np.mean(vals) if vals else float("nan")

    return results


# ── Williams test ────────────────────────────────────────────────────

def williams_test(r12, r13, r23, n):
    """Williams test for dependent correlations.
    H0: r13 <= r23, H1: r13 > r23 (your metric better than baseline).

    r12: corr(your_metric, baseline_metric)
    r13: corr(your_metric, human)
    r23: corr(baseline_metric, human)
    n: number of samples
    """
    K = 1 - r12**2 - r13**2 - r23**2 + 2 * r12 * r13 * r23
    numerator = (r13 - r23) * np.sqrt((n - 1) * (1 + r12))
    denominator = np.sqrt(
        2 * K * (n - 1) / (n - 3) + (r13 + r23)**2 / 4 * (1 - r12)**3
    )
    t_stat = numerator / denominator
    from scipy.stats import t as t_dist
    p_value = 1 - t_dist.cdf(t_stat, n - 3)
    return t_stat, p_value


def compute_williams_all(metric_scores, human_data):
    """Run Williams test: each metric vs each baseline."""
    if not human_data:
        return []

    # Collect paired metric-human scores
    results = []
    return results  # TODO: implement once human data shape is finalized


# ── plotting ─────────────────────────────────────────────────────────

def plot_figure1(kendall_results, output_path):
    """Per-grain Kendall τ line chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    grains = GRAINS
    x = range(len(grains))

    colors = {
        "human": "black", "bleu4": "#9ca3af", "meteor": "#6b7280",
        "spice": "#4b5563", "clipscore": "#2563eb",
        "capture": "#dc2626", "comprescore": "#ea580c",
        "capsbench": "#059669", "llava_critic": "#7c3aed",
    }
    styles = {"human": "--", "capture": "-"}

    for metric, grain_scores in kendall_results.items():
        if all(np.isnan(grain_scores.get(g, float("nan"))) for g in grains):
            continue
        y = [grain_scores.get(g, float("nan")) for g in grains]
        color = colors.get(metric, "#888888")
        ls = styles.get(metric, "-" if "llava" in metric or "capsbench" in metric else "-.")
        lw = 2.5 if metric in ("human", "capture", "llava_critic") else 1.5
        ax.plot(x, y, "o-", color=color, linestyle=ls, linewidth=lw,
                markersize=7, label=metric.upper(), alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(["Coarse (G1)", "Medium (G2)", "Detailed (G3)"])
    ax.set_ylabel("Kendall τ")
    ax.set_ylim(0.0, 0.80)
    ax.legend(loc="lower left", ncol=2, fontsize=8)
    ax.set_title("Within-Grain Kendall τ: Metric vs Human Ranking")
    ax.axhline(y=0.70, color="black", linestyle=":", alpha=0.4, label="Human ceiling")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Figure 1 saved to {output_path}")


def plot_figure2(pairwise_results, output_path):
    """Cross-grain pairwise accuracy bar chart (P1 focus)."""
    if "P1" not in pairwise_results:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    p1 = pairwise_results["P1"]

    metrics = list(p1.keys())
    accs = [p1[m] * 100 for m in metrics]
    colors = ["#dc2626" if m == "capture" else "#7c3aed" if "llava" in m
              else "#059669" if "capsbench" in m else "#2563eb" if "clipscore" in m
              else "#9ca3af" for m in metrics]

    bars = ax.bar(metrics, accs, color=colors, alpha=0.85)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Random chance")
    ax.axhline(y=85, color="black", linestyle=":", alpha=0.5, label="Human (85%)")
    ax.set_ylabel("Agreement with human (%)")
    ax.set_title("P1: Short-Accurate vs Long-Inaccurate — Who Agrees with Humans?")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Figure 2 saved to {output_path}")


def plot_figure3(penalty_results, output_path):
    """Granularity penalty: G1-A vs G3-A score comparison."""
    fig, ax = plt.subplots(figsize=(10, 5))

    metrics = list(penalty_results.keys())
    x = np.arange(len(metrics))
    width = 0.35

    g1_scores = [penalty_results[m].get("coarse", float("nan")) for m in metrics]
    g3_scores = [penalty_results[m].get("detailed", float("nan")) for m in metrics]

    # Normalize to 0-1 per metric
    all_scores = []
    for m in metrics:
        for g in GRAINS:
            v = penalty_results[m].get(g, float("nan"))
            if not np.isnan(v):
                all_scores.append(v)
    vmin, vmax = min(all_scores), max(all_scores)

    def norm(v):
        return (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5

    g1_norm = [norm(s) for s in g1_scores]
    g3_norm = [norm(s) for s in g3_scores]
    gaps = [g3_norm[i] - g1_norm[i] for i in range(len(metrics))]

    bars1 = ax.bar(x - width/2, g1_norm, width, label="G1-A (Coarse)", color="#dbeafe", edgecolor="#1e40af")
    bars2 = ax.bar(x + width/2, g3_norm, width, label="G3-A (Detailed)", color="#fee2e2", edgecolor="#991b1b")

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=9)
    ax.set_ylabel("Normalized Score")
    ax.set_title("Granularity Penalty: G1-A (coarse-accurate) vs G3-A (detailed-accurate)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Figure 3 saved to {output_path}")


# ── main ─────────────────────────────────────────────────────────────

def main():
    import matplotlib
    matplotlib.use("Agg")

    human_data = load_human_data()

    # Load all metric scores
    all_scores = {}
    for m in METRIC_NAMES:
        scores = load_metric_scores(m)
        if scores:
            all_scores[m] = scores

    print(f"Loaded {len(all_scores)} metrics: {list(all_scores.keys())}")

    # Figure 1
    kendall = compute_per_grain_kendall(all_scores, human_data)
    if kendall:
        print("\n=== Per-Grain Kendall τ ===")
        for metric, grains in kendall.items():
            print(f"  {metric}: {', '.join(f'{g}={v:.3f}' for g, v in grains.items())}")
        plot_figure1(kendall, PROJECT_ROOT / "results" / "figure1_kendall.png")

    # Figure 2
    pairwise = compute_pairwise_accuracy(all_scores, human_data)
    if pairwise:
        print("\n=== Cross-Grain Pairwise Accuracy ===")
        for pair_id, metrics in pairwise.items():
            print(f"  {pair_id}:")
            for m, acc in metrics.items():
                print(f"    {m}: {acc:.3f}")
        plot_figure2(pairwise, PROJECT_ROOT / "results" / "figure2_pairwise.png")

    # Figure 3
    penalty = compute_granularity_penalty(all_scores)
    if penalty:
        print("\n=== Granularity Penalty (normalized gap G1 vs G3) ===")
        for m, grains in penalty.items():
            g1 = grains.get("coarse", float("nan"))
            g3 = grains.get("detailed", float("nan"))
            print(f"  {m}: G1={g1:.3f}, G3={g3:.3f}, gap={g3-g1:.3f}")
        plot_figure3(penalty, PROJECT_ROOT / "results" / "figure3_penalty.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
