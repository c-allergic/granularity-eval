#!/bin/bash
# Convenience script: run the full pipeline on the datalab server (L40).
# Usage: bash run.sh [step]

set -e
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

case "${1:-all}" in
  generate)
    echo "=== Step 1: Generate captions ==="
    python generate.py --internvl
    ;;

  generate_gpt4o)
    echo "=== Step 1b: Generate GPT-4o captions ==="
    python generate.py --gpt4o
    ;;

  metrics)
    echo "=== Step 2: Compute metrics ==="
    python compute_metrics.py ngram spice clipscore capture llava_critic
    ;;

  capsbench)
    echo "=== Step 2b: CapsBench ==="
    python compute_metrics.py capsbench
    ;;

  hits)
    echo "=== Step 3: Build human evaluation HITs ==="
    python human_eval/build_hits.py
    ;;

  aggregate)
    echo "=== Step 3b: Aggregate human evaluation results ==="
    python human_eval/aggregate.py
    ;;

  analyze)
    echo "=== Step 4: Run analysis ==="
    python analysis.py
    ;;

  all)
    echo "=== Running full pipeline ==="
    python generate.py --internvl
    python generate.py --gpt4o
    python compute_metrics.py ngram spice clipscore capture llava_critic
    python human_eval/build_hits.py
    echo "=== Pipeline complete (human eval + analysis pending) ==="
    ;;

  *)
    echo "Usage: bash run.sh [generate|generate_gpt4o|metrics|capsbench|hits|aggregate|analyze|all]"
    ;;
esac
