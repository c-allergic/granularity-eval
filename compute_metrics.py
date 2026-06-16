"""
Compute all 8 evaluation metrics on generated captions.

Metrics:
  Traditional: BLEU-4, METEOR, SPICE
  Embedding:   CLIPScore
  Matching:    CAPTURE, CompreCap
  VQA-based:   CapsBench
  VLM Judge:   LLaVA-Critic-7B

Output: results/scores/{metric}/{image_id}_{model}_{cell}.json
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent
IMAGE_DIR = PROJECT_ROOT / "data" / "images"
CAPTION_DIR = PROJECT_ROOT / "results" / "captions"
REF_DIR = PROJECT_ROOT / "data" / "references"
SCORE_DIR = PROJECT_ROOT / "results" / "scores"


# ── helpers ──────────────────────────────────────────────────────────

def load_captions(model_name):
    """Iterate over all generated captions for a model.
    Yields (image_id, cell_id, caption_text)."""
    model_dir = CAPTION_DIR / model_name
    if not model_dir.exists():
        return
    for img_dir in model_dir.iterdir():
        if not img_dir.is_dir():
            continue
        for txt_file in img_dir.glob("*.txt"):
            cell_id = txt_file.stem
            caption = txt_file.read_text().strip()
            yield img_dir.name, cell_id, caption


def load_references(image_id):
    """Load reference captions for an image."""
    ref_path = REF_DIR / f"{image_id}.json"
    if ref_path.exists():
        refs = json.loads(ref_path.read_text())
        if isinstance(refs, list):
            return refs
        if isinstance(refs, dict):
            return refs.get("references", list(refs.values()))
    return []


def find_image_path(image_id):
    """Map image_id back to file path."""
    for ext in ("jpg", "jpeg", "png", "webp"):
        for p in IMAGE_DIR.rglob(f"*.{ext}"):
            rel = p.relative_to(IMAGE_DIR)
            rid = str(rel.with_suffix("")).replace("/", "_")
            if rid == image_id:
                return str(p)
    return None


def save_scores(metric_name, scores):
    """Save per-caption scores to JSON."""
    out_dir = SCORE_DIR / metric_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, val in scores.items():
        # key: "image_id_model_cell"
        p = out_dir / f"{key}.json"
        p.write_text(json.dumps({"score": val}))


# ── n-gram metrics ───────────────────────────────────────────────────

def compute_ngram_metrics():
    """BLEU-4, METEOR, SPICE using nltk + pycocoevalcap."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    bleu_scores, meteor_scores = {}, {}

    for model_name in ["internvl2-8b", "gpt4o"]:
        for image_id, cell_id, caption in load_captions(model_name):
            refs = load_references(image_id)
            if not refs:
                continue
            key = f"{image_id}_{model_name}_{cell_id}"

            # BLEU-4
            ref_tokens = [r.lower().split() for r in refs]
            cand_tokens = caption.lower().split()
            bleu = sentence_bleu(ref_tokens, cand_tokens,
                                 weights=(0.25, 0.25, 0.25, 0.25),
                                 smoothing_function=SmoothingFunction().method1)
            bleu_scores[key] = float(bleu)

            # METEOR
            meteor = meteor_score([r.split() for r in refs], cand_tokens)
            meteor_scores[key] = float(meteor)

    save_scores("bleu4", bleu_scores)
    save_scores("meteor", meteor_scores)
    print(f"BLEU-4: {len(bleu_scores)} entries, METEOR: {len(meteor_scores)} entries")


def compute_spice():
    """SPICE using pycocoevalcap."""
    # SPICE requires the Java-based SPICE implementation.
    # For simplicity, use the Python wrapper from pycocoevalcap if available.
    try:
        from pycocoevalcap.spice.spice import Spice
    except ImportError:
        print("SPICE not available (pycocoevalcap not installed), skipping")
        return

    spice_scorer = Spice()
    scores = {}

    for model_name in ["internvl2-8b", "gpt4o"]:
        for image_id, cell_id, caption in load_captions(model_name):
            refs = load_references(image_id)
            if not refs:
                continue
            gts = {0: refs}
            res = {0: [caption]}
            try:
                _, s = spice_scorer.compute_score(gts, res)
                key = f"{image_id}_{model_name}_{cell_id}"
                scores[key] = float(s[0]) if s else 0.0
            except Exception:
                continue

    save_scores("spice", scores)
    print(f"SPICE: {len(scores)} entries")


# ── CLIPScore ────────────────────────────────────────────────────────

def compute_clipscore():
    """CLIPScore (ref-free): cosine similarity between image and caption CLIP embeddings."""
    import clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()

    scores = {}

    for model_name in ["internvl2-8b", "gpt4o"]:
        for image_id, cell_id, caption in load_captions(model_name):
            img_path = find_image_path(image_id)
            if not img_path:
                continue
            try:
                img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                text = clip.tokenize([caption], truncate=True).to(device)

                with torch.no_grad():
                    img_feat = model.encode_image(img)
                    text_feat = model.encode_text(text)
                    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
                    sim = (img_feat * text_feat).sum().item()

                key = f"{image_id}_{model_name}_{cell_id}"
                scores[key] = float(sim)
            except Exception as e:
                continue

    save_scores("clipscore", scores)
    print(f"CLIPScore: {len(scores)} entries")


# ── CAPTURE ──────────────────────────────────────────────────────────

def compute_capture():
    """CAPTURE metric using capture-metric package."""
    try:
        from capture_metric.capture import CAPTURE
    except ImportError:
        print("CAPTURE not available, skipping")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = CAPTURE(device=device)

    scores = {}
    for model_name in ["internvl2-8b", "gpt4o"]:
        captions_batch = []
        keys_batch = []
        for image_id, cell_id, caption in load_captions(model_name):
            refs = load_references(image_id)
            if not refs:
                continue
            captions_batch.append({"candidate": caption, "references": refs})
            keys_batch.append(f"{image_id}_{model_name}_{cell_id}")

        results = scorer.compute_score(captions_batch)
        for key, score in zip(keys_batch, results):
            scores[key] = float(score)

    save_scores("capture", scores)
    print(f"CAPTURE: {len(scores)} entries")


# ── CompreCap ────────────────────────────────────────────────────────

def compute_comprescore(comprecap_root=None, dataset_root=None,
                        llama_path=None, bert_path=None):
    """CompreCap: directed scene graph metric (CVPR 2025).

    Only works for images in the CompreCap dataset (560 MSCOCO panoptic images).
    For images NOT in the dataset, skips silently.

    Requires:
      - CompreCap repo: https://github.com/LuFan31/CompreCap
      - CompreCap dataset from HuggingFace (with anno.json)
      - Llama-3-8B-Instruct weights
      - Sentence-BERT model
    """
    from metrics.comprescore_wrapper import CompreCapScorer

    # Default paths on datalab server
    if comprecap_root is None:
        comprecap_root = os.environ.get("COMPRECAP_ROOT", "./CompreCap")
    if dataset_root is None:
        dataset_root = os.environ.get("COMPRECAP_DATASET", "./CompreCap_dataset")
    if llama_path is None:
        llama_path = os.environ.get("LLAMA3_PATH", "/mnt/shared_resources/models/Llama-3-8B-Instruct")
    if bert_path is None:
        bert_path = os.environ.get("SBERT_PATH", "sentence-transformers/all-MiniLM-L6-v2")

    try:
        scorer = CompreCapScorer(
            comprecap_root=comprecap_root,
            dataset_root=dataset_root,
            llama_path=llama_path,
            bert_path=bert_path,
        )
    except FileNotFoundError as e:
        print(f"CompreCap not available: {e}")
        return

    scores = {}
    skipped = 0
    for model_name in ["internvl2-8b", "gpt4o"]:
        for image_id, cell_id, caption in load_captions(model_name):
            # CompreCap uses MSCOCO image names (e.g., "000000000285.jpg")
            # Map our image_id back to the original MSCOCO name if possible
            result = scorer.score(image_id, caption)
            if result is None:
                skipped += 1
                continue
            key = f"{image_id}_{model_name}_{cell_id}"
            scores[key] = result["unified_score"]

    save_scores("comprescore", scores)
    print(f"CompreCap: {len(scores)} entries ({skipped} skipped — not in CompreCap dataset)")


# ── CapsBench ────────────────────────────────────────────────────────

def compute_capsbench(capsbench_root=None, questions_jsonl=None):
    """CapsBench: VQA-based caption evaluation (Playground v3).

    Uses the user's existing CapsBench implementation at /mnt/lixiaofeng/capsbench.
    CapsBench works in two modes:
      1. If questions_jsonl provided: uses pre-generated question bank (fast)
      2. Otherwise: generates reference captions + questions per image (expensive, API calls)

    Args:
        capsbench_root: path to CapsBench repo
        questions_jsonl: path to pre-generated questions JSONL (optional)
    """
    from metrics.capsbench_wrapper import CapsBenchScorer

    if capsbench_root is None:
        capsbench_root = os.environ.get("CAPSBENCH_ROOT", "/mnt/lixiaofeng/capsbench")

    scorer = CapsBenchScorer(
        capsbench_root=capsbench_root,
        questions_jsonl=questions_jsonl,
    )

    scores = {}
    for model_name in ["internvl2-8b", "gpt4o"]:
        for image_id, cell_id, caption in load_captions(model_name):
            img_path = find_image_path(image_id)
            if not img_path:
                continue

            # Generate question bank on first encounter (cached)
            try:
                scorer.ensure_questions(img_path, image_id)
                score = scorer.score(img_path, caption, image_id)
                key = f"{image_id}_{model_name}_{cell_id}"
                scores[key] = float(score)
            except Exception as e:
                print(f"CapsBench error {image_id}/{cell_id}: {e}")
                continue

    save_scores("capsbench", scores)
    print(f"CapsBench: {len(scores)} entries")


# ── LLaVA-Critic ─────────────────────────────────────────────────────

def compute_llava_critic():
    """LLaVA-Critic-7B: open-source VLM evaluation judge (CVPR 2025).
    Ref-free, pointwise scoring."""
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print("transformers not available, skipping LLaVA-Critic")
        return

    model_name = "lmms-lab/llava-critic-7b"
    print(f"Loading {model_name}...")
    model = AutoModel.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    scores = {}
    judge_prompt = (
        "Evaluate the quality of this image caption on a scale of 1 to 5, "
        "considering accuracy, completeness, and clarity. "
        "Respond with only the score."
    )

    for model_name_ in ["internvl2-8b", "gpt4o"]:
        for image_id, cell_id, caption in load_captions(model_name_):
            img_path = find_image_path(image_id)
            if not img_path:
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                full_prompt = f"{judge_prompt}\n\nCaption: {caption}"
                response = model.chat(
                    tokenizer, img, full_prompt,
                    generation_config={"max_new_tokens": 32, "temperature": 0.0}
                )
                # Extract score from response
                score = float(response.strip().split()[0])
                key = f"{image_id}_{model_name_}_{cell_id}"
                scores[key] = score
            except Exception as e:
                continue

    save_scores("llava_critic", scores)
    print(f"LLaVA-Critic: {len(scores)} entries")


# ── main ─────────────────────────────────────────────────────────────

def main():
    metric_map = {
        "ngram": compute_ngram_metrics,
        "spice": compute_spice,
        "clipscore": compute_clipscore,
        "capture": compute_capture,
        "comprescore": compute_comprescore,
        "capsbench": compute_capsbench,
        "llava_critic": compute_llava_critic,
    }

    if len(sys.argv) > 1:
        selected = [m for m in sys.argv[1:] if m in metric_map]
    else:
        selected = list(metric_map.keys())

    for name in selected:
        print(f"\n{'='*50}\nRunning {name}...\n{'='*50}")
        metric_map[name]()


if __name__ == "__main__":
    main()
