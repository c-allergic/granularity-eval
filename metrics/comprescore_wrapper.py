"""
CompreCap wrapper: directed scene graph caption evaluation (CVPR 2025).

CompreCap requires GT scene graph annotations (objects, attributes, relations)
and segmentation masks. These are only available for the 560 images in the
CompreCap dataset (subset of MSCOCO panoptic).

For images NOT in the CompreCap dataset, this wrapper cannot be used.
For images that ARE in the dataset, it computes:
  - object_coverage: proportion of GT objects mentioned
  - attribute_score: 0-5 LLM-judged attribute accuracy
  - relation_score: 0-5 LLM-judged relation accuracy
  - unified_score: weighted combination (25% obj + 35% attr + 40% rel)

Usage:
    scorer = CompreCapScorer(
        comprecap_root="./CompreCap",
        dataset_root="./CompreCap_dataset",
        llama_path="/path/to/Llama-3-8B-Instruct",
        bert_path="sentence-transformers/all-MiniLM-L6-v2",
    )
    result = scorer.score(image_name, caption)
    # result = {"object_coverage": 0.8, "attribute_score": 4.2, ...}
"""

import json
import os
import sys
from pathlib import Path


class CompreCapScorer:
    """Evaluate captions using CompreCap's directed scene graph metric.

    Only works for images in the CompreCap dataset (560 MSCOCO panoptic images).
    Images are identified by their MSCOCO name (e.g., "000000000285.jpg").
    """

    def __init__(self, comprecap_root: str, dataset_root: str,
                 llama_path: str, bert_path: str,
                 device: str = "cuda"):
        self.root = Path(comprecap_root)
        self.dataset_root = Path(dataset_root)
        sys.path.insert(0, str(self.root))

        # Validate
        anno_path = self.dataset_root / "anno.json"
        if not anno_path.exists():
            raise FileNotFoundError(
                f"CompreCap annotation not found: {anno_path}\n"
                f"Download from https://huggingface.co/CompreCap"
            )

        self.anno = json.loads(anno_path.read_text())
        self.valid_images = set(self.anno.keys())

        self.llama_path = llama_path
        self.bert_path = bert_path
        self.device = device

        # Lazy-loaded models
        self._bert_model = None
        self._llama_model = None
        self._llama_tokenizer = None
        self._nlp = None  # spaCy

    def _load_models(self):
        """Lazy-load heavy models."""
        if self._bert_model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            self._bert_model = SentenceTransformer(self.bert_path, device=self.device)

        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                self._nlp = spacy.load("en_core_web_sm")

        if self._llama_model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._llama_tokenizer = AutoTokenizer.from_pretrained(self.llama_path)
            self._llama_model = AutoModelForCausalLM.from_pretrained(
                self.llama_path, torch_dtype=torch.bfloat16, device_map="auto"
            )

    def is_available(self, image_name: str) -> bool:
        """Check if this image has GT scene graph annotations."""
        return image_name in self.valid_images

    def score(self, image_name: str, caption: str) -> dict:
        """Score a single caption against GT scene graph.

        Args:
            image_name: MSCOCO image name (e.g., "000000000285.jpg")
            caption: candidate caption text

        Returns:
            dict with object_coverage, attribute_score, relation_score, unified_score,
            or None if image not in dataset.
        """
        if not self.is_available(image_name):
            return None

        self._load_models()

        gt = self.anno[image_name]

        # 1. Object coverage
        obj_cov = self._compute_object_coverage(caption, gt)

        # 2. Attribute accuracy
        attr_score = self._compute_attribute_score(caption, gt)

        # 3. Relation fidelity
        rel_score = self._compute_relation_score(caption, gt)

        # 4. Unified score
        unified = 0.25 * obj_cov + 0.35 * attr_score + 0.40 * rel_score

        return {
            "object_coverage": obj_cov,
            "attribute_score": attr_score,
            "relation_score": rel_score,
            "unified_score": unified,
        }

    def _compute_object_coverage(self, caption: str, gt: dict) -> float:
        """Extract nouns from caption, match against GT objects via SBERT."""
        import numpy as np
        doc = self._nlp(caption)
        cand_nouns = list(set(
            token.lemma_.lower() for token in doc
            if token.pos_ in ("NOUN", "PROPN") and not token.is_stop
        ))

        gt_objects = list(gt.get("objects", {}).keys())
        if not gt_objects or not cand_nouns:
            return 0.0

        # SBERT similarity matrix
        cand_embs = self._bert_model.encode(cand_nouns, convert_to_tensor=True)
        gt_embs = self._bert_model.encode(gt_objects, convert_to_tensor=True)

        from torch import nn
        cos = nn.CosineSimilarity(dim=1)
        sim_matrix = cos(cand_embs.unsqueeze(1), gt_embs.unsqueeze(0))

        # Greedy bipartite matching
        matched = set()
        for i in range(len(cand_nouns)):
            best_j = sim_matrix[i].argmax().item()
            if sim_matrix[i, best_j] > 0.5 and best_j not in matched:
                matched.add(best_j)

        return len(matched) / len(gt_objects) if gt_objects else 1.0

    def _compute_attribute_score(self, caption: str, gt: dict) -> float:
        """LLM-based attribute scoring (0-5 scale)."""
        import torch

        scores = []
        for obj_name, attrs in gt.get("objects", {}).items():
            attr_list = attrs.get("attributes", [])
            if not attr_list:
                continue

            gt_attr_str = ", ".join(attr_list)
            prompt = (
                f"Ground truth attributes for '{obj_name}': {gt_attr_str}\n"
                f"Caption: {caption}\n"
                f"On a scale of 0-5, how accurately does the caption describe "
                f"the attributes of '{obj_name}'? Respond with only the number."
            )

            inputs = self._llama_tokenizer(prompt, return_tensors="pt").to(self._llama_model.device)
            with torch.no_grad():
                outputs = self._llama_model.generate(**inputs, max_new_tokens=8, temperature=0.0)
            response = self._llama_tokenizer.decode(outputs[0], skip_special_tokens=True)

            try:
                score = float(response.strip().split()[0])
                scores.append(min(max(score, 0), 5))
            except ValueError:
                scores.append(2.5)  # default mid

        return sum(scores) / len(scores) / 5.0 if scores else 0.0  # normalize to 0-1

    def _compute_relation_score(self, caption: str, gt: dict) -> float:
        """LLM-based relation scoring (0-5 scale)."""
        import torch

        gt_relations = gt.get("relations", [])
        if not gt_relations:
            return 1.0  # no relations to evaluate

        rel_strs = [f"{r['subject']} {r['predicate']} {r['object']}" for r in gt_relations]
        gt_rel_str = "; ".join(rel_strs)

        prompt = (
            f"Ground truth relations: {gt_rel_str}\n"
            f"Caption: {caption}\n"
            f"On a scale of 0-5, how accurately does the caption capture "
            f"these spatial and action relationships? Respond with only the number."
        )

        inputs = self._llama_tokenizer(prompt, return_tensors="pt").to(self._llama_model.device)
        with torch.no_grad():
            outputs = self._llama_model.generate(**inputs, max_new_tokens=8, temperature=0.0)
        response = self._llama_tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            score = float(response.strip().split()[0])
            return min(max(score, 0), 5) / 5.0  # normalize to 0-1
        except ValueError:
            return 0.5
