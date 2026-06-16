"""
CapsBench wrapper: VQA-based caption evaluation.

Usage:
    scorer = CapsBenchScorer(capsbench_root="/mnt/lixiaofeng/capsbench")
    score = scorer.score(image_path, caption)  # returns float 0-1
"""

import json
import os
import sys
import tempfile
from pathlib import Path


class CapsBenchScorer:
    """Minimal wrapper around the user's CapsBench implementation.

    Uses CapsBench's Stage 4 judge directly with a question bank.
    If no question bank exists for an image, falls back to generating one
    via Stages 1-2 (expensive, one-time cost per image).
    """

    def __init__(self, capsbench_root: str = "/mnt/lixiaofeng/capsbench",
                 questions_jsonl: str = None):
        self.root = Path(capsbench_root)
        sys.path.insert(0, str(self.root))

        from src.config import BenchmarkConfig
        from src.pipeline.stage4_judge import run_stage4
        from src.models.factory import create_model

        self.config = BenchmarkConfig()
        self.run_stage4 = run_stage4
        self.create_model = create_model
        self._judge_model = None

        # Pre-load question bank if provided
        self._questions = {}
        if questions_jsonl and os.path.exists(questions_jsonl):
            from src.data_schemas.schemas import QuestionRecord
            from src.utils.io_utils import load_jsonl
            records, _ = load_jsonl(questions_jsonl, QuestionRecord.from_dict)
            for q in records:
                self._questions.setdefault(q.image_id, []).append(q)

    @property
    def judge_model(self):
        if self._judge_model is None:
            self._judge_model = self.create_model(vars(self.config.judge_model))
        return self._judge_model

    def ensure_questions(self, image_path: str, image_id: str = None):
        """Generate question bank for an image via CapsBench Stages 1-2.
        Runs once per image; results are cached."""
        if image_id is None:
            image_id = Path(image_path).stem

        if image_id in self._questions:
            return  # already cached

        # Run stages 1+2 for this single image via CapsBench main
        from main import run_stage1, run_stage2
        import shutil

        tmp_dir = tempfile.mkdtemp(prefix="capsbench_")
        tmp_img = os.path.join(tmp_dir, os.path.basename(image_path))
        shutil.copy(image_path, tmp_img)

        cfg = self.config
        cfg.data_dir = tmp_dir
        cfg.output_dir = tmp_dir

        run_stage1(cfg)
        run_stage2(cfg)

        # Load generated questions
        from src.data_schemas.schemas import QuestionRecord
        from src.utils.io_utils import load_jsonl
        records, _ = load_jsonl(
            os.path.join(tmp_dir, "questions_caption.jsonl"),
            QuestionRecord.from_dict,
        )
        for q in records:
            q.image_id = image_id  # normalize
            self._questions.setdefault(image_id, []).append(q)

        # Cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def score(self, image_path: str, caption: str, image_id: str = None) -> float:
        """Score a single caption. Returns accuracy (0-1).

        Requires the CapsBench question bank for this image.
        Call ensure_questions() first, or provide questions_jsonl."""
        if image_id is None:
            image_id = Path(image_path).stem

        questions = self._questions.get(image_id, [])
        if not questions:
            raise ValueError(
                f"No question bank for {image_id}. "
                f"Call ensure_questions() first or provide questions_jsonl."
            )

        from src.pipeline.stage4_judge import _single_judge_call, majority_vote

        correct = 0
        total = 0
        for q in questions:
            if q.question.startswith("[ERROR]"):
                continue
            votes = []
            for _ in range(self.config.consensus_rounds):
                verdict, _ = _single_judge_call(
                    self.judge_model, q.question, caption, self.config.judge_model.temperature
                )
                votes.append(verdict)
            final = majority_vote(votes)
            if final == "Yes":
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.0

    def score_batch(self, items: list) -> dict:
        """Batch score multiple (image_path, caption) pairs.

        Args:
            items: list of (image_path, caption, image_id) tuples

        Returns:
            dict mapping image_id -> score
        """
        results = {}
        for image_path, caption, image_id in items:
            try:
                results[image_id] = self.score(image_path, caption, image_id)
            except Exception as e:
                results[image_id] = None
        return results
