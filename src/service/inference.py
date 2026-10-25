"""CPU adapter for Javier's unchanged evaluation and exported artifacts."""

import json
import math
import os
import threading
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from ..training.evaluate import apply_threshold, run_inference
from .schemas import Info, PredictRequest, PredictResponse


class PostprocessConfig(BaseModel):
    # evaluate.py also exports calibration metrics, rule and score descriptions.
    model_config = ConfigDict(strict=True, extra="ignore")

    threshold: float = Field(allow_inf_nan=False)
    max_seq_length: int = Field(gt=3)
    doc_stride: int = Field(ge=0)
    n_best: int = Field(gt=0)
    max_answer_len: int = Field(gt=0)

    @model_validator(mode="after")
    def valid_windows(self):
        if self.doc_stride >= self.max_seq_length - 3:
            raise ValueError("Invalid overlap")
        if max(self.n_best, self.max_answer_len) > self.max_seq_length:
            raise ValueError("Invalid span configuration")
        return self


def positive_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError("Expected a positive integer configuration")
    return value


class QAEngine:
    def __init__(self, model, tokenizer, config, team, base_model, batch_size=8):
        self.model = model.to(torch.device("cpu")).eval()
        self.tokenizer = tokenizer
        self.config = config
        self.batch_size = batch_size
        self.lock = threading.Lock()
        self.info = Info(
            team=team, model=base_model,
            params=sum(p.numel() for p in self.model.parameters()),
            max_seq_length=config.max_seq_length, doc_stride=config.doc_stride,
        )
        if not tokenizer.is_fast or tokenizer.padding_side != "right":
            raise ValueError("Evaluation requires fast tokenization and right padding")
        probe = tokenizer("question", "context")["input_ids"]
        if tokenizer.cls_token_id is None or probe[0] != tokenizer.cls_token_id:
            raise ValueError("Evaluation requires CLS at position zero")
        limits = [tokenizer.model_max_length]
        positions = getattr(model.config, "max_position_embeddings", None)
        if positions is not None:
            limits.append(positions)
        if config.max_seq_length > min(limits):
            raise ValueError("Window exceeds model/tokenizer capacity")
        self.question_budget = (config.max_seq_length - config.doc_stride
                                - tokenizer.num_special_tokens_to_add(pair=True) - 1)
        if self.question_budget < 1:
            raise ValueError("No room for question and context")

    @classmethod
    def load(cls):
        directory = Path(os.environ.get("QA_MODEL_DIR", "models/final_model"))
        if not directory.is_dir():
            raise ValueError("Local model directory is missing")
        config = PostprocessConfig.model_validate_json(
            (directory / "postprocess_config.json").read_text(encoding="utf-8")
        )
        team = os.environ.get("QA_TEAM", "")
        run_path = os.environ.get("QA_RUN_CONFIG")
        if run_path:
            metadata_path = Path(run_path)
        elif (directory / "run_config.json").is_file():
            metadata_path = directory / "run_config.json"
        else:
            metadata_path = directory.parent / "config" / "run_config.json"
        metadata = (json.loads(metadata_path.read_text(encoding="utf-8"))
                    if metadata_path.is_file() else {})
        if run_path and not metadata_path.is_file():
            raise ValueError("Explicit run configuration is missing")
        recorded_base = metadata.get("base_checkpoint")
        configured_base = os.environ.get("QA_BASE_MODEL")
        if recorded_base and configured_base and recorded_base != configured_base:
            raise ValueError("Conflicting base checkpoint metadata")
        base_model = recorded_base or configured_base
        if not isinstance(base_model, str) or not base_model or any(c.isspace() for c in base_model):
            raise ValueError("Base checkpoint metadata is required")
        if not team or any(c.isspace() for c in team):
            raise ValueError("Registered team name is required")

        torch.set_num_threads(positive_env("QA_TORCH_THREADS", min(4, os.cpu_count() or 1)))
        batch_size = positive_env("QA_BATCH_SIZE", 8)
        tokenizer = AutoTokenizer.from_pretrained(
            directory, use_fast=True, local_files_only=True, trust_remote_code=False,
        )
        model, loading = AutoModelForQuestionAnswering.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False,
            output_loading_info=True,
        )
        # Transformers can silently initialize a missing QA head. Refuse that model.
        if any(loading.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("Incomplete trained weights")
        engine = cls(model, tokenizer, config, team, base_model, batch_size)
        # Warm up CPU execution and check offsets/finite scores before publishing readiness.
        engine.predict(PredictRequest(id="warmup", context="A river flows.", question="What flows?"))
        return engine

    def _question(self, text):
        question = text.lstrip()  # Same normalization as evaluate.py.
        encoded = self.tokenizer(question, add_special_tokens=False, return_offsets_mapping=True)
        if len(encoded["input_ids"]) > self.question_budget:
            # A valid 500-character question can overflow a token window. Preserve the
            # calibrated stride and ALL context; shorten only the question in this case.
            question = question[:encoded["offset_mapping"][self.question_budget - 1][1]]
        return question

    def predict(self, body: PredictRequest) -> PredictResponse:
        # Serialize CPU work to avoid oversubscribing PyTorch threads; health remains async.
        with self.lock, torch.inference_mode():
            result = run_inference(
                self.model, self.tokenizer, [self._question(body.question)], [body.context],
                max_len=self.config.max_seq_length, stride=self.config.doc_stride,
                batch_size=self.batch_size, device=torch.device("cpu"),
                n_best=self.config.n_best, max_answer_len=self.config.max_answer_len,
            )[0]
        if not all(math.isfinite(result[key]) for key in ("span_score", "null_score", "delta")):
            raise RuntimeError("Non-finite model scores")
        answer, start, score = apply_threshold(result, self.config.threshold)
        output = PredictResponse(id=body.id, answer=answer, start=start, score=score)
        if answer is not None and body.context[start:start + len(answer)] != answer:
            raise RuntimeError("Invalid extractive output")
        return output
