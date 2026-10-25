"""Offline fixtures only: these tests do not measure the trained model's quality."""

import json
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient
from pydantic import ValidationError
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import DistilBertConfig, DistilBertForQuestionAnswering, PreTrainedTokenizerFast

from src.service import inference
from src.service.app import create_app
from src.service.inference import PostprocessConfig, QAEngine
from src.service.schemas import PredictRequest
from src.tests.conftest import wait_loaded
from src.training.evaluate import apply_threshold, decision_score, run_inference


@pytest.fixture
def tokenizer():
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "word", "target",
             "a", "river", "flows", ".", "what", "?"]
    backend = Tokenizer(models.WordPiece({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
        model_max_length=512, model_input_names=["input_ids", "attention_mask"],
    )


class LogitFixture(torch.nn.Module):
    """Predicts only the token 'target'; no pretrained model is downloaded."""
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(max_position_embeddings=512)
        self.calls = []

    def forward(self, input_ids, attention_mask):
        assert not self.training
        assert torch.is_inference_mode_enabled()
        assert input_ids.device.type == "cpu"
        self.calls.append(tuple(input_ids.shape))
        logits = torch.full(input_ids.shape, -10.0)
        logits[:, 0] = 0.0
        logits[input_ids == 6] = 10.0
        return SimpleNamespace(start_logits=logits, end_logits=logits.clone())


@pytest.fixture
def config():
    # Synthetic calibration for tests ONLY; no production threshold default.
    return {"threshold": 0.0, "max_seq_length": 64, "doc_stride": 16,
            "n_best": 20, "max_answer_len": 10}


@pytest.fixture
def engine(tokenizer, config):
    return QAEngine(LogitFixture(), tokenizer, PostprocessConfig(**config),
                    "test-fixture", "fixture/base", batch_size=3)


@pytest.mark.parametrize("context", ["word target word", "word " * 650 + "target",
                                    "🙂 café target, target!", "word word"])
def test_real_tokenizer_evaluation_and_offsets(engine, context, monkeypatch):
    observed = {}
    def record(*args, **kwargs):
        result = run_inference(*args, **kwargs)
        observed.update(kwargs=kwargs, result=result[0])
        return result
    monkeypatch.setattr(inference, "run_inference", record)
    body = PredictRequest(id="id preserved", context=context, question="what ?")
    out = engine.predict(body)
    assert out.id == body.id
    assert (out.answer, out.start, out.score) == apply_threshold(observed["result"], 0.0)
    assert out.score == decision_score(observed["result"]["delta"], 0.0)
    assert 0 <= out.score <= 1
    if "target" in context:
        assert out.answer == "target"
        assert context[out.start:out.start + len(out.answer)] == out.answer
    else:
        assert out.answer is None and out.start is None
    if len(context) > 3000:
        assert len(engine.tokenizer(context)["input_ids"]) > 512
        assert observed["result"]["n_windows"] > 1
        assert out.start > 3000  # Answer near the end, beyond the first 512 tokens.
    for key, value in {"max_len": 64, "stride": 16, "batch_size": 3,
                       "n_best": 20, "max_answer_len": 10}.items():
        assert observed["kwargs"][key] == value
    assert all(batch <= 3 and length <= 64 for batch, length in engine.model.calls)
    assert engine.info.params == sum(p.numel() for p in engine.model.parameters())


def test_short_inputs_trim_padding(engine):
    engine.predict(PredictRequest(id="x", context="word target", question="what?"))
    assert engine.model.calls[0][1] < engine.config.max_seq_length


def test_long_question_preserves_all_context(engine):
    out = engine.predict(PredictRequest(id="x", context="word " * 650 + "target", question="?" * 500))
    assert out.answer == "target" and out.start == 3250
    assert len(engine.tokenizer(engine._question("?" * 500), add_special_tokens=False)["input_ids"]) <= engine.question_budget


@pytest.mark.parametrize("delta,answer", [(1.5, "target"), (1.5001, None), (-2, "target")])
def test_threshold_boundary_and_score(delta, answer):
    pred = {"delta": delta, "answer": "target", "start": 5}
    result = apply_threshold(pred, 1.5)
    assert result[0] == answer
    assert result[1] == (5 if answer else None)
    assert result[2] == decision_score(delta, 1.5)


@pytest.mark.parametrize("field,value", [("threshold", float("nan")), ("threshold", float("inf")),
                                        ("max_seq_length", 3), ("doc_stride", 64),
                                        ("n_best", 0), ("max_answer_len", -1),
                                        ("max_seq_length", "64"), ("doc_stride", True)])
def test_invalid_calibration(config, field, value):
    config[field] = value
    with pytest.raises(ValidationError):
        PostprocessConfig(**config)


@pytest.mark.parametrize("field", ["threshold", "max_seq_length", "doc_stride", "n_best", "max_answer_len"])
def test_no_invented_calibration_defaults(config, field):
    config.pop(field)
    with pytest.raises(ValidationError):
        PostprocessConfig(**config)


@pytest.fixture
def local_export(tmp_path, tokenizer, config, monkeypatch):
    directory = tmp_path / "final_model"
    directory.mkdir()
    (directory / "postprocess_config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "run_config.json").write_text(json.dumps({"base_checkpoint": "fixture/base"}), encoding="utf-8")
    tokenizer.save_pretrained(directory)
    monkeypatch.setenv("QA_MODEL_DIR", str(directory))
    monkeypatch.setenv("QA_TEAM", "test-fixture")
    monkeypatch.setenv("QA_TORCH_THREADS", "1")
    monkeypatch.delenv("QA_BASE_MODEL", raising=False)
    monkeypatch.delenv("QA_RUN_CONFIG", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    return directory


def patch_weights(monkeypatch, tokenizer, loading=None):
    calls = []
    def load_tokenizer(*args, **kwargs):
        calls.append(kwargs)
        return tokenizer
    def load_model(*args, **kwargs):
        calls.append(kwargs)
        return LogitFixture(), loading or {}
    monkeypatch.setattr(inference.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(inference.AutoModelForQuestionAnswering, "from_pretrained", load_model)
    return calls


def test_loader_local_only_and_warmup(local_export, tokenizer, monkeypatch):
    calls = patch_weights(monkeypatch, tokenizer)
    engine = QAEngine.load()
    assert engine.model.calls  # Warmup actually executed.
    assert engine.info.model == "fixture/base" and engine.info.params == 1
    assert len(calls) == 2
    assert all(c["local_files_only"] and not c["trust_remote_code"] for c in calls)
    assert calls[1]["output_loading_info"] is True


@pytest.mark.parametrize("failure", ["no_directory", "no_weights", "no_threshold", "no_team",
                                    "no_metadata", "conflicting_base", "bad_threads", "bad_batch",
                                    "missing_metadata_path"])
def test_unready_artifacts(local_export, monkeypatch, token, failure):
    if failure == "no_directory":
        monkeypatch.setenv("QA_MODEL_DIR", str(local_export / "missing"))
    elif failure == "no_threshold":
        (local_export / "postprocess_config.json").unlink()
    elif failure == "no_team":
        monkeypatch.delenv("QA_TEAM")
    elif failure == "no_metadata":
        (local_export / "run_config.json").unlink()
    elif failure == "conflicting_base":
        monkeypatch.setenv("QA_BASE_MODEL", "fixture/other")
    elif failure == "bad_threads":
        monkeypatch.setenv("QA_TORCH_THREADS", "0")
    elif failure == "bad_batch":
        monkeypatch.setenv("QA_BATCH_SIZE", "0")
    elif failure == "missing_metadata_path":
        monkeypatch.setenv("QA_RUN_CONFIG", str(local_export / "missing.json"))
    # No weights are present in this fixture, including the no_weights case.
    with TestClient(create_app(QAEngine.load)) as client:
        wait_loaded(client)
        assert client.get("/health").status_code == 503
        assert client.get("/health").json() == {"status": "loading"}


@pytest.mark.parametrize("key", ["missing_keys", "mismatched_keys", "error_msgs"])
def test_incomplete_weights_rejected(local_export, tokenizer, monkeypatch, key):
    patch_weights(monkeypatch, tokenizer, {key: ["qa_outputs.weight"]})
    with pytest.raises(ValueError, match="Incomplete"):
        QAEngine.load()


def test_training_metadata_layout(local_export, tokenizer, monkeypatch):
    patch_weights(monkeypatch, tokenizer)
    target = local_export.parent / "config"
    target.mkdir()
    (local_export / "run_config.json").rename(target / "run_config.json")
    assert QAEngine.load().info.model == "fixture/base"


def test_explicit_base_metadata(local_export, tokenizer, monkeypatch):
    patch_weights(monkeypatch, tokenizer)
    (local_export / "run_config.json").unlink()
    monkeypatch.setenv("QA_BASE_MODEL", "fixture/confirmed-base")
    assert QAEngine.load().info.model == "fixture/confirmed-base"


def test_actual_save_pretrained_roundtrip(local_export, tokenizer, token, headers):
    # Tiny random architecture ONLY validates serialization/CPU integration offline.
    # It is not a QA checkpoint, quality evaluation or replacement for Javier's model.
    model = DistilBertForQuestionAnswering(DistilBertConfig(
        vocab_size=len(tokenizer), max_position_embeddings=128,
        n_layers=1, n_heads=2, dim=16, hidden_dim=32,
    ))
    model.save_pretrained(local_export)
    with TestClient(create_app(QAEngine.load)) as client:
        wait_loaded(client)
        assert client.get("/health").status_code == 200
        assert client.get("/info", headers=headers).json()["params"] == sum(p.numel() for p in model.parameters())
        body = {"id": "offline-fixture", "context": "word " * 650 + "target", "question": "what?"}
        response = client.post("/predict", json=body, headers=headers)
        assert response.status_code == 200
        result = response.json()
        assert result["id"] == body["id"] and 0 <= result["score"] <= 1
        if result["answer"] is not None:
            assert body["context"][result["start"]:result["start"] + len(result["answer"])] == result["answer"]
