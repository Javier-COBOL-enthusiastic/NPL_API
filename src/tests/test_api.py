import secrets
import threading
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from src.service.app import create_app
from src.tests.conftest import wait_loaded


def test_health_public(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("path", ["/info", "/predict"])
@pytest.mark.parametrize("kind", ["missing", "wrong", "lower", "upper", "case_token",
                                 "double_space", "trailing", "basic", "duplicate"])
def test_auth(client, token, body, path, kind):
    values = {
        "missing": None, "wrong": "Bearer " + secrets.token_hex(32),
        "lower": "bearer " + token, "upper": "BEARER " + token,
        "case_token": "Bearer " + token.upper(), "double_space": "Bearer  " + token,
        "trailing": "Bearer " + token + " ", "basic": "Basic " + token,
    }
    if kind == "duplicate":
        headers = [("Authorization", "Bearer " + token)] * 2
    else:
        headers = {} if values[kind] is None else {"Authorization": values[kind]}
    response = (client.get(path, headers=headers) if path == "/info"
                else client.post(path, headers=headers, json=body))
    assert response.status_code == 401
    assert response.json() == {"error": "unauthorized"}
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("value", [None, ""])
def test_no_configured_token(client, headers, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("QA_API_TOKEN")
    else:
        monkeypatch.setenv("QA_API_TOKEN", value)
    assert client.get("/info", headers=headers).status_code == 401
    assert client.get("/health").status_code == 503


def test_info(client, headers):
    response = client.get("/info", headers=headers)
    assert response.status_code == 200
    info = response.json()
    assert info["team"] == "test-fixture"
    assert info["model"] == "fixture/base"
    assert type(info["params"]) is int and info["params"] > 0


@pytest.mark.parametrize("field,limit", [("id", 64), ("context", 5000), ("question", 500)])
@pytest.mark.parametrize("invalid", ["missing", "empty", "long", 1, 1.5, True, None, [], {}])
def test_invalid_fields(client, headers, body, field, limit, invalid):
    if invalid == "missing":
        body.pop(field)
    elif invalid == "empty":
        body[field] = ""
    elif invalid == "long":
        body[field] = "x" * (limit + 1)
    else:
        body[field] = invalid
    response = client.post("/predict", headers=headers, json=body)
    assert response.status_code in (400, 422)
    assert response.json()["error"] == "bad_request"
    assert isinstance(response.json()["detail"], str)


@pytest.mark.parametrize("content", ['{', 'null', '[]', '"text"', '', '{"id":NaN}'])
def test_invalid_json(client, headers, content):
    response = client.post("/predict", headers={**headers, "Content-Type": "application/json"},
                           content=content)
    assert response.status_code in (400, 422)
    assert response.json()["error"] == "bad_request"


def test_extra_fields_and_no_input_leak(client, headers, body):
    private = secrets.token_hex(20)
    body[private] = private
    response = client.post("/predict", headers=headers, json=body)
    assert response.status_code == 422
    assert private not in response.text


def test_auth_precedes_json_parsing(client):
    response = client.post("/predict", content="{", headers={"Content-Type": "application/json"})
    assert response.status_code == 401


@pytest.mark.parametrize("content_type", [None, "text/plain"])
def test_json_content_type_required(client, headers, content_type):
    if content_type:
        headers["Content-Type"] = content_type
    response = client.post("/predict", headers=headers,
                           content='{"id":"x","context":"river","question":"what"}')
    assert response.status_code in (400, 422)
    assert response.json()["error"] == "bad_request"


@pytest.mark.parametrize("context", ["A river flows.", "No matching span.", "🙂 Café: river, river!"])
def test_prediction(client, headers, body, context):
    body["context"] = context
    response = client.post("/predict", headers=headers, json=body)
    assert response.status_code == 200
    out = response.json()
    assert set(out) == {"id", "answer", "start", "score"}
    assert out["id"] == body["id"]
    assert 0 <= out["score"] <= 1
    assert (out["answer"] is None) == (out["start"] is None)
    if out["answer"] is not None:
        assert context[out["start"]:out["start"] + len(out["answer"])] == out["answer"]


def test_maximum_lengths(client, headers):
    response = client.post("/predict", headers=headers,
                           json={"id": "x" * 64, "context": "x" * 5000, "question": "x" * 500})
    assert response.status_code == 200


def test_internal_error_is_sanitized(client, fake_engine, headers, body):
    private = secrets.token_hex(20)
    def fail(body):
        raise RuntimeError("C:/private/" + private)
    fake_engine.predict = fail
    response = client.post("/predict", headers=headers, json=body)
    assert response.status_code == 500
    assert response.json() == {"error": "internal_error"}


@pytest.mark.parametrize("patch", [{"start": -1}, {"answer": "invented"}, {"id": "changed"},
                                  {"start": None}, {"answer": None}, {"score": 1.01},
                                  {"score": float("nan")}, {"start": True}])
def test_invalid_engine_output_is_never_served(client, fake_engine, headers, body, patch):
    fake_engine.predict = lambda _: {"id": body["id"], "answer": "river", "start": 2,
                                     "score": 0.8, **patch}
    response = client.post("/predict", headers=headers, json=body)
    assert response.status_code == 500
    assert response.json() == {"error": "internal_error"}


def test_background_loading_and_single_load(token, headers, fake_engine):
    release = threading.Event()
    calls = []
    def load():
        calls.append(1)
        assert release.wait(timeout=10)
        return fake_engine
    with TestClient(create_app(load)) as client:
        try:
            response = client.get("/health")
            assert response.status_code == 503
            assert response.json() == {"status": "loading"}
            assert client.get("/info", headers=headers).json() == {"error": "internal_error"}
        finally:
            release.set()
        wait_loaded(client)
        assert client.get("/health").status_code == 200
        for _ in range(3):
            assert client.get("/info", headers=headers).status_code == 200
        assert len(calls) == 1


def test_failed_load(token, headers, body):
    def fail():
        raise FileNotFoundError("private path")
    with TestClient(create_app(fail)) as client:
        wait_loaded(client)
        assert client.get("/health").json() == {"status": "loading"}
        assert client.get("/health").status_code == 503
        response = client.post("/predict", headers=headers, json=body)
        assert response.status_code == 500
        assert response.json() == {"error": "internal_error"}


def test_responses_against_professor_yaml(client, headers, body):
    contract = yaml.safe_load((Path(__file__).resolve().parents[2] /
                               "docs/qa_api_contract.yaml").read_text(encoding="utf-8"))
    schemas = contract["components"]["schemas"]
    cases = [(client.get("/health"), "Health"),
             (client.get("/info", headers=headers), "Info"),
             (client.post("/predict", headers=headers, json=body), "PredictResponse"),
             (client.get("/info"), "Error"),
             (client.post("/predict", headers=headers, json={}), "Error")]
    for response, schema in cases:
        assert response.headers["content-type"] == "application/json"
        Draft202012Validator(schemas[schema]).validate(response.json())
    Draft202012Validator(schemas["PredictRequest"]).validate(body)
    generated = client.app.openapi()["components"]["schemas"]["PredictRequest"]
    for field in ("id", "context", "question"):
        for constraint in ("type", "minLength", "maxLength"):
            assert generated["properties"][field][constraint] == schemas["PredictRequest"]["properties"][field][constraint]
    assert generated["additionalProperties"] is False
    for path in ("/info", "/predict"):
        method = "get" if path == "/info" else "post"
        assert client.app.openapi()["paths"][path][method]["security"] == [{"bearerAuth": []}]
    assert not client.app.openapi()["paths"]["/health"]["get"].get("security")
