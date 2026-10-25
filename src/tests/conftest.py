import asyncio
import secrets
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.service.app import create_app


@pytest.fixture
def token(monkeypatch):
    value = secrets.token_hex(32)
    monkeypatch.setenv("QA_API_TOKEN", value)
    return value


@pytest.fixture
def headers(token):
    return {"Authorization": "Bearer " + token}


@pytest.fixture
def fake_engine():
    def predict(body):
        start = body.context.find("river")
        return {"id": body.id, "answer": "river" if start >= 0 else None,
                "start": start if start >= 0 else None, "score": 0.75}

    return SimpleNamespace(
        info={"team": "test-fixture", "model": "fixture/base", "params": 1,
              "max_seq_length": 384, "doc_stride": 128}, predict=predict,
    )


def wait_loaded(client):
    async def wait():
        await asyncio.wait_for(client.app.state.load_finished.wait(), timeout=15)
    client.portal.call(wait)


@pytest.fixture
def client(token, fake_engine):
    with TestClient(create_app(lambda: fake_engine), raise_server_exceptions=False) as client:
        wait_loaded(client)
        yield client


@pytest.fixture
def body():
    return {"id": " q-01 ", "context": "A river flows.", "question": "What flows?"}
