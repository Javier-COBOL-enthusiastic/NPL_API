"""Exact Bearer authentication, exclusively from the process environment."""

import hmac
import os

from starlette.requests import Request


def authorized(request: Request) -> bool:
    expected = os.environ.get("QA_API_TOKEN")
    values = request.headers.getlist("authorization")
    if not expected or len(values) != 1:
        return False
    # Comparing the entire header also rejects alternate case and extra spaces.
    return hmac.compare_digest(
        values[0].encode("utf-8"), ("Bearer " + expected).encode("utf-8")
    )
