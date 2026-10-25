"""Strict HTTP schemas matching docs/qa_api_contract.yaml."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: Annotated[str, Field(min_length=1, max_length=64)]
    context: Annotated[str, Field(min_length=1, max_length=5000)]
    question: Annotated[str, Field(min_length=1, max_length=500)]


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    answer: str | None
    start: Annotated[int, Field(ge=0)] | None
    score: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]

    @model_validator(mode="after")
    def paired_nulls(self):
        if (self.answer is None) != (self.start is None):
            raise ValueError("answer and start must both be null or both present")
        return self


class Info(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    team: Annotated[str, Field(min_length=1, pattern=r"^\S+$")]
    model: Annotated[str, Field(min_length=1)]
    params: Annotated[int, Field(ge=1)]
    max_seq_length: int
    doc_stride: int


class Health(BaseModel):
    status: Literal["ok", "loading"]


class Error(BaseModel):
    error: Literal["unauthorized", "bad_request", "internal_error"]
    detail: str | None = None
