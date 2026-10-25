"""Run from the repository root: python -m uvicorn src.service.app:app."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Security
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from starlette.exceptions import HTTPException

from .auth import authorized
from .schemas import Error, Health, Info, PredictRequest, PredictResponse

log = logging.getLogger("qa.service")
bearer = HTTPBearer(scheme_name="bearerAuth", auto_error=False)


def load_engine():
    # Even importing torch must not block the event loop or the health endpoint.
    from .inference import QAEngine

    return QAEngine.load()


def create_app(loader=None) -> FastAPI:
    loader = loader or load_engine

    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = None
        app.state.load_finished = asyncio.Event()

        async def initialize():
            try:
                engine = await asyncio.to_thread(loader)
                app.state.engine = engine  # Publish only a fully validated engine.
            except Exception as exc:
                # Do not log exception text: loaders can include paths or credentials.
                log.error("QA initialization failed (%s); check local model/configuration.",
                          type(exc).__name__)
            finally:
                app.state.load_finished.set()

        task = asyncio.create_task(initialize())
        try:
            yield
        finally:
            await task
            app.state.engine = None

    app = FastAPI(title="Extractive QA as a Service", lifespan=lifespan)
    app.state.engine = None

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if request.url.path.rstrip("/") in {"/info", "/predict"}:
            if not authorized(request):
                return JSONResponse({"error": "unauthorized"}, status_code=401,
                                    headers={"WWW-Authenticate": "Bearer"})
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Never echo input values, extra keys, exception messages or request bodies.
        return JSONResponse({"error": "bad_request", "detail": "Invalid request body."},
                            status_code=422)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        code = "internal_error" if exc.status_code >= 500 else "bad_request"
        content = {"error": code}
        if code == "bad_request":
            content["detail"] = "Invalid request."
        return JSONResponse(content, status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def internal_error(request, exc):
        log.error("QA request failed (%s).", type(exc).__name__)
        return JSONResponse({"error": "internal_error"}, status_code=500)

    def engine():
        if app.state.engine is None:
            # The contract reserves 503/loading for /health. No invented metadata.
            raise HTTPException(500)
        return app.state.engine

    @app.get("/health", response_model=Health, responses={503: {"model": Health}})
    async def health():
        ready = app.state.engine is not None and bool(os.environ.get("QA_API_TOKEN"))
        return JSONResponse({"status": "ok" if ready else "loading"},
                            status_code=200 if ready else 503)

    # Security dependencies document Bearer in OpenAPI. Exact enforcement is above,
    # before JSON parsing; HTTPBearer alone would accept lowercase scheme names.
    @app.get("/info", response_model=Info, dependencies=[Security(bearer)],
             responses={401: {"model": Error}, 500: {"model": Error}})
    def info():
        return engine().info

    @app.post("/predict", response_model=PredictResponse,
              dependencies=[Security(bearer)],
              responses={400: {"model": Error}, 401: {"model": Error},
                         422: {"model": Error}, 500: {"model": Error}})
    def predict(body: PredictRequest, request: Request):
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
            raise HTTPException(400)
        result = PredictResponse.model_validate(engine().predict(body))
        if result.id != body.id or (result.answer is not None and
                body.context[result.start:result.start + len(result.answer)] != result.answer):
            raise RuntimeError("Invalid extractive output")
        return result

    return app


app = create_app()
