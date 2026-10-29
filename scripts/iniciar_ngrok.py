"""Interactive Windows launcher; reads .env without printing credentials."""

import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
LOCAL_URL = "http://127.0.0.1:8000"


def domain_url(value):
    domain = value.strip().removeprefix("https://").rstrip("/")
    # Accept a hostname only, never userinfo, paths, ports or shell arguments.
    labels = domain.split(".")
    if len(domain) > 253 or len(labels) < 2 or any(
        not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label)
        for label in labels
    ):
        raise ValueError("NGROK_DOMAIN debe ser tu dominio fijo, sin rutas ni puertos.")
    return "https://" + domain.lower()


def wait_ready(api, client, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if api.poll() is not None:
            raise RuntimeError("La API se cerro antes de cargar el modelo. Revisa el mensaje anterior.")
        try:
            response = client.get("/health")
            if response.status_code == 200 and response.json() == {"status": "ok"}:
                return
        except (httpx.TransportError, ValueError):
            pass
        time.sleep(0.25)
    raise RuntimeError("El modelo no esta listo tras 180 s. Revisa QA_MODEL_DIR, QA_TEAM y sus archivos.")


def stop_owned(process):
    if process is None or process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.terminate()
        process.wait(timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        if process.poll() is None:
            if os.name == "nt":
                # Kill only the tree started by this launcher (venv can add a child).
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               check=False)
            else:
                process.kill()
            process.wait(timeout=10)


def main():
    api = tunnel = None
    try:
        os.chdir(ROOT)
        if not (ROOT / ".env").is_file():
            raise ValueError("Crea .env en la raiz con QA_API_TOKEN, QA_TEAM y QA_MODEL_DIR.")
        # Same precedence as uvicorn --env-file: existing environment wins.
        load_dotenv(ROOT / ".env", override=False)
        if not os.environ.get("QA_API_TOKEN"):
            raise ValueError("Falta QA_API_TOKEN en .env o en el entorno.")
        team = os.environ.get("QA_TEAM", "")
        if not team or any(char.isspace() for char in team):
            raise ValueError("QA_TEAM es obligatorio y no debe contener espacios.")
        ngrok = shutil.which(os.environ.get("NGROK_PATH") or "ngrok")
        if not ngrok:
            raise ValueError("No se encuentra ngrok. Agregalo al PATH o define NGROK_PATH en .env.")
        value = os.environ.get("NGROK_DOMAIN") or input("Dominio fijo asignado por ngrok: ")
        url = domain_url(value)
        try:
            with socket.socket() as probe:
                if os.name == "nt":
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                probe.bind(("127.0.0.1", 8000))
        except OSError:
            raise ValueError("El puerto 8000 esta ocupado. Deten la API anterior antes de usar este lanzador.") from None

        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        print("Iniciando API en CPU. Esperando a que el modelo este listo...", flush=True)
        api = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "src.service.app:app",
             "--host", "127.0.0.1", "--port", "8000", "--workers", "1"],
            cwd=ROOT, creationflags=flags,
        )
        with httpx.Client(base_url=LOCAL_URL, timeout=2, trust_env=False) as client:
            wait_ready(api, client)
            # Check access control before publishing the endpoint.
            if client.get("/info").status_code != 401:
                raise RuntimeError("La API no rechazo la peticion sin token. No se abrira el tunel.")
            response = client.get("/info", headers={
                "Authorization": "Bearer " + os.environ["QA_API_TOKEN"]})
            if response.status_code != 200:
                raise RuntimeError("No se pudo autenticar contra la API. No se abrira el tunel.")

        print(f"API lista. Solicitando dominio: {url}", flush=True)
        print("Espera la confirmacion de ngrok. Ctrl+C detiene el tunel y la API.", flush=True)
        tunnel = subprocess.Popen(
            [ngrok, "http", LOCAL_URL, "--url", url, "--inspect=false"],
            cwd=ROOT, creationflags=flags,
        )
        while tunnel.poll() is None:
            if api.poll() is not None:
                raise RuntimeError("La API se cerro; se detendra tambien el tunel.")
            time.sleep(0.25)
        if tunnel.returncode:
            raise RuntimeError("ngrok termino con error. Revisa su mensaje, autenticacion y dominio asignado.")
        return 0
    except KeyboardInterrupt:
        print("\nDeteniendo tunel y API...")
        return 0
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 1
    except (OSError, httpx.HTTPError):
        print("ERROR: No se pudo iniciar o conectar con los procesos locales.")
        return 1
    finally:
        stop_owned(tunnel)
        stop_owned(api)


if __name__ == "__main__":
    sys.exit(main())
