# Extractive QA as a Service

API local en CPU para el modelo entrenado por el equipo. Contrato definitivo:
[`docs/qa_api_contract.yaml`](docs/qa_api_contract.yaml). El servidor reutiliza
`run_inference()` y `apply_threshold()` de Javier, sin modificar `src/training/`.
No descarga modelos ni utiliza servicios externos de inferencia.

## Instalar en Windows / VS Code

Usar **Python 3.12 de 64 bits**. Desde la terminal PowerShell en la raíz del
repositorio existente:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r src/requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Se instala primero la distribución CPU de PyTorch; no hace falta CUDA. En VS Code,
seleccionar `.venv\Scripts\python.exe` con **Python: Select Interpreter**.
Los comandos usan ese intérprete directamente y no requieren activar scripts.
Entorno comprobado: Python 3.12.10, PyTorch 2.14.1+cpu, Transformers 5.20.2,
FastAPI 0.162.0, Pydantic 2.14.1, Uvicorn 0.54.0, pytest 9.1.1 y httpx 0.28.1.
Al recibir el export, contrastar las versiones con `environment` en su
`run_config.json`; la prueba con los pesos finales sigue pendiente.
Se conservan las dependencias de entrenamiento; `tqdm` se declara explícitamente
porque evaluación lo importa. PyTorch requiere al menos 2.6 para la carga segura
de pesos `.bin`, y Pydantic se mantiene en su API 2.x.

## Configurar y ejecutar

En la misma terminal PowerShell, desde la raíz:

```powershell
$env:QA_API_TOKEN = [System.Net.NetworkCredential]::new('', (Read-Host 'Token entregado por el profesor' -AsSecureString)).Password
$env:QA_TEAM = Read-Host 'Nombre real registrado del equipo, sin espacios'
$env:QA_MODEL_DIR = '.\models\final_model'
.\.venv\Scripts\python.exe -m uvicorn src.service.app:app --host 127.0.0.1 --port 8000 --workers 1
```

El token solo se lee de `QA_API_TOKEN`; no se imprime ni se guarda en archivos.
`.env` no se carga automáticamente. Cada terminal tiene su propio entorno.
Variables opcionales:

| Variable | Uso / valor por defecto |
| --- | --- |
| `QA_RUN_CONFIG` | Ruta explícita al `run_config.json` original; si se define, debe existir. |
| `QA_BASE_MODEL` | Checkpoint base confirmado, solo si falta ese metadato. |
| `QA_TORCH_THREADS` | Entero positivo; por defecto `min(4, núcleos lógicos)`. |
| `QA_BATCH_SIZE` | Ventanas por batch, entero positivo; por defecto `8`. |

La carga y el calentamiento ocurren una sola vez en segundo plano. `/health`
responde `503 {"status":"loading"}` hasta terminar. Pesos ausentes/incompletos,
configuración inválida o un nombre de equipo ausente mantienen el servicio no
disponible. Sin token configurado tampoco se anuncia disponibilidad. Revisar los
archivos/variables y reiniciar después de corregir un fallo o copiar nuevos pesos;
no hay recarga automática. Con credenciales válidas, `/info` y `/predict` devuelven
`500 {"error":"internal_error"}` mientras no exista un modelo listo: el contrato
reserva `503/loading` para `/health`.

## Probar la API

Con el servidor abierto, usar otra terminal que tenga `QA_API_TOKEN` configurado
mediante la misma lectura segura anterior:

```powershell
curl.exe -i http://127.0.0.1:8000/health
curl.exe -i http://127.0.0.1:8000/info
$headers = @{ Authorization = "Bearer $env:QA_API_TOKEN" }
Invoke-RestMethod http://127.0.0.1:8000/info -Headers $headers
$body = @{ id = 'q-017'; context = 'The river is 422 km long.'; question = 'How long is the river?' } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/predict -Method Post -Headers $headers -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

`/health` es público; `/info` sin token devuelve 401. El esquema `Bearer` y el token
son sensibles a mayúsculas. Las respuestas de predicción contienen exactamente
`id`, `answer`, `start`, `score`; los spans siempre se verifican contra el contexto.
La abstención usa `delta = null_score - span_score`, `null` cuando `delta > threshold`,
y el score original `sigmoid(min(abs(delta - threshold), 50))` de evaluación.

## Pruebas y límites

```powershell
.\.venv\Scripts\python.exe -m compileall -q src
.\.venv\Scripts\python.exe -m pytest src/tests -q
```

Las pruebas no necesitan pesos ni descargas: usan mocks, un tokenizer construido
localmente y un modelo diminuto aleatorio en una carpeta temporal para comprobar
la exportación/carga. No evalúan exactitud ni los cinco casos públicos del modelo
final. Comprueban además las respuestas contra los esquemas del YAML del profesor.

La inferencia usa CPU, `model.eval()` y `torch.inference_mode()`, batches de ventanas,
recorte del padding ya implementado por Javier y un bloqueo para evitar inferencias
CPU simultáneas que saturen los hilos. Usar **un worker**, sin `--reload`, durante
la presentación. La cola de peticiones concurrentes puede aumentar la latencia.
El código de evaluación requiere tokenizer rápido, padding a la derecha y CLS en
posición cero; se verifican al cargar. Si una pregunta válida de hasta 500 caracteres
supera el presupuesto de tokens de una ventana, solo se recorta su final para dejar
espacio al contexto y conservar el stride calibrado. Nunca se descarta parte del
contexto. Este caso límite puede reducir la información de la pregunta.

**Pendiente con Javier:** pesos finales, tokenizer, configuración de postprocesado,
metadatos originales y versiones usadas en Colab. Luego ejecutar los cinco casos
públicos y medir casos cortos, largos y sin respuesta en CPU y a través del túnel.
No se garantiza todavía el límite de 5 s ni el promedio inferior a 3 s (150 s para
50 preguntas, incluyendo red). No se han inventado métricas ni completado REPORT.md.

## Túnel HTTPS (solo en la etapa de presentación)

Después de obtener `200/ok` local, instalar/configurar una herramienta permitida y
ejecutar manualmente una de estas opciones en otra terminal:

```powershell
cloudflared tunnel --url http://127.0.0.1:8000
# Alternativa, con ngrok previamente autenticado:
ngrok http 8000
```

Probar la URL HTTPS desde otra red, medir el tiempo total y preparar una segunda
herramienta como respaldo. Mantener la laptop encendida y conectada. Compartir el
token por el canal privado acordado, nunca en URLs, capturas o Git. Esta implementación
no abre túneles, no despliega servicios y no crea el tag `submission`.

Referencias técnicas: [instalación CPU de PyTorch](https://pytorch.org/get-started/locally/),
[ciclo de vida de FastAPI](https://fastapi.tiangolo.com/advanced/events/) y
[errores de validación](https://fastapi.tiangolo.com/tutorial/handling-errors/).
