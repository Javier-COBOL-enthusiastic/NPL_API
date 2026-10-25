# Plan de acción — Extractive QA as a Service

## Objetivo

Construir un servicio de **preguntas y respuestas extractivas en inglés** basado en SQuAD 2.0. Dado un contexto y una pregunta, la API devuelve un fragmento literal del contexto o `null` si la pregunta no se puede responder con ese contexto.

El equipo debe elegir un checkpoint base permitido y **hacer su propio fine-tuning**. El entrenamiento se ejecutará en Google Colab cuando haya GPU disponible; los pesos, el tokenizer y los checkpoints de recuperación se guardarán en Google Drive. La API final se ejecutará en la computadora local usando CPU.

---

## 1. Arquitectura general

```text
Google Colab (entrenamiento)
  ├─ Descarga SQuAD 2.0 y el checkpoint base
  ├─ Fine-tuning del modelo
  ├─ Guarda checkpoints, métricas y logs en Google Drive
  └─ Exporta el mejor modelo + tokenizer a Drive
                  │
                  ▼
          Google Drive
     ├─ checkpoints de recuperación
     ├─ mejor modelo entrenado
     ├─ tokenizer/configuración
     ├─ logs y métricas CSV
     └─ resultados de validación
                  │
         Descargar / copiar
                  ▼
Computadora local (CPU)
  ├─ FastAPI: validación, autenticación y predicción
  ├─ Uvicorn: servidor ASGI en el puerto 8000
  ├─ Modelo y tokenizer descargados desde Drive
  └─ Túnel HTTPS público (ngrok / Cloudflare Quick Tunnel / Tailscale Funnel)
```

**Separación importante:** Colab se usa para entrenar, no para alojar la API final. La API no dependerá de que el notebook de Colab siga encendido.

## 2. Requisitos del proyecto

### Tarea y modelo

- Entrada: `context` y `question`, ambos en inglés.
- Salida: una respuesta que sea un fragmento literal del contexto, o `null` si no hay respuesta.
- Usar un encoder base permitido y hacer fine-tuning propio.
- No usar un modelo ya fine-tuned en SQuAD como checkpoint inicial.
- No implementar un decoder GPT; investigarlo y discutirlo en el informe en relación con problemas reales encontrados.

### Contrato de la API

- `GET /health`: público. Devuelve `{"status":"ok"}` únicamente cuando el modelo está cargado; mientras carga, responde HTTP 503 con `{"status":"loading"}`.
- `GET /info`: protegido. Incluye `team`, `model` (identificador del checkpoint base de Hugging Face), `params` (entero) y, opcionalmente, `max_seq_length` y `doc_stride`.
- `POST /predict`: protegido. El JSON acepta exactamente `id`, `context`, `question`; rechazar campos adicionales.
- `id`: 1–64 caracteres.
- `context`: 1–5000 caracteres.
- `question`: 1–500 caracteres.
- Respuesta: `id`, `answer` (string o `null`), `start` (entero o `null`) y `score` (float entre 0 y 1).
- Si `answer` no es `null`, debe cumplirse `context[start:start+len(answer)] == answer`. `start` será `null` si y solo si `answer` es `null`.
- Errores JSON: `{"error":"unauthorized"}`, `{"error":"bad_request"}` o `{"error":"internal_error"}`.
- Autenticación: `Authorization: Bearer <token>`, comparado de forma exacta y sensible a mayúsculas/minúsculas. Leer el token desde la variable de entorno `QA_API_TOKEN`; nunca guardarlo en el repositorio.
- La API debe responder dentro del límite de 5 segundos por solicitud.
- Ejecutar el servicio en el puerto 8000 y exponerlo mediante un túnel HTTPS público.

### Entregables

- Repositorio con historial Git conservado; no hacer squash de los commits.
- Código fuente, sin subir los pesos del modelo al repositorio.
- Logs de entrenamiento y curvas de loss.
- API desplegada y accesible mediante HTTPS.
- `public_sample.json` evaluado; incluye 5 casos, 2 sin respuesta y 1 contexto de más de 512 tokens.
- `REPORT.md` en inglés, 1–2 páginas, con todos los encabezados de la plantilla:
  1. Team
  2. Model choice
  3. Training
  4. No-answer threshold
  5. Serving architecture
  6. Public sample results
  7. Model size/latency
  8. At least two real problems with cause/fix/commit
  9. Decoder investigation linked to actual problems
  10. Known limitations
- Pitch de 3 minutos: elección del modelo, entrenamiento, arquitectura de servicio e investigación sobre decoders.

---

## 3. Organización en Google Drive

Crear una carpeta, por ejemplo `MyDrive/ExtractiveQA/`, con esta estructura:

```text
ExtractiveQA/
├── checkpoints/
│   ├── checkpoint-.../
│   └── ...
├── final_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── otros archivos exportados por save_pretrained()
├── logs/
│   ├── train_metrics.csv
│   ├── eval_metrics.csv
│   └── training_log.txt
├── results/
│   ├── validation_predictions.json
│   ├── threshold_sweep.csv
│   └── public_sample_results.json
└── config/
    └── run_config.json
```

Los nombres exactos de los archivos del modelo pueden variar según el tokenizer y la versión de Transformers. Lo importante es guardar el modelo completo y el tokenizer con `save_pretrained()`.

**No sincronizar ni subir los pesos a Git.** Añadir a `.gitignore` rutas como `checkpoints/`, `final_model/`, `*.safetensors`, `pytorch_model.bin` y cualquier archivo de credenciales. Los pesos deben estar en Drive y, cuando toque desplegar, copiarse a la máquina local.

---

## 4. Fases de trabajo

### Fase 1 — Preparar el repositorio local

1. Crear el repositorio con una estructura clara:

   ```text
   src/
   ├── service/
   │   ├── app.py
   │   ├── schemas.py
   │   ├── inference.py
   │   └── auth.py
   ├── training/
   │   ├── train.ipynb
   │   ├── train.py
   │   └── evaluate.py
   ├── tests/
   ├── public_sample.json
   ├── REPORT.md
   ├── requirements.txt
   └── .gitignore
   ```

2. Guardar el notebook de Colab en el repositorio como archivo `.ipynb` o conservar un enlace documentado al notebook del equipo.
3. Añadir al `.gitignore` los checkpoints, pesos, entornos virtuales, cachés y secretos.
4. Hacer commits pequeños y descriptivos por fase.

### Fase 2 — Elegir y validar el checkpoint base

1. Seleccionar un encoder base compatible con extractive QA, por ejemplo `distilbert/distilbert-base-uncased`, después de comprobar que el identificador existe y que no es un checkpoint fine-tuned en SQuAD.
2. Registrar el identificador exacto, tamaño del modelo y número de parámetros.
3. Usar una cabeza de QA adecuada para predecir posiciones de inicio y fin. No entrenar desde cero un modelo de lenguaje.
4. Registrar las versiones de Python, PyTorch, Transformers y Datasets para poder reproducir el experimento.

### Fase 3 — Preparar Google Colab y Drive

1. Abrir un notebook en Google Colab e iniciar sesión con la cuenta que contiene el Drive del equipo.
2. En la configuración del notebook, elegir una GPU si está disponible. El acceso y la duración de las sesiones de Colab pueden variar; no asumir que la GPU estará disponible ni que una sesión gratuita durará toda la noche.
3. Montar Google Drive desde el notebook:

   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   ```

4. Definir una ruta de trabajo, por ejemplo:

   ```python
   from pathlib import Path

   DRIVE_ROOT = Path("/content/drive/MyDrive/ExtractiveQA")
   CHECKPOINT_DIR = DRIVE_ROOT / "checkpoints"
   FINAL_MODEL_DIR = DRIVE_ROOT / "final_model"
   LOG_DIR = DRIVE_ROOT / "logs"
   RESULTS_DIR = DRIVE_ROOT / "results"

   for directory in (CHECKPOINT_DIR, FINAL_MODEL_DIR, LOG_DIR, RESULTS_DIR):
       directory.mkdir(parents=True, exist_ok=True)
   ```

5. Guardar la configuración de la ejecución (`run_config.json`): checkpoint base, seed, hiperparámetros, número de ejemplos y versiones de las bibliotecas.
6. No imprimir ni guardar `QA_API_TOKEN` en el notebook. El token corresponde al servicio y se configura por separado en la máquina donde corre la API.

### Fase 4 — Entrenar con recuperación ante desconexiones

1. Descargar SQuAD 2.0 y cargar el checkpoint base dentro de Colab.
2. Hacer un primer piloto con una muestra de entrenamiento para medir tiempo por época y consumo de memoria.
3. Para el experimento inicial, probar como punto de partida:
   - 20 000–30 000 ejemplos para el piloto.
   - 2–3 épocas.
   - Learning rate de `3e-5`.
   - Batch size de 8, ajustándolo según la memoria disponible.
   - `max_seq_length = 384`.
   - `doc_stride = 128`.
   - Warmup de 5–10 %.
4. No tomar estos valores como definitivos: comparar los resultados de validación y la latencia, y registrar los cambios.
5. Configurar checkpoints guardados periódicamente en Drive, por ejemplo cada cierto número de pasos, y conservar un número limitado de checkpoints recientes para no llenar el espacio.
6. Guardar métricas y logs en Drive, no únicamente en el disco temporal `/content`.
7. Guardar al menos:
   - Loss de entrenamiento.
   - Loss y métricas de evaluación por época.
   - Paso y época actuales.
   - Configuración de entrenamiento.
   - Mejor checkpoint según una métrica de validación elegida.
8. Para poder **reanudar** una sesión interrumpida, conservar también el estado del optimizador y del scheduler dentro de los checkpoints de entrenamiento. Guardar solo los pesos del modelo permite reutilizarlos, pero normalmente no reanuda exactamente el entrenamiento desde el mismo paso.
9. Si la sesión se desconecta, volver a montar Drive y reanudar desde el checkpoint de entrenamiento más reciente válido.
10. Al terminar, exportar el mejor modelo y tokenizer a `final_model/`, y guardar las métricas finales.

Ejemplo conceptual de rutas para Hugging Face `Trainer`:

```python
training_args = TrainingArguments(
    output_dir=str(CHECKPOINT_DIR),
    eval_strategy="epoch",
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    logging_strategy="steps",
    logging_steps=50,
    report_to="none",
    # Añadir aquí los demás hiperparámetros del experimento.
)
```

La firma de `TrainingArguments` cambia entre versiones de Transformers. Si la versión instalada usa `evaluation_strategy` en lugar de `eval_strategy`, emplear el nombre compatible y registrar la versión utilizada. Verificar que `save_steps` sea razonable para el tamaño del conjunto y el tiempo de sesión.

Al finalizar el entrenamiento, guardar el artefacto de inferencia:

```python
trainer.save_model(str(FINAL_MODEL_DIR))
tokenizer.save_pretrained(str(FINAL_MODEL_DIR))
```

**Importante:** el código de entrenamiento debe incluir la preparación correcta de etiquetas `start_positions` y `end_positions`, el manejo de preguntas sin respuesta y la alineación de offsets. El ejemplo de configuración anterior no constituye un script de entrenamiento completo.

### Fase 5 — Evaluar el modelo y calibrar `null`

1. Evaluar en el conjunto de validación de SQuAD 2.0, manteniendo separados los datos de entrenamiento y validación.
2. Calcular métricas de respuestas extractivas, incluyendo Exact Match (EM) y F1; revisar por separado las preguntas respondibles y las no respondibles.
3. Calcular el puntaje de la mejor respuesta candidata y el puntaje de no-respuesta, y definir un margen consistente, por ejemplo:

   `delta = score_no_answer - score_best_span`

4. Elegir el umbral de `delta` usando la validación, no mirando solo los cinco ejemplos públicos.
5. Guardar el barrido de umbrales y las predicciones de validación en `results/`.
6. Probar que cada respuesta devuelta sea un fragmento exacto del contexto original, respetando mayúsculas, puntuación y espacios.
7. Ejecutar los cinco casos de `public_sample.json`, incluyendo el contexto largo y los dos casos sin respuesta. Guardar las salidas en `public_sample_results.json`.
8. Documentar el umbral elegido, la métrica usada y los errores que siguen ocurriendo.

### Fase 6 — Descargar y probar el modelo en la computadora local

1. Desde Google Drive, descargar la carpeta `final_model/` a la computadora local. También puede usarse la sincronización de Drive si está configurada.
2. Colocar el modelo en una ruta local que no se suba a Git, por ejemplo `models/final_model/`.
3. Instalar las dependencias de inferencia en un entorno virtual local.
4. Cargar el modelo desde la carpeta local, sin depender de una descarga de Hugging Face durante cada petición.
5. Ejecutar un script de prueba en CPU con ejemplos cortos, largos y sin respuesta.
6. Medir tamaño en disco y latencia de inferencia. Optimizar solo si hace falta; el límite de cada solicitud es 5 segundos.
7. Confirmar que el servicio pueda arrancar con los pesos locales aunque Colab esté apagado.

### Fase 7 — Construir la API con FastAPI

1. Implementar `GET /health`, `GET /info` y `POST /predict`.
2. Validar estrictamente los campos, tipos, límites de longitud y campos adicionales.
3. Implementar autenticación Bearer para `/info` y `/predict`; mantener `/health` público.
4. Leer `QA_API_TOKEN` desde el entorno. Si no está definido, fallar de forma segura al iniciar o dejar protegidas las rutas sin aceptar peticiones no autenticadas.
5. Cargar el modelo una vez durante el arranque y reflejar el estado correcto en `/health`.
6. En inferencia, convertir los índices de tokens a offsets del contexto original para devolver un fragmento literal y un `start` correcto.
7. Aplicar el umbral de no-respuesta seleccionado en validación.
8. Ejecutar con Uvicorn, por ejemplo:

   ```bash
   uvicorn service.app:app --host 0.0.0.0 --port 8000
   ```

9. Probar los casos normales y errores: token ausente/incorrecto, JSON inválido, campos adicionales, límites de longitud, pregunta sin respuesta, contexto largo y fallo interno.
10. Asegurar que las respuestas JSON y códigos HTTP coincidan con el contrato requerido.

### Fase 8 — Publicar con HTTPS y medir

1. Iniciar la API local en el puerto 8000.
2. Crear un túnel HTTPS público con la herramienta autorizada por el equipo, como ngrok, Cloudflare Quick Tunnel o Tailscale Funnel.
3. Probar la URL pública desde una red o dispositivo diferente si es posible.
4. Comprobar `/health`, `/info` con autenticación y `/predict` con autenticación.
5. Verificar el límite de 5 segundos y registrar varias latencias, incluyendo contexto largo.
6. No incluir el token en URLs, capturas públicas, logs ni el repositorio.
7. Recordar que un túnel gratuito puede cambiar de URL al reiniciarse; actualizar la URL entregada al evaluador cuando sea necesario.

### Fase 9 — Informe, problemas reales y presentación

1. Completar `REPORT.md` en inglés y respetar la extensión de 1–2 páginas.
2. Registrar al menos dos problemas reales encontrados durante el trabajo. Para cada uno, describir causa, solución y hash de commit; no inventar problemas ni commits.
3. Relacionar la investigación de decoders con problemas observados de latencia, generación de texto, alineación de spans o abstención, según corresponda.
4. Incluir resultados públicos, umbral de no-respuesta, tamaño del modelo, latencia y limitaciones conocidas.
5. Preparar el pitch de 3 minutos: checkpoint elegido, proceso de fine-tuning en Colab, almacenamiento en Drive, evaluación y calibración, arquitectura local de servicio e investigación sobre decoders.

---

## 5. Pruebas de aceptación

- [ ] El modelo inicial es un encoder base permitido, no un checkpoint ya fine-tuned en SQuAD.
- [ ] El equipo hizo fine-tuning y conserva logs y configuración reproducible.
- [ ] Los pesos finales y checkpoints de recuperación están en Google Drive.
- [ ] El mejor modelo y tokenizer se descargaron y cargan localmente sin depender de Colab.
- [ ] El modelo funciona en CPU.
- [ ] `GET /health` es público y refleja si el modelo está listo.
- [ ] `/info` y `/predict` exigen el Bearer token.
- [ ] El contrato JSON, las validaciones y los errores coinciden con la especificación.
- [ ] Cada respuesta no nula coincide exactamente con un fragmento del contexto y tiene un `start` correcto.
- [ ] `null` se devuelve para casos sin respuesta según el umbral calibrado.
- [ ] Los cinco ejemplos públicos están probados y guardados.
- [ ] La API está disponible mediante HTTPS público y responde dentro de 5 segundos por solicitud.
- [ ] El repositorio no contiene pesos ni secretos y conserva el historial Git.
- [ ] `REPORT.md` y el pitch están completos.

## 6. Cronograma sugerido

| Día | Trabajo |
|---|---|
| 1 | Repositorio, dependencias, elección del checkpoint y prueba local de carga |
| 2 | Notebook de Colab, montaje de Drive y prueba corta de entrenamiento |
| 3–4 | Fine-tuning, checkpoints, logs y recuperación de una sesión de prueba |
| 5 | Evaluación, EM/F1, calibración del umbral y ejemplos públicos |
| 6 | Descargar el modelo y medir inferencia en CPU |
| 7–8 | API, validación, autenticación y pruebas automatizadas |
| 9 | Túnel HTTPS, latencia y prueba externa |
| 10 | Informe, evidencia de problemas/commits y pitch |

El cronograma es orientativo. Ajustarlo según la disponibilidad de GPU en Colab y el tiempo real por época medido en el piloto.