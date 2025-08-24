Respuesta corta:
Para entrenar el detector YOLOv8 con FASDD + PyroSDIS en 2×A100 te recomiendo un plan en dos etapas que, en total, apunte a ~150-170 
épocas efectivas con early stopping:

Etapa 1 – Pre‑entrenamiento en FASDD: 100–110 épocas (objetivo: aprender variedad y objetos chicos; patience≈10, cosine LR, warmup 
5–10 épocas).

Etapa 2 – Fine‑tuning en PyroSDIS (single‑class "smoke"): 40–60 épocas (reducí LR 10×, activá single_cls=True, patience≈10).

Con early stopping lo típico es que converja entre 120–170 épocas totales; mantené el tope en 170 por eficiencia. Esta pauta está 
alineada con: (i) el tamaño y formato de FASDD (~100K imágenes, boxes de fire/smoke) y PyroSDIS (~33K imágenes, monoclase smoke con 
boxes), y (ii) la recomendación de usar YOLOv8 para localización por su cabeza anchor‑free (C2f) y eficacia en incendios/humo, 
mientras el verificador aprende temporalidad en FIgLib (SmokeyNet‑like).

Detalle operativo (para pasar directo a la IA code assistant)

Escenario: 2×A100 (80 GB o 40 GB), Distributed Data Parallel (DDP), mezcla FASDD→PyroSDIS, imagen 896–1024 para favorecer objetos 
pequeños.

1) Preparación de datos

FASDD: dos clases (fire, smoke) con bounding boxes; usar tal cual para detección. Aplicar Mosaic, MixUp, CutMix y hsv/blur moderados.

PyroSDIS: monoclase smoke; activar single_cls=True en Ultralytics/YOLOv8 para unificar a “smoke” y evitar degradar por la clase 
“fire” de FASDD en el fine‑tune.

2) Hiperparámetros base (Ultralytics YOLOv8 s/m)

Imagen: imgsz=[1440, 808] (resolución rectangular obligatoria para SAI-Net, optimizada para cámaras fijas).

Batch global (2×A100): batch=60 total (2 GPU × 30 por GPU, optimizado para resolución 1440×808 en A100-40GB).

Optimización: optimizer=sgd o adamw; lr0≈0.01 (SGD) o 0.0005 (AdamW), lrf=0.01, cos_lr=True, warmup_epochs=5–10.

Aumentos: mosaic=1.0 en FASDD (bajalo a 0.5 en PyroSDIS), mixup=0.1–0.2, copy_paste=0.0–0.1, color jitter leve (el humo es sutil).

Small objects: imgsz=[1440,808], iou=0.6–0.7, box=7.5–8.0, cls=0.5–0.8 (ajustá loss weights si ves exceso de falsos positivos).

Callbacks: early stopping patience=10 (ambas etapas), EMA activado, amp (FP16), DDP.

Checkpointing: guardar el mejor por mAP50 y monitorear también recall (nos interesa no perder humos tenues).

3) Calendario de entrenamiento propuesto

Etapa 1 (FASDD)

epochs=110 (tope), early stop patience=10.

Objetivo: estabilizar mAP50 y subir recall en smoke; dejar fire como auxiliar (si tu data.yaml lo incluye).

Etapa 2 (PyroSDIS, single‑class)

Cargar best.pt de Etapa 1 → reducir LR 10× → epochs=60 (tope), early stop patience=10.

Objetivo: adaptar el head al dominio monoclase y cámaras fijas tipo Pyronear, mejorando precision para “smoke”.

Por qué dos etapas: FASDD es más grande y diverso (aprende “qué es humo” en muchos contextos); PyroSDIS refina al dominio “cámara 
fija, smoke monoclase” que usaremos como trigger del verificador temporal.

4) Métricas de corte (aceptación)

Val: mAP50-smoke ≥ 0.85 en FASDD y precision ≥ 0.90 + recall ≥ 0.80 en PyroSDIS (ajustá según ground truth).

Operativo: recordá que el detector es la primera etapa; el verificador SmokeyNet‑like (CNN+LSTM+ViT) sobre FIgLib reduce falsos 
positivos persistiendo sobre 2–3 frames (ventana corta).

Notas y justificación técnica

YOLOv8: cabeza anchor‑free + C2f y buen rendimiento en humo/fuego con latencia baja en GPU; es la elección natural para la 
localización inicial antes del verificador temporal.

Datasets:

FASDD: >100K imágenes con cajas de fire/smoke (gran variedad, muchos objetos pequeños).

PyroSDIS: ~33K imágenes, monoclase “smoke” con cajas, ya pensado para YOLO; ideal para el fine‑tune final.

Arquitectura global SAI: detector rápido + verificador temporal (SmokeyNet‑like) entrenado con FIgLib (secuencias etiquetadas 
smoke/no‑smoke) para capturar dinámica y bajar falsas alarmas.

TL;DR

Apuntá a ~170 épocas totales con early stopping: 110 en FASDD + 60 en PyroSDIS (probablemente se corte antes entre 120–170). Con 
2×A100, usá DDP, imgsz=[1440,808], aumentos fuertes en FASDD y más suaves en PyroSDIS, y single_cls=True en la segunda etapa. El 
verificador SmokeyNet‑like entrenado en FIgLib hará la segunda pasada temporal para confirmar.
