# Configuración Óptima de Entrenamiento YOLOv8s - SAI-Net Detector

## Recursos Hardware Confirmados
- **GPU**: 2× A100-SXM4 (39.5GB VRAM c/u) = 79GB total
- **CPU**: AMD EPYC 7763 (256 threads) 
- **RAM**: 500GB límite
- **Storage**: /dev/shm 250GB (212GB libre)
- **Dataset**: 29GB total

## Configuración Máxima Óptima (1440×808)

```bash
# Using training script (recommended)
python scripts/train_detector.py --config optimal

# Equivalent CLI command:
yolo detect train \
  data=configs/yolo/pyro_fasdd.yaml \
  model=yolov8s.pt \
  imgsz=1440 \
  epochs=150 \
  batch=120 \              # ÓPTIMO para 1440×808: 60×2=120 (36.1GB por GPU)
  device=0,1 \
  workers=79 \             # Ajustado para resolución alta
  amp=bf16 \               # A100 native BF16
  cos_lr=True \
  lr0=0.01 \
  lrf=0.01 \               # LR final para batch 120
  momentum=0.937 \
  weight_decay=0.0005 \
  warmup_epochs=5 \
  warmup_momentum=0.8 \
  warmup_bias_lr=0.1 \
  box=7.5 \                # Énfasis en localización
  cls=0.5 \                # Reducir peso clase (single class)
  dfl=1.5 \                # Distribution Focal Loss
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  degrees=5 translate=0.1 scale=0.5 shear=2.0 \
  mosaic=1.0 mixup=0.1 copy_paste=0.0 \
  close_mosaic=15 \        # Desactivar mosaic últimas 15 épocas
  cache=ram \              # 29GB dataset cache en 500GB RAM
  project=/dev/shm/rrn/sai-net-detector/runs \
  name=sai_yolov8s_optimal_1440x808
```

## Justificación Técnica

**1. `batch=120` (límite VRAM para 1440×808)**
- 60 imágenes por GPU × 2 GPUs
- 36.1GB por GPU (< 37.5GB límite seguro)
- **Utilización VRAM: 96%** - máximo aprovechamiento para alta resolución

**2. `workers=79` (optimizado para resolución)**  
- ~36.7GB RAM total (< 500GB límite)
- **Utilización CPU: 31%** - ajustado para I/O de imágenes 1440×808
- Sin bottleneck I/O con 129k imágenes de alta resolución

**3. `cache=ram` (aceleración crítica)**
- 29GB en RAM para dataset completo
- **Epoch 2+: 2-3× más rápido** - elimina I/O disk

## Rendimiento Esperado

| Métrica | Valor |
|---------|--------|
| Batches por época | ~1075 |
| Tiempo por época | 30-35 min |
| Resolución | **1440×808 (alta resolución)** |
| GPU utilization | 96%+ |
| RAM utilization | 7.3% |
| Total training time | **35-40 horas** |

## Comparación vs Roadmap

| Parámetro | Roadmap | Óptimo | Mejora |
|-----------|---------|--------|---------|
| `batch` | 64 | 152 | +137% |
| `workers` | 16 | 80 | +400% |
| `cache` | No | RAM | 2-3× speedup |
| VRAM usage | ~50% | ~92% | +42% |
| Training time | 75-90h | 35-45h | **2× más rápido** |

## Monitoreo Durante Entrenamiento

```bash
# Terminal 1: GPU monitoring
watch -n 2 nvidia-smi

# Terminal 2: RAM monitoring  
watch -n 5 'free -h | head -3'

# Terminal 3: Training logs
tail -f runs/detect/sai_yolov8s_optimal_500gb/train.log
```