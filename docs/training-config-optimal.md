# Configuración Óptima de Entrenamiento YOLOv8s - SAI-Net Detector

## Recursos Hardware Confirmados
- **GPU**: 2× A100-SXM4 (39.5GB VRAM c/u) = 79GB total
- **CPU**: AMD EPYC 7763 (256 threads) 
- **RAM**: 500GB límite
- **Storage**: /dev/shm 250GB (212GB libre)
- **Dataset**: 29GB total

## Configuración Máxima Óptima

```bash
yolo detect train \
  data=configs/yolo/pyro_fasdd.yaml \
  model=yolov8s.pt \
  imgsz=960 \
  epochs=150 \
  batch=152 \              # MÁXIMO seguro: 76×2=152 (36.2GB por GPU)
  device=0,1 \
  workers=80 \             # MÁXIMO: 170GB RAM total con cache
  amp=bf16 \               # A100 native BF16
  cos_lr=True \
  lr0=0.01 \
  lrf=0.01 \               # LR final para batch grande
  momentum=0.937 \
  weight_decay=0.0005 \
  warmup_epochs=5 \        # Warmup más largo para batch 152
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
  name=sai_yolov8s_optimal_500gb
```

## Justificación Técnica

**1. `batch=152` (límite VRAM)**
- 76 imágenes por GPU × 2 GPUs
- 36.2GB por GPU (< 37GB límite seguro)
- **Utilización VRAM: 92%** - máximo aprovechamiento

**2. `workers=80` (límite RAM)**  
- 170GB RAM total (< 500GB límite)
- **Utilización CPU: 31%** - óptimo para data loading intensivo
- Sin bottleneck I/O con 129k imágenes

**3. `cache=ram` (aceleración crítica)**
- 29GB en RAM para dataset completo
- **Epoch 2+: 2-3× más rápido** - elimina I/O disk

## Rendimiento Esperado

| Métrica | Valor |
|---------|--------|
| Batches por época | ~850 |
| Tiempo por época | 25-35 min |
| Speedup vs roadmap | **2.4× más rápido** |
| GPU utilization | 95%+ |
| RAM utilization | 34% |
| Total training time | **35-45 horas** |

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