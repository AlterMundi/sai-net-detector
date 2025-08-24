# 📋 Auditoría de Configuración de Entrenamiento SAI-Net
## Comparación contra Documentación Sagrada

**Fecha**: 2025-01-15  
**Auditor**: Claude Code  
**Documentos sagrados**: `docs/planentrenamientoyolov8.md`, `docs/Guia Descarga PyroSDIS y FASDD.md`

---

## 🎯 RESUMEN EJECUTIVO

**Estado**: ❌ **MÚLTIPLES DESVIACIONES CRÍTICAS DETECTADAS**  
**Configuraciones actuales**: DESALINEADAS con documentación sagrada  
**Acción requerida**: CORRECCIÓN INMEDIATA antes de entrenar

---

## 🔍 ANÁLISIS DETALLADO DE DESVIACIONES

### 🚨 CRÍTICAS (Prioridad Máxima)

#### 1. **ÉPOCAS INCORRECTAS**
| Parámetro | Sagrado | Actual | Estado |
|-----------|---------|---------|---------|
| **Stage 1 epochs** | **140-160** | **110** | ❌ **BAJO** |
| **Stage 2 epochs** | **40-60** | **60** | ✅ OK |
| **Early stopping Stage 1** | **patience=30** | **patience=10** | ❌ **DEMASIADO AGRESIVO** |
| **Early stopping Stage 2** | **patience=20** | **patience=10** | ❌ **DEMASIADO AGRESIVO** |

**Impacto**: Entrenamiento insuficiente, convergencia prematura

#### 2. **RESOLUCIÓN INCORRECTA**
| Parámetro | Sagrado | Actual | Estado |
|-----------|---------|---------|---------|
| **imgsz** | **896 o 1024** | **[1440, 808]** | ❌ **DESVIACIÓN TOTAL** |
| **Objetivo** | **Small objects** | **High-res custom** | ⚠️ **NECESITA JUSTIFICACIÓN** |

**Impacto**: No alineado con plan para objetos pequeños

#### 3. **BATCH SIZE HARDWARE-SPECIFIC**
| Hardware | Sagrado (2×A100) | Actual (H200) | Batch Efectivo |
|----------|------------------|---------------|----------------|
| **Target batch** | **128 global** | **96 single** | ❌ **25% MENOR** |
| **GPU distribution** | **2×64 o 2×32** | **1×96** | ⚠️ **ARQUITECTURA DIFERENTE** |

**Impacto**: Posible subentrenamiento por batch insuficiente

#### 4. **AUGMENTATION DESALINEADO**
| Parámetro | Sagrado Stage 1 | Actual | Estado |
|-----------|-----------------|---------|---------|
| **mosaic** | **1.0** | **1.0** | ✅ OK |
| **mixup** | **0.1-0.2** | **0.15** | ✅ OK |
| **copy_paste** | **0.0-0.1** | **0.1** | ✅ OK |

---

### ✅ ALINEACIONES CORRECTAS

#### 1. **OPTIMIZER Y LEARNING RATE**
- ✅ `optimizer=SGD` (sagrado)
- ✅ `lr0=0.01` (sagrado SGD)
- ✅ `lrf=0.01` (sagrado)
- ✅ `cos_lr=True` (sagrado)
- ✅ `warmup_epochs=5` (sagrado 5-10)

#### 2. **LOSS WEIGHTS**
- ✅ `box=7.5` (sagrado 7.5-8.0)
- ✅ `cls=0.5` (sagrado 0.5-0.8)
- ✅ `dfl=1.5` (razonable)

#### 3. **DATASET CONFIGURATION**
- ✅ Stage 1: Multi-clase (fire + smoke)
- ✅ Stage 2: Single-clase (smoke only)
- ✅ `single_cls=True` en Stage 2

---

## 🖥️ ANÁLISIS HARDWARE H200 vs 2×A100

### **Hardware Comparison**
| Spec | Sagrado (2×A100) | Actual (H200) | Factor |
|------|------------------|---------------|---------|
| **VRAM** | 2×40GB = 80GB | 140GB | **1.75× MÁS** |
| **Memory BW** | 2×1.6TB/s = 3.2TB/s | 4.8TB/s | **1.5× MÁS** |
| **Compute** | 2×312 TFLOPS = 624 TFLOPS | 989 TFLOPS | **1.58× MÁS** |
| **Architecture** | DDP Multi-GPU | Single GPU | **DIFERENTE** |

### **Configuración Hardware-Adjusted**

#### ✅ **VENTAJAS H200**
- **VRAM superior**: Permite batch más grande → **batch=128** en lugar de 96
- **Memoria unificada**: Sin overhead DDP → **workers más altos**
- **Bandwidth superior**: Mejor para resoluciones altas

#### ⚠️ **LIMITACIONES H200**  
- **Single GPU**: Sin paralelización natural de DDP
- **RAM sistema**: 258GB vs posibles configuraciones 2×A100 con más RAM

---

## 🎯 CONFIGURACIÓN ÓPTIMA PROPUESTA

### **Stage 1: FASDD Pre-training**
```python
config_stage1_optimal = {
    # SACRED PARAMETERS (NO CAMBIAR)
    'epochs': 160,              # Sagrado 140-160 → usar máximo
    'patience': 30,             # Sagrado patience=30 
    'imgsz': 896,              # Sagrado 896 (objetos pequeños)
    'single_cls': False,       # Multi-clase (fire + smoke)
    
    # HARDWARE-OPTIMIZED (H200)
    'batch': 128,              # H200 puede manejar batch sagrado
    'workers': 12,             # H200 RAM superior
    'device': 0,               # Single H200
    
    # OPTIMIZER (SAGRADO)
    'optimizer': 'SGD',
    'lr0': 0.01,
    'lrf': 0.01,
    'cos_lr': True,
    'warmup_epochs': 5,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    
    # LOSS WEIGHTS (SAGRADO)
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    
    # AUGMENTATION (SAGRADO)
    'mosaic': 1.0,
    'mixup': 0.15,             # Centro del rango 0.1-0.2
    'copy_paste': 0.05,        # Centro del rango 0.0-0.1
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    
    # SYSTEM
    'amp': True,
    'cache': 'disk',           # O RAM si /dev/shm disponible
    'save_period': 3,          # Por request usuario
}
```

### **Stage 2: PyroSDIS Fine-tuning**
```python
config_stage2_optimal = {
    # SACRED PARAMETERS
    'epochs': 60,              # Sagrado 40-60 → usar máximo
    'patience': 20,            # Sagrado patience=20
    'imgsz': 896,              # Mantener consistencia
    'single_cls': True,        # Single-clase smoke
    
    # FINE-TUNING (SAGRADO)
    'lr0': 0.001,              # 10× reduced (sagrado)
    
    # REDUCED AUGMENTATION (SAGRADO)
    'mosaic': 0.5,             # Sagrado: bajar en Stage 2
    'mixup': 0.1,              # Limite inferior
    'copy_paste': 0.0,         # Sin copy_paste en fine-tuning
    'hsv_h': 0.015,
    'hsv_s': 0.5,              # Reducido vs Stage 1
    'hsv_v': 0.3,              # Reducido vs Stage 1
}
```

---

## 🚨 ACCIONES CORRECTIVAS INMEDIATAS

### **1. CRÍTICO: Actualizar épocas y patience**
```bash
# Actualizar train_h200.py y train_h200_shm.py
sed -i 's/epochs=110/epochs=160/g' scripts/train_h200.py
sed -i 's/patience.*10/patience=30/g' scripts/train_h200.py
```

### **2. CRÍTICO: Cambiar resolución a sagrada**
```python
# En get_h200_config():
'imgsz': 896,  # Era [1440, 808] → cambiar a sagrado
```

### **3. CRÍTICO: Ajustar batch para H200**
```python
# Aprovechar VRAM superior H200
'batch': 128,  # Era 96 → subir a sagrado
```

### **4. VALIDAR: Configuración /dev/shm**
- ✅ Mantener cache RAM si disponible
- ⚠️ Verificar que `cache: false` en config sea correcto

---

## 📊 JUSTIFICACIÓN TÉCNICA DESVIACIONES ACTUALES

### **¿Por qué 1440×808 en lugar de 896?**
- **No documentado** en sagrado → necesita justificación
- **Posible razón**: Mejor para detección long-range
- **Riesgo**: No optimizado para small objects (objetivo sagrado)

### **¿Por qué patience=10 en lugar de 30?**
- **Posible razón**: Entrenamiento más rápido  
- **Riesgo**: Convergencia prematura, subentrenamiento

### **¿Por qué batch=96 en lugar de 128?**
- **Posible razón**: Conservativo para VRAM
- **Realidad**: H200 140GB puede manejar batch=128 fácilmente

---

## 🎯 RECOMENDACIÓN FINAL

**CONFIGURACIÓN ÓPTIMA ULTRA-REVISADA:**

### **Opción A: SAGRADA ESTRICTA**
- ✅ epochs=160, patience=30, imgsz=896, batch=128
- ✅ Máxima fidelidad a documentación sagrada
- ✅ Resultados predecibles según plan

### **Opción B: SAGRADA + OPTIMIZACIONES H200**  
- ✅ Sagrada + batch=128, workers=12
- ✅ Aprovechar ventajas hardware H200
- ✅ Mantener core sagrado intacto

### **Opción C: HÍBRIDA JUSTIFICADA**
- ⚠️ Mantener 1440×808 SI se justifica para use case
- ✅ Resto estrictamente sagrado
- ⚠️ Requiere validación experimental

**RECOMENDACIÓN**: **Opción B** - Sagrada + H200 optimizations

---

## ✅ CONFIGURACIÓN FINAL RECOMENDADA

La configuración debe seguir la documentación sagrada con adaptaciones mínimas para H200:

- **epochs**: 160 (Stage 1), 60 (Stage 2)  
- **patience**: 30 (Stage 1), 20 (Stage 2)
- **imgsz**: 896 (sagrado para small objects)
- **batch**: 128 (aprovechar H200)
- **workers**: 12 (H200 superior)
- **Resto**: Exactamente según documentación sagrada

**Esta configuración garantiza máxima fidelidad al plan sagrado mientras aprovecha las ventajas del hardware H200.**