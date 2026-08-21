#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_MEM=1000    
export DEBUG_DENSIFY=1   

# Directorio principal para logs
mkdir -p logs
MASTER_LOG="logs/master_progress.log"

# Registrar inicio del bloque completo
echo "==================================================" >> "$MASTER_LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🚀 INICIO DEL BATCH COMPLETO" | tee -a "$MASTER_LOG"
echo "==================================================" >> "$MASTER_LOG"

DATASETS=(
    "mipnerf360/bicycle" 
    "mipnerf360/bonsai" 
    "mipnerf360/counter" 
    "mipnerf360/flowers" 
    "mipnerf360/garden" 
    "mipnerf360/kitchen" 
    "mipnerf360/room" 
    "mipnerf360/stump" 
    "mipnerf360/treehill"
    "tandt/truck"
    "tandt/train"
)

RUNS=(81 82 83 84 85 86 87 88 89 90 91)

export SCALE_CLAMP_FACTOR=0.1
PRUNE_SUSTAIN=25
OPACITY_RESET_INTERVAL=3000
DENSIFY_FROM=500
DENSIFY_UNTIL=15000
DENSIFICATION_INTERVAL=100
DENSIFY_GRAD_THRESHOLD=0.0002
PERCENT_DENSE=0.01
OPACITY_CULL=0.005
LAMBDA_DIST=0
LAMBDA_NORMAL=0.05
OPACITY_REG=0
SCALE_REG=0
ITERATIONS=30000
DENSIFY_OPACITY_MODE=transmittance
PRUNE_WORLD_RAW="--classic_prune_world_raw"

for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    RUN="${RUNS[$i]}"
    MODEL="output/m360/${DATASET}_beta_run${RUN}"
    LOG="logs/${DATASET}_run${RUN}.log"

    mkdir -p "$(dirname "$LOG")"

    # Escribir en el log maestro antes de empezar
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⏳ INICIANDO: $DATASET (Run $RUN)..." | tee -a "$MASTER_LOG"

    python train.py -s "Datasets/${DATASET}" \
        -m "$MODEL" \
        --eval \
        --densify_mode classic \
        --iterations $ITERATIONS \
        --test_iterations 7000 15000 20000 25000 30000 \
        --densify_from_iter $DENSIFY_FROM \
        --densify_until_iter $DENSIFY_UNTIL \
        --densification_interval $DENSIFICATION_INTERVAL \
        --densify_grad_threshold $DENSIFY_GRAD_THRESHOLD \
        --percent_dense $PERCENT_DENSE \
        --opacity_reset_interval $OPACITY_RESET_INTERVAL \
        --opacity_cull $OPACITY_CULL \
        --lambda_normal $LAMBDA_NORMAL \
        --lambda_dist $LAMBDA_DIST \
        --opacity_reg $OPACITY_REG \
        --scale_reg $SCALE_REG \
        --classic_prune_sustain $PRUNE_SUSTAIN \
        --densify_opacity_mode $DENSIFY_OPACITY_MODE \
        $PRUNE_WORLD_RAW \
        2>&1 | tee "$LOG"

    # Escribir en el log maestro al terminar
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ TERMINADO: $DATASET (Run $RUN)" | tee -a "$MASTER_LOG"
    echo "--------------------------------------------------" >> "$MASTER_LOG"

done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🎉 BATCH COMPLETADO EXITOSAMENTE" | tee -a "$MASTER_LOG"

