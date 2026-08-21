#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ═══════════════════════════════════════════════════════════════════════════
#  RENDER + METRICS EN BARRIDO — pareja de train_CLASSIC_sweep_server.sh
# ═══════════════════════════════════════════════════════════════════════════
# Hace, para CADA escena del barrido, los mismos 3 pasos que render_server.sh
# hacía para una sola:
#   1) vídeo de trayectoria  (--render_path)         -> $MODEL/traj/ours_$ITER/
#   2) comparativas render|GT de las vistas de test  -> $MODEL/test/ours_$ITER/
#   3) metrics.py (SSIM/PSNR/LPIPS honesto)          -> $MODEL/results.json
# El paso 3 es OBLIGATORIO en este proyecto: sin él results.json sale vacío.
#
# DATASETS y RUNS deben coincidir EXACTAMENTE con train_CLASSIC_sweep_server.sh
# (mismo orden), porque de ahí se deriva la ruta del modelo a renderizar.
# ═══════════════════════════════════════════════════════════════════════════

mkdir -p logs
MASTER_LOG="logs/master_render_progress.log"

echo "==================================================" >> "$MASTER_LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🎬 INICIO DEL BATCH DE RENDER+METRICS" | tee -a "$MASTER_LOG"
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

ITER=30000        # iteración (checkpoint) a renderizar
RENDER_TRAJ=1     # 1 = genera también el vídeo de trayectoria (paso 1).
                  # Es el paso MÁS CARO y multiplica el tiempo del barrido ×N escenas:
                  # ponlo a 0 si solo quieres las métricas honestas y los render|GT.
RUN_METRICS=1     # 1 = corre metrics.py por escena (workflow obligatorio del proyecto)

OK_LIST=()
FAIL_LIST=()
SKIP_LIST=()

for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    RUN="${RUNS[$i]}"
    MODEL="output/m360/${DATASET}_beta_run${RUN}"
    SOURCE="Datasets/${DATASET}"
    LOG_BASE="logs/${DATASET}_run${RUN}"

    mkdir -p "$(dirname "$LOG_BASE")"

    # --- Guard: no abortar el batch entero si a una escena le falta el modelo ---
    PLY="${MODEL}/point_cloud/iteration_${ITER}/point_cloud.ply"
    if [ ! -f "$PLY" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  SALTANDO: $DATASET (Run $RUN) — no existe $PLY" | tee -a "$MASTER_LOG"
        SKIP_LIST+=("${DATASET} (run${RUN})")
        continue
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⏳ RENDERIZANDO: $DATASET (Run $RUN)..." | tee -a "$MASTER_LOG"
    SCENE_OK=1

    # 1) Vídeo de trayectoria (vistas nuevas interpoladas).
    #    --skip_train --skip_test --skip_mesh + --render_path => SOLO el vídeo.
    #    Salida: $MODEL/traj/ours_${ITER}/render_traj_color.mp4
    if [ "$RENDER_TRAJ" = "1" ]; then
        python render.py -s "$SOURCE" \
            -m "$MODEL" \
            --iteration $ITER \
            --skip_train --skip_test --skip_mesh \
            --render_path \
            2>&1 | tee "${LOG_BASE}_render.log"
        [ "${PIPESTATUS[0]}" -ne 0 ] && SCENE_OK=0
    fi

    # 2) Comparativas render|GT de las vistas de test.
    #    --skip_train --skip_mesh => exporta SOLO test (sin vídeo ni malla).
    #    Salida: $MODEL/test/ours_${ITER}/{renders,gt,vis}
    #    Este paso es el que ALIMENTA a metrics.py -> no se puede saltar.
    python render.py -s "$SOURCE" \
        -m "$MODEL" \
        --iteration $ITER \
        --skip_train --skip_mesh \
        2>&1 | tee "${LOG_BASE}_test.log"
    [ "${PIPESTATUS[0]}" -ne 0 ] && SCENE_OK=0

    # 3) Métricas honestas (SSIM/PSNR/LPIPS con lpipsPyTorch). SIN esto results.json
    #    sale vacío y el run no queda medido.
    if [ "$RUN_METRICS" = "1" ] && [ "$SCENE_OK" = "1" ]; then
        python metrics.py -m "$MODEL" 2>&1 | tee "${LOG_BASE}_metrics.log"
        [ "${PIPESTATUS[0]}" -ne 0 ] && SCENE_OK=0
    fi

    if [ "$SCENE_OK" = "1" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ TERMINADO: $DATASET (Run $RUN)" | tee -a "$MASTER_LOG"
        OK_LIST+=("${DATASET} (run${RUN})")
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ FALLÓ: $DATASET (Run $RUN) — revisa ${LOG_BASE}_*.log" | tee -a "$MASTER_LOG"
        FAIL_LIST+=("${DATASET} (run${RUN})")
    fi
    echo "--------------------------------------------------" >> "$MASTER_LOG"
done

# ═══ RESUMEN FINAL ═══════════════════════════════════════════════════════════
echo "==================================================" | tee -a "$MASTER_LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🎉 BATCH DE RENDER COMPLETADO" | tee -a "$MASTER_LOG"
echo "   OK: ${#OK_LIST[@]}  ·  FALLOS: ${#FAIL_LIST[@]}  ·  SALTADAS: ${#SKIP_LIST[@]}" | tee -a "$MASTER_LOG"
[ ${#FAIL_LIST[@]} -gt 0 ] && printf '   ❌ %s\n' "${FAIL_LIST[@]}" | tee -a "$MASTER_LOG"
[ ${#SKIP_LIST[@]} -gt 0 ] && printf '   ⚠️  %s\n' "${SKIP_LIST[@]}" | tee -a "$MASTER_LOG"

# Tabla de métricas honestas de todo el barrido, leída de los results.json.
# Es lo que hay que copiar al historial y a docs/comparativa_runs.html.
echo "" | tee -a "$MASTER_LOG"
echo "═══ MÉTRICAS HONESTAS DEL BARRIDO (metrics.py) ═══" | tee -a "$MASTER_LOG"
python - "${DATASETS[@]}" <<'PY' 2>&1 | tee -a "$MASTER_LOG"
import json, os, sys
runs = [81,82,83,84,85,86,87,88,89,90]
print(f"{'escena':<24}{'run':>5}{'PSNR':>10}{'SSIM':>10}{'LPIPS':>10}{'N splats':>12}")
print("-"*71)
for ds, run in zip(sys.argv[1:], runs):
    model = f"output/m360/{ds}_beta_run{run}"
    rj = os.path.join(model, "results.json")
    if not os.path.isfile(rj):
        print(f"{ds:<24}{run:>5}{'— sin results.json —':>32}")
        continue
    try:
        d = json.load(open(rj))
    except Exception as e:
        print(f"{ds:<24}{run:>5}   ERROR leyendo results.json: {e}")
        continue
    for method, m in d.items():
        # nº de splats: se saca del header del ply (no hace falta cargarlo entero)
        n = ""
        ply = os.path.join(model, "point_cloud", "iteration_30000", "point_cloud.ply")
        if os.path.isfile(ply):
            with open(ply, "rb") as f:
                for line in f:
                    if line.startswith(b"element vertex"):
                        n = f"{int(line.split()[2]):,}"; break
                    if line.startswith(b"end_header"): break
        print(f"{ds:<24}{run:>5}{m.get('PSNR',0):>10.4f}{m.get('SSIM',0):>10.4f}"
              f"{m.get('LPIPS',0):>10.4f}{n:>12}")
PY

# ─── QUÉ HACER DESPUÉS ───────────────────────────────────────────────────────
# 1. Copiar la tabla de arriba al historial + docs/comparativa_runs.html.
# 2. Comparar HONESTO-vs-HONESTO contra el 2DGS oficial de cada escena
#    (mismo metrics.py). Baselines ya medidos: flowers 20.89/0.556/0.402 ·
#    bicycle 24.6091/0.7134/0.3064 · kitchen 30.3389/0.9210/0.1383 ·
#    bonsai 31.36/0.9359/0.2042. Las demás escenas NO tienen baseline propio aún.
# 3. Verificar en los logs de train que los fixes estuvieron activos:
#    grep "DENSIFY-OPA" · grep "err +0.00%" · grep "d_beta_max" · grep "PRUNE-WORLD"
