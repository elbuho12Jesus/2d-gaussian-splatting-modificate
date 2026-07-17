export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_MEM=1000    # pico de memoria + dev_free cada 1000 iters (delata zombies/OOM)
export DEBUG_DENSIFY=1   # imprime [DENSIFY]/[RESET] (default ON; =0 silencia)
# (DEBUG_NOISE NO aplica: el ruido posicional MCMC está OFF en el camino clásico)

# ═══════════════════════════════════════════════════════════════════════════
#  TRAIN — DENSIFICACIÓN CLÁSICA 2DGS (clone/split + prune + opacity_reset)
# ═══════════════════════════════════════════════════════════════════════════
# Camino clásico: --densify_mode classic. densify_and_prune dirigido por
# ‖∂L/∂μ₂D‖ (clone bajo under-recon + split bajo over-recon) + opacity_reset
# periódico. El ruido posicional MCMC, el cull de floaters y mcmc_* NO se aplican
# (son del camino MCMC, el clásico los ignora). cap_max tampoco aplica: el clásico
# crece libre → VIGILAR nº de splats por OOM. Para el MCMC → train_MCMC_server.sh.
#
# OJO opacity_reset: en clásico es SEGURO (3000 = ciclo nativo reset→recupera/prune;
# el colapso de runs 10/11/12 era la compuerta (1−o)^100 del MCMC, OFF aquí). En MCMC
# rompe. Palanca propia del clásico: classic_prune_sustain (prune sostenido N pasos).
#
# DEFAULTS = run15 (mejor clásico honesto: 20.13 / 0.548 / 0.387; prune inmediato +
# FIX load_ply beta). Modos de error opuestos al MCMC: el clásico hace HUECOS NEGROS
# (sub-cobertura del fondo rasante), el MCMC hace VELO translúcido. Tras el run:
# render_server.sh + metrics.py + fila al historial y a docs/comparativa_runs.html.
# ───────────────────────────────────────────────────────────────────────────
# ÚNICO bloque a editar entre runs. Todo lo demás (source, model, log) se deriva.
DATASET=flowers               # carpeta en Datasets/ (flowers, bonsai, garden…)
RUN=65                        # nº de run → output/m360/${DATASET}_beta_run${RUN}

PRUNE_SUSTAIN=5               # prune sostenido (análogo del gate dead_sustain=5 de run64; 0=inmediato run15)
OPACITY_RESET_INTERVAL=3000   # ciclo reset→recupera/prune (SEGURO en clásico; del 2DGS original)
DENSIFY_FROM=500              # inicio de densificación
DENSIFY_UNTIL=15000           # fin de densificación (ventana del 2DGS original)
DENSIFICATION_INTERVAL=100    # cada cuántas iters se densifica/poda
DENSIFY_GRAD_THRESHOLD=0.0002 # umbral de ‖∂L/∂μ₂D‖ para clone/split
PERCENT_DENSE=0.01            # umbral de tamaño clone vs split
OPACITY_CULL=0.01             # min_opacity del prune (0.005=2DGS original; 0.01=poda más agresiva, run65)
LAMBDA_DIST=0                 # reg distorsión (0 = receta 2DGS original; el 10 era nuestro)
LAMBDA_NORMAL=0.05            # reg de consistencia de normales
OPACITY_REG=0.06              # reg L1 opacidad = run64 (⚠ óptimo de flowers-MCMC; en clásico puede vaciar fondo)
SCALE_REG=0.06                # reg L1 escala = run64 (⚠ en clásico EMPEORA el fondo históricamente, ver run21)
ITERATIONS=30000

MODEL=output/m360/${DATASET}_beta_run${RUN}
LOG=logs/${DATASET}${RUN}.log
# ───────────────────────────────────────────────────────────────────────────

python train.py -s Datasets/${DATASET} \
    -m $MODEL \
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
    2>&1 | tee $LOG
