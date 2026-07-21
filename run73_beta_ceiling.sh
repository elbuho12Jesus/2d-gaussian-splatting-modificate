#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run73 — SUBIR EL TECHO DE β  (29.556 → 60), réplica EXACTA de run67.
#
# BASE = run67 (mejor clásico del historial: 20.6684 / 0.5811 / 0.3675, train@30k
# 23.63, 4.875M splats) = ancla run26 + FIX DEL TRINQUETE DE β. Config clásica:
# clone/split + opacity_reset 3000, opacity_reg=0, scale_reg=0, prune_sustain=25,
# opacity_cull=0.005, densify 500→15000, lambda_dist=0, SCALE_CLAMP_FACTOR=0.1,
# CULL_SUBPIXEL=1 (compilado en el .so del servidor, común con run67).
#
# ═══ EL ÚNICO DELTA vs run67 = el TECHO de β ═══
# El clamp superior de _beta pasa de 2.0 a 2.7081 en LOS DOS sitios (a la vez, o el
# forward recorta y el cambio es un NO-OP):
#   · train.py:399                 _beta.data.clamp_(min=-4.0, max=2.7081)
#   · scene/gaussian_model.py:184  get_beta -> _beta.clamp(min=-4.0, max=2.7081)
# beta = 4·exp(_beta)  ⇒  techo pasa de 4·e² = 29.556  a  4·e^2.7081 = 60.
# Además se IGUALÓ el min a -4.0 en ambos (get_beta tenía -6.0, holgura muerta:
# train.py ya proyecta _beta a [-4,2] cada iteración → nunca baja de -4).
# NO requiere recompilar el rasterizer (el cambio es 100% Python; la beta activada
# se pasa al .so como tensor).
#
# ═══ QUÉ SE ESPERA (esto es un TEST DE FALSACIÓN, no una búsqueda de mejora) ═══
# El análisis docs/techo_beta_clamp_superior.html PREDICE que subir el techo es un
# NO-OP: en run67/run72 solo 1–9 splats de ~4,8M tocaban el techo viejo (0.0002%),
# el p99.9 estaba en β≈11 (37% del tope) y ∂G/∂β ≤ 0 (subir β solo ENCOGE el kernel,
# y para eso el modelo tiene la escala, sin tope). Predicción: métrica ≈ run67
# dentro del ruido (±0.1–0.2 dB) y CASI NADIE se mueve por encima de 29.556.
# Este run FALSA o CONFIRMA esa teoría de forma directa.
#
# ═══ CÓMO LEERLO — el print [BETA-TECHO] (añadido en train.py) es la clave ═══
#   · topados %        : cuántos _beta llegan al NUEVO techo 2.7081 (β≈60). Si ≈0.00%,
#                        el techo nuevo TAMPOCO muerde → teoría CONFIRMADA.
#   · beta>29.556 %    : cuántos superan el techo VIEJO (los que antes estaban
#                        artificialmente recortados). Si es ínfimo, subir el techo no
#                        liberó población real = no cambia nada, como predice el doc.
#   · si la métrica @30k ≈ run67 (20.67) → NO-OP confirmado, cerrar la palanca.
#   · si SUBE de forma clara → la teoría del histograma estaba incompleta (sorpresa;
#     habría que entender qué población nueva se activó).
#
# EJECUTAR EN EL SERVIDOR, dentro del contenedor:
#   docker compose exec surfel_env bash run73_beta_ceiling.sh
# (metrics.py corre inline; si falla en el server, re-lanzar en local con los plys
#  sincronizados: python metrics.py -m output/m360/flowers_beta_run73)
#
# TIEMPO/DISCO: ≈1–2 h en la Blackwell + ~5 GB de salida (como run67).
# ─────────────────────────────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_MEM=1000    # pico de memoria + dev_free (delata zombies/OOM)
export DEBUG_DENSIFY=1   # [DENSIFY]/[RESET] — clave en clásico

# run65 se tituló "small clamp" pero NUNCA exportó esta env var → corrió con el
# default sin avisar. Explícita aquí + el print [CLAMP] evita repetir el fallo.
export SCALE_CLAMP_FACTOR=0.1  # 0.1 = default 3DGS/2DGS (= run67)

# ───────────────────────── BLOQUE EDITABLE ──────────────────────────────────
DATASET=flowers
RUN=73

# FIJOS = run67 EXACTO. NADA de esto se mueve: el único delta es el techo de β,
# que vive en train.py:399 y gaussian_model.py:184 (no es un flag CLI).
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

MODEL=output/m360/${DATASET}_beta_run${RUN}
LOG=logs/${DATASET}${RUN}.log
# ─────────────────────────────────────────────────────────────────────────────

echo "════════════════════════════════════════════════════════════════════"
echo " run73 = run67 EXACTO + TECHO de β 29.556 → 60 (único delta)"
echo " ancla run67: 20.6684 / 0.5811 / 0.3675 · train@30k 23.63 · 4.875M"
echo " predicción (docs/techo_beta_clamp_superior.html): NO-OP (≈ run67)"
echo " vigilar el print [BETA-TECHO]: topados% en el techo nuevo y beta>29.556"
echo "════════════════════════════════════════════════════════════════════"

# ── 1) TRAIN (config = run67 EXACTO; el techo de β está en el código) ──
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

# ── 2) RENDER (solo test: lo que metrics.py necesita; sin vídeo ni malla) ──
python render.py -s Datasets/${DATASET} \
    -m $MODEL \
    --iteration 30000 \
    --skip_train --skip_mesh \
    2>&1 | tee logs/${DATASET}${RUN}_test.log

# ── 3) METRICS (honesto, genera results.json) ──
python metrics.py -m $MODEL 2>&1 | tee logs/${DATASET}${RUN}_metrics.log

echo ""
echo "── run73 hecho. Comparar con run67 (20.6684/0.5811/0.3675) ──"
echo "   · si ≈ igual y [BETA-TECHO] topados≈0.00% → NO-OP confirmado (teoría OK)"
echo "   · volcar a historial_runs.csv + docs/comparativa_runs.html + CLAUDE.md"
