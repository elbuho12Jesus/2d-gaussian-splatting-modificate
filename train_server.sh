export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_NOISE=20    # estadísticas [NOISE]/[GRAD]/[FLOATER]/[DEADGATE] cada 20 aplicaciones de ruido
export DEBUG_MEM=1000    # pico de memoria + dev_free cada 1000 iters (delata zombies/OOM)

# ───────────────────────────────────────────────────────────────────────────
# ÚNICO bloque a editar entre runs. Todo lo demás (source, model, log) se deriva.
DATASET=flowers          # nombre de la carpeta en Datasets/ (flowers, bonsai, garden…)
RUN=30                    # número de run → output/m360/${DATASET}_beta_run${RUN}
DEAD_SUSTAIN=5           # base run16 (óptimo del gate; >5 = más niebla, monótono)
CAP_MAX=7500000          # 7.5M = base run16 (cap saturado; mantenido por decisión del usuario)
OPACITY_REG=0.01         # base run16
SCALE_REG=0.01           # base run16

MODEL=output/m360/${DATASET}_beta_run${RUN}
LOG=logs/${DATASET}${RUN}.log
# ───────────────────────────────────────────────────────────────────────────

# run30: A/B DEL FIX #2 DEL BACKWARD (run9). Réplica EXACTA de run16 (mejor MCMC) + --freeze_low_beta.
# Objetivo: aislar por fin cuánto aporta el FIX #2 (descongelar geometría/opacidad de los splats
# con beta<0.1). run9 metió los 3 fixes del backward JUNTOS y medidos in-train → "mejora ligera"
# combinada, nunca aislada ni honesta. Este es el 1er A/B aislado: --freeze_low_beta ON reproduce
# BIT-EXACTO el comportamiento PRE-run9 (esos splats CONGELADOS), dejando FIX #1 y #3 intactos.
# Único cambio vs run16: + --freeze_low_beta. Todo lo demás idéntico a run16 (la base): lambda_dist
# 10 (¡ojo, run28 lo bajó a 0; aquí VUELVE a 10!), dead_sustain 5, scale_reg 0.01, opacity_reg 0.01,
# cap 7.5M, ruido run9.
#
# REQUIERE RECOMPILAR EL RASTERIZER en el servidor (cambió la ABI del backward):
#   docker compose exec surfel_env pip install --force-reinstall --no-deps /workspace/submodules/diff-surfel-rasterization
#
# HIPÓTESIS: run30 = comportamiento viejo (congelado), run16 = nuevo (run9). Si run30 ≈ run16 →
# el FIX #2 es casi un no-op (pocos splats con beta<0.1) y la "mejora de run9" venía de FIX #1/#3.
# Si run30 PEOR que run16 → descongelar SÍ ayudó (FIX #2 justificado). Si run30 MEJOR → sorpresa:
# congelar low-beta sería preferible (el fix habría sido contraproducente).
#
# Resto = run16: ruido run9 (--cov_noise --cov_noise_normal 1.0 --noise_lr 3e3), reset OFF,
# lambda_normal=0.05, opacity_cull=0.01, floater_cull_dist=0.2, mcmc_error_weight=3.5,
# mcmc_jitter_scale=1.5, iterations=30000, densify_until_iter=25000.
#
# En el log vigilar: distort debe volver a ≈0.0016 (lambda_dist=10 otra vez); ¿cuántos splats
# con beta<0.1 hay? (si son ~0, el A/B saldrá plano). Tras el run: render_server.sh (RUN=30) +
# metrics.py. Comparar HONESTO vs run16 (0.339/0.597/20.21).
python train.py -s Datasets/${DATASET} \
    -m $MODEL \
    --eval \
    --densify_mode mcmc \
    --iterations 30000 \
    --test_iterations 7000 15000 20000 25000 30000 \
    --densify_until_iter 25000 \
    --lambda_normal 0.05 \
    --lambda_dist 10 \
    --freeze_low_beta \
    --opacity_reset_interval 1000000000 \
    --cap_max $CAP_MAX \
    --noise_lr 3e3 \
    --scale_reg $SCALE_REG \
    --opacity_reg $OPACITY_REG \
    --opacity_cull 0.01 \
    --floater_cull_dist 0.2 \
    --mcmc_error_weight 3.5 \
    --mcmc_jitter_scale 1.5 \
    --cov_noise \
    --cov_noise_normal 1.0 \
    --mcmc_dead_sustain $DEAD_SUSTAIN \
    2>&1 | tee $LOG
