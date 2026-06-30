export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_NOISE=20    # estadísticas [NOISE]/[GRAD]/[FLOATER]/[DEADGATE] cada 20 aplicaciones de ruido
export DEBUG_MEM=1000    # pico de memoria + dev_free cada 1000 iters (delata zombies/OOM)

# ───────────────────────────────────────────────────────────────────────────
# ÚNICO bloque a editar entre runs. Todo lo demás (source, model, log) se deriva.
DATASET=bonsai           # nombre de la carpeta en Datasets/ (flowers, bonsai, garden…)
RUN=44                    # número de run → output/m360/${DATASET}_beta_run${RUN}
DEAD_SUSTAIN=5           # base run16 (óptimo del gate; >5 = más niebla, monótono)
CAP_MAX=1500000          # 1.5M = cap OFICIAL de bonsai (= run43)
OPACITY_REG=0.01         # = base run16 (AISLA si el 0.06 de run43 estorba en bonsai; flowers no tiene fondo rasante)
SCALE_REG=0.01           # base run16

MODEL=output/m360/${DATASET}_beta_run${RUN}
LOG=logs/${DATASET}${RUN}.log
# ───────────────────────────────────────────────────────────────────────────

# run44: AÍSLA opacity_reg en BONSAI — réplica EXACTA de run43 salvo opacity_reg 0.06 → 0.01 (base run16).
#
# CONTEXTO (medido honesto 2026-06-30, mismo metrics.py, 37 vistas, iter 30000):
#   original 2DGS (798K splats) ... 31.36 / 0.9359 / 0.2042   ← baseline honesto de bonsai
#   run42 (cap 4.5M, op_reg 0.06) . 27.54 / 0.9120 / 0.2131   ← sobre-densificó (4.5M topado), perdió en las 3
#   run43 (cap 1.5M, op_reg 0.06) . 28.38 / 0.9151 / 0.2191   ← bajar cap: PSNR +0.84, SSIM +0.0031, LPIPS PEOR +0.006
# run43 mejoró PSNR/SSIM (confirma que la sobre-densificación dañaba) pero el LPIPS EMPEORÓ y sigue −2.98 dB
# bajo el original. Sospecha: opacity_reg=0.06 ESTORBA en bonsai. En flowers ese dial subía PSNR adelgazando
# el VELO DEL FONDO RASANTE; bonsai es INTERIOR ACOTADA y no tiene fondo rasante → la presión L1 fuerte solo
# quita opacidad/detalle útil (coherente con el LPIPS = peor de los tres en run43; en flowers LPIPS empeoraba
# monótono con opacity_reg).
#
# PREGUNTA que responde: ¿opacity_reg 0.06 ayuda o estorba en bonsai? Hipótesis: estorba → volver a 0.01
# sube las tres métricas (sobre todo LPIPS).
#
# ÚNICO CAMBIO vs run43: opacity_reg 0.06 → 0.01. (cap sigue 1.5M, DATASET sigue bonsai.)
#
# OJOS EN ESTE RUN:
#   - lambda_dist=10: en bonsai (acotada) la distorsión SÍ es activa → vigilar `distort` en el log.
#   - cap 1.5M sobre escena acotada: ¿topa exacto (como run43) o sobra presupuesto? Vigilar N final.
#   - comparar run44 vs run43 AÍSLA opacity_reg (todo lo demás idéntico).
#
# NO requiere recompilar el rasterizer (solo λ de Python; --freeze_low_beta OFF = default run9/run16).
#
# Resto = run43/run16: lambda_dist 10, dead_sustain 5, scale_reg 0.01, ruido run9
# (--cov_noise --cov_noise_normal 1.0 --noise_lr 3e3), reset OFF, lambda_normal=0.05, opacity_cull=0.01,
# floater_cull_dist=0.2, mcmc_error_weight=3.5, mcmc_jitter_scale=1.5, iters=30000, densify_until=25000.
#
# Tras el run: render_server.sh (DATASET=bonsai, RUN=44) + metrics.py. Baseline honesto de bonsai YA medido
# (2026-06-30): original 2DGS = 31.36/0.9359/0.2042 (mismo metrics.py, 37 vistas).
# Añadir fila al historial + docs/comparativa_runs.html (nota: el historial es de flowers; marcar que es bonsai).
python train.py -s Datasets/${DATASET} \
    -m $MODEL \
    --eval \
    --densify_mode mcmc \
    --iterations 30000 \
    --test_iterations 7000 15000 20000 25000 30000 \
    --densify_until_iter 25000 \
    --lambda_normal 0.05 \
    --lambda_dist 10 \
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
