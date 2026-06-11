export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_NOISE=20    # estadísticas [NOISE]/[GRAD] cada 20 aplicaciones de ruido (verificar |disp|~0.05)
export DEBUG_MEM=1000    # pico de memoria + dev_free cada 1000 iters (delata zombies/OOM)

# ───────────────────────────────────────────────────────────────────────────
# ÚNICO bloque a editar entre runs. Todo lo demás (source, model, log) se deriva.
DATASET=flowers          # nombre de la carpeta en Datasets/ (flowers, bonsai, garden…)
RUN=13                    # número de run → output/m360/${DATASET}_beta_run${RUN}
OPACITY_REG=0.05         # barrido L1 opacidad (default 0.01; 0.02 contaminado por reset en run11/12 → 0.05)
# NOTA: runs 11/12 (reset=3000) COLAPSARON → PSNR ~12, LPIPS 0.53-0.67 = run10 redux.
# opacity_reset INCOMPATIBLE con MCMC (resets 3k-30k caen en ventana de ruido →
# terremoto + nube translúcida). reset DESACTIVADO (1e9). Vía limpia = opacity_reg.

MODEL=output/m360/${DATASET}_beta_run${RUN}
LOG=logs/${DATASET}${RUN}.log
# ───────────────────────────────────────────────────────────────────────────

# run13: SH3 (sin Spherical Betas), reset OFF (1e9). Barrido opacity_reg = 0.05
# (vía MCMC-nativa de limpiar el fondo translúcido: L1 continua sobre opacidad →
# relocate recicla los splats de baja opacidad). Config ALINEADA con el ciclo del
# oficial: iterations 50k→30k, densify_until 35k→25k → la fase MCMC (densify+
# ruido+regs) cierra en 25k y los últimos 5k son consolidación fotométrica pura.
# Métrica a vigilar: LPIPS+SSIM de test (no solo PSNR) — correr metrics.py tras
# render. Éxito = LPIPS baja vs run9 aunque el PSNR no suba.
python train.py -s Datasets/${DATASET} \
    -m $MODEL \
    --eval \
    --iterations 30000 \
    --test_iterations 7000 15000 20000 25000 30000 \
    --densify_until_iter 25000 \
    --lambda_normal 0.05 \
    --lambda_dist 10 \
    --opacity_reset_interval 1000000000 \
    --cap_max 7500000 \
    --noise_lr 3e3 \
    --scale_reg 0.01 \
    --opacity_reg $OPACITY_REG \
    --opacity_cull 0.01 \
    --floater_cull_dist 0.2 \
    --mcmc_error_weight 3.5 \
    --mcmc_jitter_scale 1.5 \
    --cov_noise \
    --cov_noise_normal 1.0 \
    2>&1 | tee $LOG
