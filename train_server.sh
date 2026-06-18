export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_NOISE=20    # estadísticas [NOISE]/[GRAD]/[FLOATER]/[DEADGATE] cada 20 aplicaciones de ruido
export DEBUG_MEM=1000    # pico de memoria + dev_free cada 1000 iters (delata zombies/OOM)

# ───────────────────────────────────────────────────────────────────────────
# ÚNICO bloque a editar entre runs. Todo lo demás (source, model, log) se deriva.
DATASET=flowers          # nombre de la carpeta en Datasets/ (flowers, bonsai, garden…)
RUN=21                    # número de run → output/m360/${DATASET}_beta_run${RUN}
DEAD_SUSTAIN=5           # gate MCMC de vuelta a 5 (run19 con 25 ALIMENTÓ la niebla)
CAP_MAX=7500000          # 7.5M = base run16 (el mejor MCMC honesto; menor niebla de los MCMC)
OPACITY_REG=0.01         # REVERTIDO a base run16 (run20: 0.02 empeoró −0.67 PSNR y NO limpió)
SCALE_REG=0.05           # ÚNICA variable nueva: 0.01→0.05 (L1 contra surfels gigantes del velo)

MODEL=output/m360/${DATASET}_beta_run${RUN}
LOG=logs/${DATASET}${RUN}.log
# ───────────────────────────────────────────────────────────────────────────

# run21: atacar la NIEBLA por el TAMAÑO (no por opacidad). Base = run16 EXACTA (el mejor
# MCMC honesto: PSNR 20.21 / SSIM 0.597 / LPIPS 0.339), cambiando UNA sola variable:
# scale_reg 0.01 → 0.05. opacity_reg REVERTIDO a 0.01 (run20 = 0.02 empeoró −0.67 PSNR y
# no limpió → no apilar un cambio malo). SIN tocar la lógica del relocate (decisión usuario).
#
# LÓGICA: las "nubes blancas" son floaters sobre-brillantes; el log de ruido revela
# s(max=4.816)=EXTENT → hay surfels GIGANTES que abarcan toda la escena = candidatos al velo.
# scale_reg es L1 sobre get_scaling → castiga proporcionalmente MÁS a esos gigantones →
# palanca directa contra el velo "grande" (run20 confirmó que la opacidad NO lo limpia).
#
# AVISO IMPORTANTE: scale_reg NO distingue un surfel gigante MALO (floater blanco delante)
# de uno BUENO (el surfel grande/alargado/translúcido que CUBRE el fondo rasante). Es un
# instrumento contundente → puede reducir las nubes A COSTA de re-abrir huecos / oscurecer
# el fondo (mismo modo de fallo que opacity_reg; = lo que hizo el clásico run14 con fondo
# negro). VIGILAR render mean vs gt: si cae por debajo del GT, nos pasamos (probar 0.03).
#
# Resto = run16: ruido run9 (--cov_noise --cov_noise_normal 1.0 --noise_lr 3e3), reset OFF,
# lambda_dist=10, lambda_normal=0.05, opacity_reg=0.01, opacity_cull=0.01, floater_cull_dist=0.2,
# mcmc_error_weight=3.5, mcmc_jitter_scale=1.5, iterations=30000, densify_until_iter=25000.
#
# En el log vigilar: ¿baja el exceso-brillo/niebla per-vista? ¿se vacía el fondo
# (render mean << gt)? ¿baja s(max)? nº de splats. NO recompila CUDA. Tras el run:
# render_server.sh (RUN=21) + metrics.py. Comparar vs run16 (0.339/0.597/20.21) y run20.
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
