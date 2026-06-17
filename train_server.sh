export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_NOISE=20    # estadísticas [NOISE]/[GRAD]/[FLOATER]/[DEADGATE] cada 20 aplicaciones de ruido
export DEBUG_MEM=1000    # pico de memoria + dev_free cada 1000 iters (delata zombies/OOM)

# ───────────────────────────────────────────────────────────────────────────
# ÚNICO bloque a editar entre runs. Todo lo demás (source, model, log) se deriva.
DATASET=flowers          # nombre de la carpeta en Datasets/ (flowers, bonsai, garden…)
RUN=20                    # número de run → output/m360/${DATASET}_beta_run${RUN}
DEAD_SUSTAIN=5           # gate MCMC de vuelta a 5 (run19 con 25 ALIMENTÓ la niebla)
CAP_MAX=7500000          # 7.5M = base run16 (el mejor MCMC honesto; menor niebla de los MCMC)
OPACITY_REG=0.02         # ÚNICA variable nueva: 0.01→0.02 (presión L1 para limpiar la niebla)

MODEL=output/m360/${DATASET}_beta_run${RUN}
LOG=logs/${DATASET}${RUN}.log
# ───────────────────────────────────────────────────────────────────────────

# run20: atacar la NIEBLA por la OPACIDAD (no por el gate). Base = run16 EXACTA (el mejor
# MCMC honesto: PSNR 20.21 / SSIM 0.597 / LPIPS 0.339), cambiando UNA sola variable:
# opacity_reg 0.01 → 0.02. SIN tocar la lógica del relocate (decisión del usuario).
#
# LÓGICA: la niebla SON floaters translúcidos de baja opacidad en el fondo (run19 demostró
# que darles más tiempo —gate 25— los multiplica). opacity_reg es una presión L1 que empuja
# TODAS las opacidades hacia 0 → los floaters caen bajo el cull → el relocate (intacto) se
# los lleva a zonas de alto error. Vía MCMC-nativa de limpiar el fondo. Ver
# docs/regularizacion_opacidad_l1.html. (gate de vuelta a 5: run19 confirmó que 25 = más velo.)
#
# AVISOS: (1) opacity_reg empuja TODA la opacidad → con exceso puede VACIAR el fondo
# (translúcido/oscuro). Por eso 0.02 moderado, no 0.05. Si limpia sin vaciar → probar 0.03.
# (2) El 0.05 de run13 quedó contaminado por el bug de beta (sin lectura limpia); 0.02 aquí
# es la primera medición honesta de la palanca opacity_reg.
#
# Resto = run16: ruido run9 (--cov_noise --cov_noise_normal 1.0 --noise_lr 3e3), reset OFF,
# lambda_dist=10, lambda_normal=0.05, scale_reg=0.01, opacity_cull=0.01, floater_cull_dist=0.2,
# mcmc_error_weight=3.5, mcmc_jitter_scale=1.5, iterations=30000, densify_until_iter=25000.
#
# En el log vigilar: ¿baja el exceso-brillo/niebla en metrics per-vista? ¿se vacía el fondo
# (render mean << gt)? nº de splats. NO recompila CUDA. Tras el run: render_server.sh
# (RUN=20) + metrics.py. Comparar vs run16 (0.339/0.597/20.21, niebla 8.47).
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
    --scale_reg 0.01 \
    --opacity_reg $OPACITY_REG \
    --opacity_cull 0.01 \
    --floater_cull_dist 0.2 \
    --mcmc_error_weight 3.5 \
    --mcmc_jitter_scale 1.5 \
    --cov_noise \
    --cov_noise_normal 1.0 \
    --mcmc_dead_sustain $DEAD_SUSTAIN \
    2>&1 | tee $LOG
