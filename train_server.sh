export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_NOISE=20    # estadísticas [NOISE]/[GRAD]/[FLOATER]/[DEADGATE] cada 20 aplicaciones de ruido
export DEBUG_MEM=1000    # pico de memoria + dev_free cada 1000 iters (delata zombies/OOM)

# ───────────────────────────────────────────────────────────────────────────
# ÚNICO bloque a editar entre runs. Todo lo demás (source, model, log) se deriva.
DATASET=flowers          # nombre de la carpeta en Datasets/ (flowers, bonsai, garden…)
RUN=19                    # número de run → output/m360/${DATASET}_beta_run${RUN}
DEAD_SUSTAIN=25          # N del gate de muerte sostenida MCMC (relocate solo tras N checks bajo cull)
CAP_MAX=5000000          # 5M (run16=7.5M saturó velo; run17=3M lo perdió capacidad → 5M intermedio)

MODEL=output/m360/${DATASET}_beta_run${RUN}
LOG=logs/${DATASET}${RUN}.log
# ───────────────────────────────────────────────────────────────────────────

# run19: MCMC con GATE MÁS ALTO (dead_sustain 5→25) + cap 5M. Hipótesis del usuario:
# el MCMC "presta poca atención al fondo" porque recicla sus splats demasiado pronto.
# En el clásico (run18) subir el prune sostenido a N=25 SÍ rellenó el fondo (más árboles).
# Trasladamos la idea al MCMC: con --mcmc_dead_sustain 25 el relocate exige 25 checks de
# densify CONSECUTIVOS bajo cull (25·100=2500 iters) antes de reubicar un splat → da al
# fondo (que fluctúa) mucho más tiempo para recuperar opacidad antes de reciclarlo.
# (run16 usó 5 = 500 iters.) VIABLE en MCMC porque opacity_reset está OFF (1e9) → nada
# borra el counter (a diferencia del clásico, donde el reset cada 3000 lo limita a <30).
#
# cap 5M: run16 (7.5M) saturó el cap y alimentó el velo translúcido; run17 (3M) perdió
# capacidad y EMPEORÓ (19.87 vs 20.21) sin quitar el velo. 5M = punto intermedio.
#
# Resto = config run16 (mejor MCMC honesto: PSNR 20.21 / SSIM 0.597 / LPIPS 0.339):
#   ruido run9 (--cov_noise --cov_noise_normal 1.0 --noise_lr 3e3), reset OFF,
#   lambda_dist=10, lambda_normal=0.05, scale_reg=0.01, opacity_reg=0.01, opacity_cull=0.01,
#   floater_cull_dist=0.2, mcmc_error_weight=3.5, mcmc_jitter_scale=1.5,
#   iterations=30000, densify_until_iter=25000.
#
# En el log vigilar [DEADGATE] (con N=25 el "sostenido(>25)" debe ser MUCHO menor que con
# N=5 → menos relocate → más splats de fondo sobreviven) y nº de splats (¿llega a 5M?).
# NO recompila CUDA. Tras el run: render_server.sh (RUN=19, ITER=30000) + metrics.py.
# Comparar LPIPS/SSIM/PSNR vs run16 (0.339/0.597/20.21) y run18 clásico (0.387/0.549/20.12).
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
    --opacity_reg 0.01 \
    --opacity_cull 0.01 \
    --floater_cull_dist 0.2 \
    --mcmc_error_weight 3.5 \
    --mcmc_jitter_scale 1.5 \
    --cov_noise \
    --cov_noise_normal 1.0 \
    --mcmc_dead_sustain $DEAD_SUSTAIN \
    2>&1 | tee $LOG
