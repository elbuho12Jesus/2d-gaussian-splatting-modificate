export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_NOISE=20    # estadísticas [NOISE]/[GRAD]/[FLOATER]/[DEADGATE] cada 20 aplicaciones de ruido
export DEBUG_MEM=1000    # pico de memoria + dev_free cada 1000 iters (delata zombies/OOM)

# ───────────────────────────────────────────────────────────────────────────
# ÚNICO bloque a editar entre runs. Todo lo demás (source, model, log) se deriva.
DATASET=flowers          # nombre de la carpeta en Datasets/ (flowers, bonsai, garden…)
RUN=32                    # número de run → output/m360/${DATASET}_beta_run${RUN}
DEAD_SUSTAIN=5           # base run16 (óptimo del gate; >5 = más niebla, monótono)
CAP_MAX=7500000          # 7.5M = base run16 (cap saturado; mantenido por decisión del usuario)
OPACITY_REG=0.05         # run31: 5× sobre run16 (0.01). run20 ya probó 0.02 (FALLÓ); empuje fuerte
SCALE_REG=0.01           # base run16

MODEL=output/m360/${DATASET}_beta_run${RUN}
LOG=logs/${DATASET}${RUN}.log
# ───────────────────────────────────────────────────────────────────────────

# run31: SUBIR opacity_reg 0.01→0.05 (5×). Réplica EXACTA de run16 (mejor MCMC), --freeze_low_beta OFF.
# MOTIVACIÓN (usuario): el kernel beta aproxima mejor el blending cuando la opacidad se mantiene baja
# (alpha = opacity·(1−r²)^beta, forward.cu:416). La L1 de opacidad (train.py:156) empuja TODAS las
# opacidades hacia abajo; pega más fuerte justo en los floaters semitransparentes del fondo (o~0.1–0.5,
# donde λ·o(1−o) es máxima → docs/regularizacion_opacidad_l1.html).
#
# OJO HISTORIAL: run20 YA subió opacity_reg 0.01→0.02 → 19.54/0.596/0.343 = PEOR PSNR (3er sweep reg.
# fallido: "la opacidad no distingue floater de fondo; tras densify_until la L1 vacía el fondo").
# run31 empuja 5× (vs el 2× tímido de run20) para testear de verdad la hipótesis del kernel.
# RIESGO: oscurecer/vaciar el fondo (la L1 sin reciclaje tras densify_until apaga la cobertura rasante).
#
# NO requiere recompilar el rasterizer (solo cambia un λ de Python; quitar --freeze_low_beta vuelve al
# default OFF = comportamiento run9/run16, sin tocar ABI). Recompilar SOLO si el rasterizer del servidor
# quedó en la variante de run30 y quieres asegurar el binario base.
#
# Resto = run16: lambda_dist 10, dead_sustain 5, scale_reg 0.01, cap 7.5M, ruido run9
# (--cov_noise --cov_noise_normal 1.0 --noise_lr 3e3), reset OFF, lambda_normal=0.05, opacity_cull=0.01,
# floater_cull_dist=0.2, mcmc_error_weight=3.5, mcmc_jitter_scale=1.5, iters=30000, densify_until=25000.
#
# En el log vigilar: opacidad media debe BAJAR claramente vs run16; ¿se vacía el fondo en las vistas de
# test? Tras el run: render_server.sh (RUN=31) + metrics.py. Comparar HONESTO vs run16 (20.21/0.597/0.339)
# y vs run20 (19.54/0.596/0.343, opacity_reg 0.02).
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
