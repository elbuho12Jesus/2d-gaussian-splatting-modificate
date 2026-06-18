export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_NOISE=20    # estadísticas [NOISE]/[GRAD]/[FLOATER]/[DEADGATE] cada 20 aplicaciones de ruido
export DEBUG_MEM=1000    # pico de memoria + dev_free cada 1000 iters (delata zombies/OOM)

# ───────────────────────────────────────────────────────────────────────────
# ÚNICO bloque a editar entre runs. Todo lo demás (source, model, log) se deriva.
DATASET=flowers          # nombre de la carpeta en Datasets/ (flowers, bonsai, garden…)
RUN=22                    # número de run → output/m360/${DATASET}_beta_run${RUN}
DEAD_SUSTAIN=15          # punto medio entre 5 (limpio) y 25 (run19 = más niebla) → ¿sweet spot?
CAP_MAX=7500000          # 7.5M = base run16 (cap saturado; mantenido por decisión del usuario)
OPACITY_REG=0.01         # base run16
SCALE_REG=0.03           # punto medio entre run16 (0.01) y run21 (0.05): afinar sin pasarse

MODEL=output/m360/${DATASET}_beta_run${RUN}
LOG=logs/${DATASET}${RUN}.log
# ───────────────────────────────────────────────────────────────────────────

# run22: buscar SWEET SPOT en el punto medio de las 2 palancas que combinamos. Idea del
# usuario: dead_sustain↑ ayuda a CAPTAR el fondo (run18/19 = "más árboles traseros") y
# scale_reg afina el detalle. run22 cruza ambas a la mitad para ver si hay un equilibrio
# entre captar fondo y no reintroducir el velo. Base = run16 + 2 cambios:
#   - dead_sustain 5 → 15  (punto medio; run19=25 dio MÁS niebla, run16=5 limpio)
#   - scale_reg    0.01 → 0.03  (punto medio; run21=0.05 afinó cerca pero no tocó el fondo)
# cap_max 7.5M y opacity_reg 0.01 SIN cambios (decisión del usuario). SIN tocar relocate.
#
# AVISOS (de runs previos): (1) dead_sustain↑ recicla los moribundos MÁS TARDE → más
# floaters translúcidos sobreviven = más niebla (run19 con 25: exceso-brillo 8.86 vs run16
# 8.47). 15 < 25 debería ser menos, pero vigilar que no vuelva el velo. (2) scale_reg NO
# encoge los surfels gigantes del fondo (clavados en s(max)=extent=4.816 con 0.01 Y 0.05) →
# afina primer plano, no fondo. (3) VIGILAR render mean vs gt: si cae por debajo, nos pasamos.
#
# Resto = run16: ruido run9 (--cov_noise --cov_noise_normal 1.0 --noise_lr 3e3), reset OFF,
# lambda_dist=10, lambda_normal=0.05, opacity_cull=0.01, floater_cull_dist=0.2,
# mcmc_error_weight=3.5, mcmc_jitter_scale=1.5, iterations=30000, densify_until_iter=25000.
#
# En el log vigilar: ¿exceso-brillo/niebla per-vista vs run16/21? ¿[DEADGATE] recicla más
# tarde? ¿se vacía el fondo (render mean << gt)? nº de splats. NO recompila CUDA. Tras el run:
# render_server.sh (RUN=22) + metrics.py. Comparar vs run16 (0.339/0.597/20.21) y run21 (0.344/0.592/19.63).
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
