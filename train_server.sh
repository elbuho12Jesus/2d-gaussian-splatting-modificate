export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_NOISE=20    # estadísticas [NOISE]/[GRAD]/[FLOATER]/[DEADGATE] cada 20 aplicaciones de ruido
export DEBUG_MEM=1000    # pico de memoria + dev_free cada 1000 iters (delata zombies/OOM)

# ───────────────────────────────────────────────────────────────────────────
# ÚNICO bloque a editar entre runs. Todo lo demás (source, model, log) se deriva.
DATASET=flowers          # nombre de la carpeta en Datasets/ (flowers, bonsai, garden…)
RUN=28                    # número de run → output/m360/${DATASET}_beta_run${RUN}
DEAD_SUSTAIN=5           # base run16 (óptimo del gate; >5 = más niebla, monótono)
CAP_MAX=7500000          # 7.5M = base run16 (cap saturado; mantenido por decisión del usuario)
OPACITY_REG=0.01         # base run16
SCALE_REG=0.01           # base run16

MODEL=output/m360/${DATASET}_beta_run${RUN}
LOG=logs/${DATASET}${RUN}.log
# ───────────────────────────────────────────────────────────────────────────

# run28: ABLACIÓN DE DISTORSIÓN. Réplica EXACTA de run16 (mejor MCMC) salvo lambda_dist 10→0.
# Objetivo: aislar cuánto aporta la distorsión, que run16 SÍ llevaba (lambda_dist=10) sin que
# lo supiéramos hasta ahora (el log de run16 muestra distort=0.00004…0.00160, NO 0.00000 →
# confirmado activo). run27 fue el 1er test de distorsión EN CLÁSICO (falló); este es el 1er
# test SIN distorsión en MCMC. Único cambio vs run16: --lambda_dist 10 → 0. Todo lo demás
# idéntico a run16 (la base): dead_sustain 5, scale_reg 0.01, opacity_reg 0.01, cap 7.5M.
#
# HIPÓTESIS: en flowers (no acotada) distort≈0.0016 = efecto débil pero NO nulo (~3-8% de la
# loss). Si run28 ≈ run16 → la distorsión es irrelevante aquí (confirma que está bien descartada).
# Si run28 mejora SSIM/LPIPS → la distorsión adelgazaba superficies y alimentaba el velo (como
# en clásico run27). Si run28 empeora → la distorsión sí ayudaba algo en MCMC.
#
# Resto = run16: ruido run9 (--cov_noise --cov_noise_normal 1.0 --noise_lr 3e3), reset OFF,
# lambda_normal=0.05, opacity_cull=0.01, floater_cull_dist=0.2, mcmc_error_weight=3.5,
# mcmc_jitter_scale=1.5, iterations=30000, densify_until_iter=25000.
#
# En el log vigilar: distort DEBE ser 0.00000 todo el run (verifica que lambda_dist=0 surtió
# efecto); ¿exceso-brillo/niebla per-vista vs run16? nº de splats. NO recompila CUDA. Tras el
# run: render_server.sh (RUN=28) + metrics.py. Comparar HONESTO vs run16 (0.339/0.597/20.21).
python train.py -s Datasets/${DATASET} \
    -m $MODEL \
    --eval \
    --densify_mode mcmc \
    --iterations 30000 \
    --test_iterations 7000 15000 20000 25000 30000 \
    --densify_until_iter 25000 \
    --lambda_normal 0.05 \
    --lambda_dist 0 \
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
