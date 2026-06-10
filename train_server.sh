export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_NOISE=20    # estadísticas [NOISE]/[GRAD] cada 20 aplicaciones de ruido (verificar |disp|~0.05)
export DEBUG_MEM=1000    # pico de memoria + dev_free cada 1000 iters (delata zombies/OOM)

# ───────────────────────────────────────────────────────────────────────────
# ÚNICO bloque a editar entre runs. Todo lo demás (source, model, log) se deriva.
DATASET=flowers          # nombre de la carpeta en Datasets/ (flowers, bonsai, garden…)
RUN=11                    # número de run → output/m360/${DATASET}_beta_run${RUN}
OPACITY_REG=0.02         # barrido L1 opacidad (default código 0.01 → probar 0.02 / 0.05)

MODEL=output/m360/${DATASET}_beta_run${RUN}
LOG=logs/${DATASET}${RUN}.log
# ───────────────────────────────────────────────────────────────────────────

# run7: SIN Spherical Betas (run6 EMPEORÓ: test 20.65 vs 21.14 run5 — los lóbulos
# degeneran en "faros" que memorizan el cono de cámaras train; ver
# docs/diagnostico_lobulos_sb_run6.html). Se vuelve al color SH3 de run5:
# sin --sb_number → sb=0 → ruta antigua exacta (shs directo al rasterizer).
# Cambio único de este run vs run5: densify_until_iter 45000 → 35000. En runs
# 4/5/6 el test pica ~30k y la fase tardía (densify+ruido activos) solo
# memoriza → se corta la fase MCMC en 35k y los últimos 15k son consolidación
# pura (sin relocate/add/ruido/regs, solo fotométrico con lr decayendo).
# Métrica a vigilar: si el test @50k mantiene (o supera) el pico de ~30k.
python train.py -s Datasets/${DATASET} \
    -m $MODEL \
    --eval \
    --iterations 50000 \
    --test_iterations 7000 15000 30000 35000 50000 \
    --densify_until_iter 35000 \
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
