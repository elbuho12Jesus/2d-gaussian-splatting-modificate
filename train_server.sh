export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_NOISE=20    # estadísticas [NOISE]/[GRAD] cada 20 aplicaciones de ruido (verificar |disp|~0.05)
export DEBUG_MEM=1000    # pico de memoria + dev_free cada 1000 iters (delata zombies/OOM)

# run5 (anti-overfit): flowers4 reveló gap train-test ~4 dB (test 21.25 vs train 25.34) y
# fondo emborronado en vistas held-out REALES. Tres cambios, todos contra el sobreajuste:
#  - scale_reg 0.005→0.01 (oficial): la bajada a 0.005 se validó con métrica train-only;
#    penaliza los surfels gigantes translúcidos del fondo (smax p95 0.54-0.85 en run4).
#  - opacity_cull 0.005→0.01: recicla antes los splats casi-muertos vía relocate.
#  - --floater_cull_dist 0.2 (NUEVO): mata splats a <0.2·extent de cualquier cámara de
#    train (en run4: 49k splats, op_med 0.25 — cola clara, el 99% vive a >0.25·extent).
python train.py -s Datasets/flowers \
    -m output/m360/flowers_beta_run5 \
    --eval \
    --iterations 50000 \
    --test_iterations 7000 15000 30000 50000 \
    --densify_until_iter 45000 \
    --lambda_normal 0.05 \
    --lambda_dist 10 \
    --opacity_reset_interval 1000000000 \
    --cap_max 7500000 \
    --noise_lr 3e3 \
    --scale_reg 0.01 \
    --opacity_cull 0.01 \
    --floater_cull_dist 0.2 \
    --mcmc_error_weight 3.5 \
    --mcmc_jitter_scale 1.5 \
    --cov_noise \
    --cov_noise_normal 1.0 \
    2>&1 | tee logs/flowers5.log
