export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_NOISE=20    # estadísticas [NOISE]/[GRAD] cada 20 aplicaciones de ruido (verificar |disp|~0.05)
export DEBUG_MEM=1000    # pico de memoria + dev_free cada 1000 iters (delata zombies/OOM)

# run4 (plan #2): ruido ∝ tamaño real (train.py:221, s crudo en vez de media-1) + noise_lr re-calibrado 5e2→3e3.
# --eval: reserva 1/8 de las vistas para test → PSNR honesto en held-out (antes eval=False = train sobre todo).
python train.py -s Datasets/flowers \
    -m output/m360/flowers_beta_run4 \
    --eval \
    --iterations 50000 \
    --test_iterations 7000 15000 30000 50000 \
    --densify_until_iter 45000 \
    --lambda_normal 0.05 \
    --lambda_dist 10 \
    --opacity_reset_interval 1000000000 \
    --cap_max 7500000 \
    --noise_lr 3e3 \
    --scale_reg 0.005 \
    --opacity_cull 0.005 \
    --mcmc_error_weight 3.5 \
    --mcmc_jitter_scale 1.5 \
    --cov_noise \
    --cov_noise_normal 1.0 \
    2>&1 | tee logs/flowers4.log
