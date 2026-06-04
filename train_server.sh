export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py -s Datasets/flowers \
    -m output/m360/flowers_beta_run2 \
    --iterations 50000 \
    --test_iterations 7000 15000 30000 50000 \
    --densify_until_iter 45000 \
    --lambda_normal 0.05 \
    --lambda_dist 40 \
    --opacity_reset_interval 1000000000 \
    --cap_max 6000000 \
    --noise_lr 5e2 \
    --scale_reg 0.005 \
    --opacity_cull 0.005 \
    --mcmc_error_weight 2.0 \
    --mcmc_jitter_scale 1.5 \
    --cov_noise \
    --cov_noise_normal 1.0 \
    2>&1 | tee logs/flowers2.log
