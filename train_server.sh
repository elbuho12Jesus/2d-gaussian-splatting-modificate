export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_NOISE=20    # estadísticas [NOISE]/[GRAD] cada 20 aplicaciones de ruido (verificar |disp|~0.05)
export DEBUG_MEM=1000    # pico de memoria + dev_free cada 1000 iters (delata zombies/OOM)

# run6 (Spherical Betas, palanca #5a): run5 (anti-overfit: scale_reg/opacity_cull ×2 +
# floater_cull) NO movió el test (21.14 vs 21.25 de run4) → los floaters eran síntoma.
# La causa dominante del blur de césped/cielo es estructural: surfel 2D + SH no
# representa fondo view-dependent desde cobertura rasante. Port del color oficial:
#  - --sh_degree 0 --sb_number 2: color = DC + 2 lóbulos Spherical Beta por splat
#    (C = SH0 + Σ cᵢ·max(dot(μᵢ,v),0)^(4·exp(bᵢ))). 15 params de color vs 48 de SH3 →
#    MENOS capacidad de memorizar por-vista y MÁS capacidad view-dependent real.
#  - sb_params_lr 0.0025 (default oficial, ya en arguments).
#  - resto idéntico a run5 para aislar el cambio.
# Métrica a vigilar: gap train−test y si el test deja de decaer 30k→50k.
python train.py -s Datasets/flowers \
    -m output/m360/flowers_beta_run6 \
    --eval \
    --sh_degree 0 \
    --sb_number 2 \
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
    2>&1 | tee logs/flowers6.log
