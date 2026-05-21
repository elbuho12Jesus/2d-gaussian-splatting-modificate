python train.py -s /home/jesus/Documents/Gaussian_splatting/360_extra_scenes/flowers -m output/m360/flowers \
    --iterations 50000 \
    --densify_until_iter 25000 \
    --opacity_reset_interval 5000 \
    --lambda_normal 0.05 \
    --lambda_dist 100 \
    --cap_max 500000
