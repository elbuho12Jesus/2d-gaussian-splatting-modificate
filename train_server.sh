export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_MEM=1000    # pico de memoria + dev_free cada 1000 iters (delata zombies/OOM)
# (DEBUG_NOISE no aplica: el ruido MCMC está OFF en el camino clásico)

# ───────────────────────────────────────────────────────────────────────────
# ÚNICO bloque a editar entre runs. Todo lo demás (source, model, log) se deriva.
DATASET=flowers          # nombre de la carpeta en Datasets/ (flowers, bonsai, garden…)
RUN=14                    # número de run → output/m360/${DATASET}_beta_run${RUN}

MODEL=output/m360/${DATASET}_beta_run${RUN}
LOG=logs/${DATASET}${RUN}.log
# ───────────────────────────────────────────────────────────────────────────

# run14: PRUEBA — densificación CLÁSICA 2DGS (clone/split + prune + opacity_reset)
# en lugar de MCMC. Objetivo: aislar si el garabato del fondo es culpa del MCMC.
# Flag nuevo --classic_densify activa densify_and_prune (clone para under-recon +
# split para over-recon, dirigido por gradiente de viewspace) + opacity_reset=3000.
# El ruido posicional MCMC y el cull de floaters NO se aplican (son del MCMC).
# Config alineada con el 2DGS ORIGINAL (el que da test 20.92 en flowers):
#   - opacity_reset_interval=3000  → ciclo nativo reset→recupera/prune (SEGURO sin
#     ruido MCMC; el colapso de runs 10/11/12 era por la compuerta (1−o)^100, OFF aquí).
#   - densify_until_iter=15000, iterations=30000 → ventana exacta del original.
#   - lambda_dist=0 (receta de malla, image-neutral en el original; el 10 era nuestro).
#   - opacity_reg=0, scale_reg=0 → regs L1 de Beta/MCMC desactivadas (no están en 2DGS).
#   - opacity_cull=0.005 → min_opacity del prune original.
# NO se pasan cap_max / noise_lr / mcmc_* / cov_noise / floater_cull_dist: son del
# camino MCMC y el clásico los ignora.
# Tras el run: render_server.sh (RUN=14, ITER=30000) + metrics.py. Comparar
# LPIPS/SSIM/PSNR vs run9 (MCMC sano ~21.5) y vs original (20.92). Vigilar el
# número de splats (clásico crece libre, sin cap) por si OOM en 95 GB.
python train.py -s Datasets/${DATASET} \
    -m $MODEL \
    --eval \
    --classic_densify \
    --iterations 30000 \
    --test_iterations 7000 15000 20000 25000 30000 \
    --densify_from_iter 500 \
    --densify_until_iter 15000 \
    --densification_interval 100 \
    --densify_grad_threshold 0.0002 \
    --percent_dense 0.01 \
    --opacity_reset_interval 3000 \
    --opacity_cull 0.005 \
    --lambda_normal 0.05 \
    --lambda_dist 0 \
    --opacity_reg 0 \
    --scale_reg 0 \
    2>&1 | tee $LOG
