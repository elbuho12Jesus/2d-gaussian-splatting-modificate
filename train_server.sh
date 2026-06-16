export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_MEM=1000    # pico de memoria + dev_free cada 1000 iters (delata zombies/OOM)
# (DEBUG_NOISE no aplica: el ruido MCMC está OFF en el camino clásico)

# ───────────────────────────────────────────────────────────────────────────
# ÚNICO bloque a editar entre runs. Todo lo demás (source, model, log) se deriva.
DATASET=flowers          # nombre de la carpeta en Datasets/ (flowers, bonsai, garden…)
RUN=15                    # número de run → output/m360/${DATASET}_beta_run${RUN}

MODEL=output/m360/${DATASET}_beta_run${RUN}
LOG=logs/${DATASET}${RUN}.log
# ───────────────────────────────────────────────────────────────────────────

# run15: FIX DE PODA — densificación CLÁSICA 2DGS con el PRUNE INMEDIATO restaurado
# (gaussian_model.py: opacity<cull + big_points_vs/ws cada densify). El run14 reveló
# (log DEBUG_DENSIFY) que el prune "DBS sostenido" era código muerto (PRUNE=0 en los
# 144 pasos → 4.37M splats invisibles sin eliminar, fondo negro, PSNR 11.30). Ahora la
# nube se limpia cada densify como en el 2DGS original. Misma config que run14 por lo
# demás. NO requiere recompilar (solo Python). Flag --classic_densify activa
# densify_and_prune (clone under-recon + split over-recon por gradiente) + opacity_reset.
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
# Tras el run: render_server.sh (RUN=15, ITER=30000) + metrics.py. Comparar
# LPIPS/SSIM/PSNR vs run14 (prune roto, 11.30), run9 (MCMC sano ~21.5) y original
# (20.92). En el log verificar que ahora PRUNE>0 y que N deja de crecer sin freno.
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
