export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_NOISE=20    # estadísticas [NOISE]/[GRAD]/[FLOATER]/[DEADGATE] cada 20 aplicaciones de ruido
export DEBUG_MEM=1000    # pico de memoria + dev_free cada 1000 iters (delata zombies/OOM)

# ───────────────────────────────────────────────────────────────────────────
# ÚNICO bloque a editar entre runs. Todo lo demás (source, model, log) se deriva.
DATASET=flowers          # nombre de la carpeta en Datasets/ (flowers, bonsai, garden…)
RUN=16                    # número de run → output/m360/${DATASET}_beta_run${RUN}
DEAD_SUSTAIN=5           # N del gate de muerte sostenida (relocate solo tras N checks bajo cull)

MODEL=output/m360/${DATASET}_beta_run${RUN}
LOG=logs/${DATASET}${RUN}.log
# ───────────────────────────────────────────────────────────────────────────

# run16: VUELTA AL MCMC + nuevo gate de muerte sostenida (--mcmc_dead_sustain).
# Tras el fix del bug de load_ply (beta→1.0) que falseaba metrics.py, el clásico run15
# dio honesto PSNR 20.13 / SSIM 0.548 / LPIPS 0.387. El MCMC (run9) tuvo in-train 21.54
# > clásico → puede ganar. Probamos MCMC con el RUIDO DE RUN9 (híbrido covarianza,
# componente normal isotrópica) + el low_opacity_counter reimplementado como gate.
#
# RUIDO = RUN9 (confirmado en logs/flowers9.log):
#   --cov_noise --cov_noise_normal 1.0  → anisótropo en el plano del surfel + normal
#       ISOTRÓPICA = 1.0·media(escalas del plano). noise_opacity_exponent=100 (default).
#   --noise_lr 3e3  (ojo: el default cambió a 5e4; run9 usó 3e3 explícito).
#
# GATE DE MUERTE SOSTENIDA (NUEVO, el objetivo del experimento):
#   --mcmc_dead_sustain N  → dead_mask del relocate pasa de instantáneo (opac<=cull) a
#       sostenido: el splat debe llevar N checks de densify (N·densification_interval =
#       N·100 iters) CONSECUTIVOS bajo el cull antes de reciclarse. Da al fondo (que
#       fluctúa) tiempo de recuperar opacidad antes de ser reubicado. N=0 = comportamiento
#       run9 (instantáneo). Empezamos con N=5 (=500 iters). VIABLE en MCMC porque
#       opacity_reset está OFF (1e9) y se arregló densification_postfix para PRESERVAR el
#       contador (antes lo reseteaba cada add_new_gs → habría sido código muerto otra vez).
#       Verás [DEADGATE it…] en el log (bajo DEBUG_NOISE): opac<=cull_ahora vs sostenido.
#
# Resto = config MCMC de la era run9/run13 (reset OFF, cap 7.5M, floater_cull 0.2 = run9):
#   opacity_reg=0.01 (default/run9; el 0.05 de run13 era un barrido aparte → revertido).
#   lambda_dist=10, lambda_normal=0.05, scale_reg=0.01, opacity_cull=0.01,
#   mcmc_error_weight=3.5, mcmc_jitter_scale=1.5 (valores de run13; run9 exactos no
#   registrados — ajustar si se quiere repro pura de run9).
#   iterations=30000 (early-stop consolidado: el test pica ~30k; run9 a 50k decaía).
#   densify_until_iter=25000 (cierre MCMC + 5k consolidación).
#
# NO recompila CUDA (solo Python). Tras el run: render_server.sh (RUN=16, ITER=30000) +
# metrics.py. Comparar LPIPS/SSIM/PSNR vs run15 clásico (0.387/0.548/20.13) y run9
# re-baseline. En el log vigilar [DEADGATE] (¿el gate reduce los relocate?) y nº de splats.
python train.py -s Datasets/${DATASET} \
    -m $MODEL \
    --eval \
    --iterations 30000 \
    --test_iterations 7000 15000 20000 25000 30000 \
    --densify_until_iter 25000 \
    --lambda_normal 0.05 \
    --lambda_dist 10 \
    --opacity_reset_interval 1000000000 \
    --cap_max 7500000 \
    --noise_lr 3e3 \
    --scale_reg 0.01 \
    --opacity_reg 0.01 \
    --opacity_cull 0.01 \
    --floater_cull_dist 0.2 \
    --mcmc_error_weight 3.5 \
    --mcmc_jitter_scale 1.5 \
    --cov_noise \
    --cov_noise_normal 1.0 \
    --mcmc_dead_sustain $DEAD_SUSTAIN \
    2>&1 | tee $LOG
