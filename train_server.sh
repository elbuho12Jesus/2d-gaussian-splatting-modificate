export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_DENSIFY=1    # [DENSIFY]/[RESET] cada densify: clone/split, prune sostenido, counter, N
export DEBUG_MEM=1000     # pico de memoria + dev_free cada 1000 iters (delata zombies/OOM)

# ───────────────────────────────────────────────────────────────────────────
# ÚNICO bloque a editar entre runs. Todo lo demás (source, model, log) se deriva.
DATASET=flowers          # nombre de la carpeta en Datasets/ (flowers, bonsai, garden…)
RUN=18                    # número de run → output/m360/${DATASET}_beta_run${RUN}
PRUNE_SUSTAIN=25         # N del prune sostenido clásico (poda tras N densifies consecutivos bajo cull)

MODEL=output/m360/${DATASET}_beta_run${RUN}
LOG=logs/${DATASET}${RUN}.log
# ───────────────────────────────────────────────────────────────────────────

# run18: densificación CLÁSICA 2DGS + PRUNE SOSTENIDO por opacidad (idea del usuario).
# Sustituye el prune INMEDIATO de run15 (opac<cull → poda en el acto) por uno SOSTENIDO:
# --classic_prune_sustain N → el splat debe llevar N pasos de densify CONSECUTIVOS bajo el
# cull (low_opacity_counter > N) antes de podarse, dándole tiempo a "asentarse"/recuperar
# opacidad por gradiente (+ el rescate del opacity_reset) antes de reciclarlo.
#
# CONFIG: reset 3000 + N=25 (decisión del usuario). Aritmética que lo hace VIABLE (no
# código muerto como run14): densification_interval=100 + opacity_reset_interval=3000 →
# 30 densifies entre resets; el reset sube todo a 0.01>cull y borra el counter, pero con
# N=25 (<30) el counter SÍ cruza el umbral en los ~4-5 densifies previos a cada reset →
# poda real. (run14 falló porque N=50 > 30 nunca se alcanzaba.) Conserva el opacity_reset
# clásico y por tanto el size-prune (big_points, activo tras el 1er reset).
#
# Resto = config clásica de run15 (la honesta: PSNR 20.13 / SSIM 0.548 / LPIPS 0.387):
# SH3, lambda_normal=0.05, lambda_dist=0 (image-neutral), opacity_cull=0.005,
# opacity_reg=0 + scale_reg=0 (regs Beta/MCMC OFF), densify_until=15000, 30k iters.
# SIN cap_max (el clásico crece libre; run15 llegó a 9.4M) y SIN flags MCMC.
#
# En el log [DENSIFY] vigilar: PRUNE total>0 con "sostenido(>25)=…" y cnt_max acercándose
# a ~30 antes de cada [RESET]. Si PRUNE total=0 siempre → counter muerto (revisar). NO
# recompila CUDA. Tras el run: render_server.sh (RUN=18, ITER=30000) + metrics.py.
# Comparar LPIPS/SSIM/PSNR vs run15 (prune inmediato, 0.387/0.548/20.13).
python train.py -s Datasets/${DATASET} \
    -m $MODEL \
    --eval \
    --densify_mode classic \
    --iterations 30000 \
    --test_iterations 7000 15000 20000 25000 30000 \
    --densify_from_iter 500 \
    --densify_until_iter 15000 \
    --densification_interval 100 \
    --densify_grad_threshold 0.0002 \
    --percent_dense 0.01 \
    --opacity_reset_interval 3000 \
    --opacity_cull 0.005 \
    --classic_prune_sustain $PRUNE_SUSTAIN \
    --lambda_normal 0.05 \
    --lambda_dist 0 \
    --scale_reg 0 \
    --opacity_reg 0 \
    2>&1 | tee $LOG
