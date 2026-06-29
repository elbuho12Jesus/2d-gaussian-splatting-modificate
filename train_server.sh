export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_NOISE=20    # estadísticas [NOISE]/[GRAD]/[FLOATER]/[DEADGATE] cada 20 aplicaciones de ruido
export DEBUG_MEM=1000    # pico de memoria + dev_free cada 1000 iters (delata zombies/OOM)

# ───────────────────────────────────────────────────────────────────────────
# ÚNICO bloque a editar entre runs. Todo lo demás (source, model, log) se deriva.
DATASET=bonsai           # nombre de la carpeta en Datasets/ (flowers, bonsai, garden…)
RUN=42                    # número de run → output/m360/${DATASET}_beta_run${RUN}
DEAD_SUSTAIN=5           # base run16 (óptimo del gate; >5 = más niebla, monótono)
CAP_MAX=4500000          # 4.5M para bonsai (3× el cap oficial 1.5M; valor pedido por el usuario)
OPACITY_REG=0.06         # = run36 (óptimo de PSNR del sweep; HITO que domina al original en flowers)
SCALE_REG=0.01           # base run16

MODEL=output/m360/${DATASET}_beta_run${RUN}
LOG=logs/${DATASET}${RUN}.log
# ───────────────────────────────────────────────────────────────────────────

# run42: RÉPLICA de run36 (config HITO de flowers) sobre BONSAI con cap_max 4.5M.
# run36 = base run16 + opacity_reg=0.06 (óptimo de PSNR del sweep; PRIMERA run que DOMINA al original 2DGS
# en las 3 métricas honestas en flowers: 21.16/0.5926/0.3555). Aquí se traslada esa MISMA config a bonsai.
#
# PREGUNTA que responde: ¿la receta ganadora de flowers (MCMC + opacity_reg 0.06 + gate 5 + ruido run9)
# generaliza a OTRA escena, y encima de tipo distinto (bonsai = INTERIOR ACOTADA vs flowers = exterior
# no acotada)? Primer salto de dataset del proyecto.
#
# DOS CAMBIOS vs run36 (lo demás idéntico):
#   1) DATASET flowers → bonsai.
#   2) cap_max 7.5M → 4.5M (pedido por el usuario; OJO: el cap OFICIAL de bonsai es 1.5M → esto es 3×).
#
# OJOS EN ESTE RUN (régimen distinto al de flowers):
#   - lambda_dist=10: en flowers (no acotada) la distorsión era ≈0; en bonsai (acotada) SÍ es activa
#     → su gradiente pesa de verdad aquí. Se mantiene en 10 para ser fiel a run36, pero vigilar `distort`
#     en el log (si domina, podría hacer falta re-evaluarlo para bonsai).
#   - El "velo translúcido del fondo rasante" era el problema de flowers (exterior). bonsai no tiene fondo
#     rasante → el efecto de opacity_reg 0.06 puede ser otro. No dar por hecho el mecanismo de flowers.
#   - cap 4.5M sobre escena acotada: ¿se satura (topa exacto) o sobra presupuesto? Vigilar N final.
#
# NO requiere recompilar el rasterizer (solo λ de Python; --freeze_low_beta OFF = default run9/run16).
#
# Resto = run36/run16: lambda_dist 10, dead_sustain 5, scale_reg 0.01, ruido run9
# (--cov_noise --cov_noise_normal 1.0 --noise_lr 3e3), reset OFF, lambda_normal=0.05, opacity_cull=0.01,
# floater_cull_dist=0.2, mcmc_error_weight=3.5, mcmc_jitter_scale=1.5, iters=30000, densify_until=25000.
#
# Tras el run: render_server.sh (DATASET=bonsai, RUN=42) + metrics.py. NO hay baseline honesto de bonsai
# todavía → si se quiere comparar vs original, medir el 2DGS original en bonsai con el MISMO metrics.py.
# Añadir fila al historial + docs/comparativa_runs.html (nota: el historial es de flowers; marcar que es bonsai).
python train.py -s Datasets/${DATASET} \
    -m $MODEL \
    --eval \
    --densify_mode mcmc \
    --iterations 30000 \
    --test_iterations 7000 15000 20000 25000 30000 \
    --densify_until_iter 25000 \
    --lambda_normal 0.05 \
    --lambda_dist 10 \
    --opacity_reset_interval 1000000000 \
    --cap_max $CAP_MAX \
    --noise_lr 3e3 \
    --scale_reg $SCALE_REG \
    --opacity_reg $OPACITY_REG \
    --opacity_cull 0.01 \
    --floater_cull_dist 0.2 \
    --mcmc_error_weight 3.5 \
    --mcmc_jitter_scale 1.5 \
    --cov_noise \
    --cov_noise_normal 1.0 \
    --mcmc_dead_sustain $DEAD_SUSTAIN \
    2>&1 | tee $LOG
