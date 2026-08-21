export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_MEM=1000    # pico de memoria + dev_free cada 1000 iters (delata zombies/OOM)
export DEBUG_DENSIFY=1   # imprime [DENSIFY]/[RESET] (default ON; =0 silencia)
# (DEBUG_NOISE NO aplica: el ruido posicional MCMC está OFF en el camino clásico)

# ═══════════════════════════════════════════════════════════════════════════
#  TRAIN — DENSIFICACIÓN CLÁSICA 2DGS (clone/split + prune + opacity_reset)
# ═══════════════════════════════════════════════════════════════════════════
# Camino clásico: --densify_mode classic. densify_and_prune dirigido por
# ‖∂L/∂μ₂D‖ (clone bajo under-recon + split bajo over-recon) + opacity_reset
# periódico. El ruido posicional MCMC, el cull de floaters y mcmc_* NO se aplican
# (son del camino MCMC, el clásico los ignora). cap_max tampoco aplica: el clásico
# crece libre → VIGILAR nº de splats por OOM. Para el MCMC → train_MCMC_server.sh.
#
# OJO opacity_reset: en clásico es SEGURO (3000 = ciclo nativo reset→recupera/prune;
# el colapso de runs 10/11/12 era la compuerta (1−o)^100 del MCMC, OFF aquí). En MCMC
# rompe. Palanca propia del clásico: classic_prune_sustain (prune sostenido N pasos).
#
# DEFAULTS = run15 (mejor clásico honesto: 20.13 / 0.548 / 0.387; prune inmediato +
# FIX load_ply beta). Modos de error opuestos al MCMC: el clásico hace HUECOS NEGROS
# (sub-cobertura del fondo rasante), el MCMC hace VELO translúcido. Tras el run:
# render_server.sh + metrics.py + fila al historial y a docs/comparativa_runs.html.
# ───────────────────────────────────────────────────────────────────────────
# ÚNICO bloque a editar entre runs. Todo lo demás (source, model, log) se deriva.
DATASET="tandt/train"               # carpeta en Datasets/ (flowers, bonsai, garden…)
RUN=91                        # nº de run → output/m360/${DATASET}_beta_run${RUN}
# ═══════════════════════════════════════════════════════════════════════════
# run80 = ANCLA run79 + UN SOLO DELTA: --classic_prune_world_raw
# ═══════════════════════════════════════════════════════════════════════════
# ANCLA = run79 (20.7385 / 0.5767 / 0.3726 honesto, in-train 20.8107, train@30k 23.40,
# 3.716M splats). Se mantiene sobre run79 y NO sobre run67 para que el A/B siga siendo
# de UN SOLO DELTA conservando los fixes de corrección ya medidos (transmitancia).
#
# QUÉ CAMBIA: el prune por TAMAÑO-MUNDO del clásico estaba MUERTO. El criterio
# `s > 0.1·extent` se evaluaba sobre `get_scaling`, que YA viene clampada a
# `scale_clamp_factor·extent` = exactamente 0.1·extent con el default ⇒ la comparación
# estricta nunca se cumplía (en TODOS los logs del historial: `world=0`). El 2DGS
# original no clampa, así que allí ese prune sí actúa. Con --classic_prune_world_raw el
# criterio pasa a evaluarse sobre la escala CRUDA (activación sin clamp) = fiel al original.
#
# MASA AFECTADA (medida en run79, print [PRUNE-WORLD]): 52.018 splats = 1.402%. Es tres
# órdenes de magnitud más que los clamps ya descartados (β: 0.0002%) → el único pendiente
# de la auditoría con masa suficiente para mover la métrica. Respaldo del [CLAMP] de run79:
# 1.344% de componentes topadas y `s_raw max` 3.21 contra un techo de 0.4816 (recorte 2.73)
# → hay gigantes reales debajo del clamp (los del mecanismo de run21).
#
# RIESGO CONOCIDO: el clásico ya falla por HUECOS NEGROS (sub-cobertura del fondo rasante)
# y esos gigantes son justo los surfels del fondo. Podarlos puede limpiar velo o abrir más
# huecos — es lo que mide el run. Vigilar `dmean(render−gt)`: run79 está en −0.8156
# (sub-brillante); si el prune abre huecos, se hará MÁS negativo.
#
# VERIFICACIÓN EN EL LOG: `[PRUNE-WORLD]` debe decir «escala usada = CRUDA -> ACTIVO» y
# el `[DENSIFY]` debe mostrar `world=N` con N>0 (histórico: world=0 siempre).
# ANCLA = run67 (MEJOR CLÁSICO: 20.6684 / 0.5811 / 0.3675 honesto, train@30k 23.63,
# 4.875M splats). TODO lo demás es idéntico (regs=0, prune_sustain 25, reset 3000,
# CULL_SUBPIXEL=1, β en [-4,2]).
#
# QUÉ CAMBIA (auditoría de β/densificación, 2026-08-17): de las 4 rutas de creación de
# splats, relocate_gs y add_new_gs (MCMC) ya usaban la regla oficial de CONSERVACIÓN DE
# TRANSMITANCIA α' = 1-(1-α)^(1/K), pero clone/split (CLÁSICO) se habían quedado con el
# reparto LINEAL, que es la misma fórmula que en su día se identificó como bug en MCMC
# ("MCMC sobre-densidad: usaban α*0.5 → fix oficial"). El reparto lineal NO conserva la luz:
#   · split: 2 hijos con α/2  → (1-α/2)² > 1-α          ⇒ cada split ACLARA
#   · clone: clon con α/2 y el padre SE QUEDA con α     ⇒ cada clone OSCURECE
# Con α=0.6: split correcto = 0.368 por hijo, no 0.30 (T 0.40 vs 0.49 = +22% de luz).
# El sesgo se aplica en CADA densificación (145 pasos entre iter 500 y 15000) y acumula.
#
# HIPÓTESIS: parte del "óptimo plano" de run67 (run68-75: ni escala, ni ritmo de poda,
# ni techo de β lo movían) es que la densificación estaba desajustando el brillo a cada
# paso. Es el único mecanismo estructural del clásico que quedaba sin auditar.
#
# VERIFICACIÓN EN EL LOG (nuevos prints, ver también DEBUG abajo):
#   [DENSIFY-OPA] al arrancar  → debe decir modo = 'transmittance'
#   [DENSIFY-SPLIT n=...] / [DENSIFY-CLONE n=...] en cada densificación → el campo
#      "T: x -> y (err +z%)" debe dar err ≈ +0.00% (en modo linear da err > 0 en split
#      y < 0 en clone), y "d_beta_max" debe ser 0.00e+00 EXACTO (β se hereda tal cual =
#      el trinquete de run65/66 sigue muerto).
#
# ⚠ REQUIERE recompilar el rasterizer SOLO si se quiere el fix de dL_dbeta (término de
#   fondo). Ese fix es BIT-EXACTO con fondo negro (verificado contra diferencias finitas:
#   mismos dígitos), así que flowers NO cambia y el A/B contra run67 sigue siendo limpio
#   se recompile o no:
#   docker compose exec surfel_env pip install --force-reinstall --no-deps /workspace/submodules/diff-surfel-rasterization
#
# ═══ (histórico) run67 = ANCLA run26 (clásico sano) + UN SOLO DELTA: fix del trinquete de β ═══
# Deltas vs run66 (los 2 primeros REVIERTEN la regresión medida, no son experimento):
#   · OPACITY_REG 0.02→0 y SCALE_REG 0.06→0  = reguladores de run25/26. A/B local midió
#     que scale_reg 0.06 cuesta 1.68 dB y opacity_reg 0.02 otros 0.14 dB en clásico.
#   · PRUNE_SUSTAIN 5→25 = igual que run26, para que el ancla sea exacta.
# DELTA NUEVO (el experimento): gaussian_model.py:574 ya NO resta math.log(N) al crear
#   los hijos del split → β se hereda tal cual (docs/beta_trinquete_split_clasico.html).
# ANCLA de comparación = run26: 19.99 / 0.537 / 0.394 honesto, in-train train@30k 22.22.
#
# CULL_SUBPIXEL=1 SE MANTIENE (decisión del usuario 2026-07-20: seguir ese experimento).
#   Consecuencia: run67 tiene DOS deltas vs run26 (fix de beta + cull), porque run25/26
#   son PRE-cull (build del cull = 2026-07-05). PERO el fix de beta YA ESTA AISLADO por
#   el A/B local (flowers, 2500 it, regs=0, prune_sustain=25) con CULL=1 en AMBOS brazos:
#       con trinquete: train 18.84 | beta<0.1 = 57.4% | beta min 0.0733 (suelo)
#       con el fix   : train 19.31 | beta<0.1 =  0.0% | beta min 0.5110
#   -> +0.47 dB y colapso de beta ELIMINADO, sin NaN. El cull es comun a los dos brazos,
#      asi que no contamina esa medida. Lo que run67 anade es la medida HONESTA a 30k.
#   OJO al interpretar: comparar el ABSOLUTO de run67 contra run26 mezcla los 2 deltas.
#   Si hace falta el single-delta a 30k, el run que falta es "clasico + CULL + regs=0 +
#   trinquete" (= run67 sin el fix), no desactivar el cull.

# ⚠ run65 se tituló "small clamp" pero NUNCA exportó esta env var → corrió con el
# default 0.1. Dejarla explícita aquí + el print [CLAMP] evita repetir el fallo.
export SCALE_CLAMP_FACTOR=0.1  # 0.1 = default 3DGS/2DGS; run64 probó 0.05

PRUNE_SUSTAIN=25              # CONFIG GANADORA = run67 (MEJOR CLÁSICO: 20.6684/0.5811/0.3675 honesto).
                              # Barrido LIMPIO (β [-4,2], trinquete muerto) CERRADO: ps 5=20.5439(run74)
                              # · 10=20.7436(run72) · 20=20.6318(run75) · 25=20.6684(run67) → plano
                              # dentro de ~0.20 dB, SSIM/LPIPS clavados → prune_sustain NO es palanca.
OPACITY_RESET_INTERVAL=3000   # ciclo reset→recupera/prune (SEGURO en clásico; del 2DGS original)
DENSIFY_FROM=500              # inicio de densificación
DENSIFY_UNTIL=15000           # fin de densificación (ventana del 2DGS original)
DENSIFICATION_INTERVAL=100    # cada cuántas iters se densifica/poda
DENSIFY_GRAD_THRESHOLD=0.0002 # umbral de ‖∂L/∂μ₂D‖ para clone/split
PERCENT_DENSE=0.01            # umbral de tamaño clone vs split
OPACITY_CULL=0.005             # min_opacity del prune (0.005=2DGS original; 0.01=poda más agresiva, run65)
LAMBDA_DIST=0                 # reg distorsión (0 = receta 2DGS original; el 10 era nuestro)
LAMBDA_NORMAL=0.05            # reg de consistencia de normales
OPACITY_REG=0                 # OFF = run25/26. En clásico la L1 cuesta PSNR (A/B local: 0.02 -> -0.14 dB)
SCALE_REG=0                   # OFF = run25/26. A/B local: 0.06 cuesta -1.68 dB en clásico
ITERATIONS=30000
# ═══ EL DELTA DE run79 ═══ "linear" = histórico (todos los runs ≤75) · "transmittance" = FIX.
# Sesgo del modo linear MEDIDO en el smoke test (bonsai, prints [DENSIFY-*]):
#   SPLIT  T 0.247 -> 0.397 = +61% de luz  (ACLARA: 2 hijos con α/2)
#   CLONE  T 0.225 -> 0.154 = −31% de luz  (OSCURECE: clon α/2 + padre con α entera)
# En modo transmittance ambos dan err +0.00%.
DENSIFY_OPACITY_MODE=transmittance   # heredado de run79 (fix de corrección, ya medido)
# ═══ EL DELTA DE run80 ═══ prune por tamaño-mundo sobre la escala CRUDA (2DGS original).
# Ponerlo a "" (vacío) revierte al comportamiento histórico (prune muerto).
PRUNE_WORLD_RAW=--classic_prune_world_raw

MODEL=output/m360/${DATASET}_beta_run${RUN}
LOG=logs/${DATASET}${RUN}.log
# ───────────────────────────────────────────────────────────────────────────

python train.py -s Datasets/${DATASET} \
    -m $MODEL \
    --eval \
    --densify_mode classic \
    --iterations $ITERATIONS \
    --test_iterations 7000 15000 20000 25000 30000 \
    --densify_from_iter $DENSIFY_FROM \
    --densify_until_iter $DENSIFY_UNTIL \
    --densification_interval $DENSIFICATION_INTERVAL \
    --densify_grad_threshold $DENSIFY_GRAD_THRESHOLD \
    --percent_dense $PERCENT_DENSE \
    --opacity_reset_interval $OPACITY_RESET_INTERVAL \
    --opacity_cull $OPACITY_CULL \
    --lambda_normal $LAMBDA_NORMAL \
    --lambda_dist $LAMBDA_DIST \
    --opacity_reg $OPACITY_REG \
    --scale_reg $SCALE_REG \
    --classic_prune_sustain $PRUNE_SUSTAIN \
    --densify_opacity_mode $DENSIFY_OPACITY_MODE \
    $PRUNE_WORLD_RAW \
    2>&1 | tee $LOG

# ─── QUÉ MIRAR EN EL LOG (verificación de que los fixes están activos) ───
#   grep "DENSIFY-OPA"   $LOG   → modo = 'transmittance'
#   grep "DENSIFY-SPLIT" $LOG   → err ≈ +0.00% (en linear daba ~+61%)
#   grep "DENSIFY-CLONE" $LOG   → err ≈ +0.00% (en linear daba ~−31%)
#   grep "d_beta_max"    $LOG   → SIEMPRE 0.00e+00 (β se hereda: trinquete muerto)
#   grep "BETA-TECHO"    $LOG   → clamp [-4,2] → beta [0.0733, 29.5562] + topados techo/suelo
#   grep "PRUNE-WORLD"   $LOG   → «escala usada = CRUDA -> ACTIVO, poda N» (antes: MUERTO)
#   grep "world="        $LOG   → en el [DENSIFY]: world > 0 (histórico: world=0 siempre)
