#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run56 — TEST DE HIPÓTESIS DEL VELO: reentreno de la config de run51 (flowers)
#         sobre el rasterizer con CULL_SUBPIXEL=1 (los splats sub-píxel dejan de existir).
#
# QUÉ PRUEBA:
#   La hipótesis "el velo / los haces de luz sobre-brillantes del fondo = splats SUB-PÍXEL
#   rescatados por el low-pass del rasterizer" (max{G(u),G((x−c)/σ)} = min(rho3d,rho2d),
#   forward.cu:381). El toggle CULL_SUBPIXEL (forward.cu:~234) descarta en el render TODO
#   splat cuya huella geométrica real sea sub-píxel (σ_pantalla < FilterSize = 0.707px), en
#   vez de inflarlo a ~1px a opacidad plena.
#
#   Aquí NO solo re-renderizamos: REENTRENAMOS con el cull activo → el optimizador ya no
#   puede EXPLOTAR el rescate del low-pass (un splat sub-píxel no aporta a la loss → no
#   recibe gradiente de imagen → o crece por encima del píxel o lo recicla el gate
#   dead_sustain). Test más fuerte que el re-render: impide construir el velo desde el origen.
#
#   Backward verificado NaN-safe: preprocessCUDA backward está gateado en radii>0
#   (backward.cu:675) → los splats culleados (radii=0) se saltan también en el backward,
#   sin gradiente ni lectura de basura. Solo los mueven los regularizadores Python
#   (opacity_reg/scale_reg) + ruido/relocate MCMC.
#
# ⚠️ REQUISITO OBLIGATORIO ANTES DE CORRER:
#   Recompilar el rasterizer con CULL_SUBPIXEL=1 (ya está a 1 en forward.cu):
#     docker compose exec surfel_env pip install --force-reinstall --no-deps \
#         /workspace/submodules/diff-surfel-rasterization
#   El cull es COMPILE-TIME (baked en el .so), NO un flag de línea de comandos. Mientras el
#   .so esté compilado con CULL_SUBPIXEL=1, TODAS las runs cullean. Para volver al
#   comportamiento normal: poner CULL_SUBPIXEL 0 en forward.cu y recompilar.
#
# CONFIG = run51 EXACTA (flowers, base run36 + jitter in-plane + cov_noise_normal 0.0):
#   MCMC, cap 7.5M, opacity_reg 0.06, scale_reg 0.01, gate 5, lambda_dist 10, ruido híbrido
#   (--cov_noise) 100% en el plano (--cov_noise_normal 0.0), mcmc_error_weight 3.5,
#   mcmc_jitter_scale 1.5, ruido run9, reset OFF. El jitter in-plane de add_new_gs vive en
#   scene/gaussian_model.py (Python) → asume que sigue activo, igual que run51.
#   ÚNICA diferencia vs run51 = el rasterizer culea los sub-píxel.
#
# ANCLA A BATIR = run51 (MISMA config, low-pass NORMAL, CON velo):
#     run51 = 20.9722 / 0.5930 / 0.3550   (dmean(render−gt) esperado ~+0.6..+1.0, sobre-brillo)
#   Referencias: run36 (opacity_reg 0.06 sin jitter/confinar) = 21.16/0.5926/0.3555;
#                run16 (techo perceptual) = 20.21/0.597/0.339 (dmean +0.999).
#
# CÓMO LEER EL RESULTADO (el observable CLAVE NO es el PSNR):
#   ✅ CONFIRMA la teoría si: dmean(render−gt) CAE hacia 0 (menos sobre-brillo) y el velo /
#      haces de luz del fondo se adelgazan/desaparecen (inspección visual de test/renders).
#   ⚠️ Efecto colateral ESPERADO: el primer plano pierde detalle fino (también tiene
#      sub-píxel) → huecos/aliasing → el PSNR/LPIPS globales pueden EMPEORAR. Eso NO refuta
#      la teoría: el test es sobre el brillo del fondo, no sobre la métrica global.
#   ❌ REFUTA la teoría si: el velo sigue y dmean no baja pese a cullear todos los sub-píxel.
#
# EJECUTAR EN EL SERVIDOR, dentro del contenedor (tras recompilar):
#   docker compose exec surfel_env bash run56_cull_subpixel.sh
# RESUMIBLE: si output/.../run56/results.json ya existe, se salta el entreno.
# TIEMPO: flowers @30k/7.5M ≈ 1–2 h.
# ─────────────────────────────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_NOISE=20
export DEBUG_MEM=1000

# ───────────────────────── BLOQUE EDITABLE ──────────────────────────────────
DATASET=flowers
RUN=56
DEAD_SUSTAIN=5
CAP_MAX=7500000
OPACITY_REG=0.06     # = run51/run36
SCALE_REG=0.01       # = run51/run16
COV_NOISE_NORMAL=0.0 # = run51 (ruido 100% en el plano)
# ─────────────────────────────────────────────────────────────────────────────

MODEL="output/m360/${DATASET}_beta_run${RUN}"
LOG="logs/${DATASET}${RUN}.log"
CSV=logs/cull_subpixel.csv
mkdir -p logs
[ -f "$CSV" ] || echo "run,dataset,cull_subpixel,opacity_reg,cov_noise_normal,cap_max,psnr,ssim,lpips,dmean_render_minus_gt" > "$CSV"

read_metrics () {  # $1 = model dir → "psnr ssim lpips" o vacío
  python - "$1/results.json" <<'PY' 2>/dev/null
import json, sys
try:
    d = json.load(open(sys.argv[1]))["ours_30000"]
    print(f'{d["PSNR"]:.4f} {d["SSIM"]:.4f} {d["LPIPS"]:.4f}')
except Exception:
    pass
PY
}

# Diagnóstico de brillo: media global de (render − gt) sobre las vistas de test.
# >0 = render sobre-brillante (velo). El observable clave de este experimento.
dmean_render_gt () {  # $1 = model dir → dmean o vacío
  python - "$1/test/ours_30000" <<'PY' 2>/dev/null
import sys, os, glob, numpy as np
from PIL import Image
base = sys.argv[1]
rd, gd = os.path.join(base,"renders"), os.path.join(base,"gt")
rs = sorted(glob.glob(os.path.join(rd,"*.png")))
if not rs:
    sys.exit(0)
diffs = []
for r in rs:
    g = os.path.join(gd, os.path.basename(r))
    if not os.path.exists(g):
        continue
    R = np.asarray(Image.open(r).convert("RGB"), np.float32)/255.0
    G = np.asarray(Image.open(g).convert("RGB"), np.float32)/255.0
    if R.shape != G.shape:  # por si el pipeline reescaló
        G = np.asarray(Image.open(g).convert("RGB").resize((R.shape[1],R.shape[0])), np.float32)/255.0
    diffs.append(float((R-G).mean())*255.0)  # en escala 0..255 como el "dmean" del historial
if diffs:
    print(f'{np.mean(diffs):.4f}')
PY
}

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo " run56  CULL_SUBPIXEL=1  (config run51: opacity_reg=${OPACITY_REG}, cov_noise_normal=${COV_NOISE_NORMAL}, jitter IN-PLANE)"
echo " ANCLA run51 (low-pass normal) = 20.9722 / 0.5930 / 0.3550"
echo "════════════════════════════════════════════════════════════════════"

if [ -f "$MODEL/results.json" ]; then
  echo "  [skip] ya existe $MODEL/results.json — no reentreno (borra ese json para forzar)."
else
  # ── 1) TRAIN (= run51 EXACTO; el cull vive en el rasterizer recompilado) ──
  python train.py -s "Datasets/${DATASET}" \
      -m "$MODEL" \
      --eval \
      --densify_mode mcmc \
      --iterations 30000 \
      --test_iterations 7000 15000 20000 25000 30000 \
      --densify_until_iter 25000 \
      --lambda_normal 0.05 \
      --lambda_dist 10 \
      --opacity_reset_interval 1000000000 \
      --cap_max "$CAP_MAX" \
      --noise_lr 3e3 \
      --scale_reg "$SCALE_REG" \
      --opacity_reg "$OPACITY_REG" \
      --opacity_cull 0.01 \
      --floater_cull_dist 0.2 \
      --mcmc_error_weight 3.5 \
      --mcmc_jitter_scale 1.5 \
      --cov_noise \
      --cov_noise_normal "$COV_NOISE_NORMAL" \
      --mcmc_dead_sustain "$DEAD_SUSTAIN" \
      2>&1 | tee "$LOG"

  # ── 2) RENDER (solo test) ──
  python render.py -s "Datasets/${DATASET}" \
      -m "$MODEL" \
      --iteration 30000 \
      --skip_train --skip_mesh \
      2>&1 | tee "logs/${DATASET}${RUN}_test.log"

  # ── 3) METRICS (honesto, genera results.json) ──
  python metrics.py -m "$MODEL" 2>&1 | tee "logs/${DATASET}${RUN}_metrics.log"
fi

# ── 4) Fila al CSV + diagnóstico de brillo ──
M=$(read_metrics "$MODEL")
D=$(dmean_render_gt "$MODEL")
if [ -n "$M" ]; then
  read -r PSNR SSIM LPIPS <<< "$M"
  echo "run${RUN},${DATASET},1,${OPACITY_REG},${COV_NOISE_NORMAL},${CAP_MAX},${PSNR},${SSIM},${LPIPS},${D:-NA}" >> "$CSV"
  echo ""
  echo "  → run${RUN} (CULL_SUBPIXEL)  PSNR=${PSNR}  SSIM=${SSIM}  LPIPS=${LPIPS}  dmean(render−gt)=${D:-NA}"
else
  echo "  !! run${RUN} sin results.json — revisar $LOG"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo " COMPARA CONTRA EL ANCLA (mismo config, low-pass NORMAL):"
echo "   run51 = 20.9722 / 0.5930 / 0.3550   (CON velo, low-pass rescata sub-píxel)"
echo "   run56 = arriba                       (SIN sub-píxel, cull activo)"
echo " CLAVE: ¿dmean(render−gt) BAJA hacia 0?  ¿el velo del fondo desaparece (ver test/renders)?"
echo "        El PSNR/LPIPS pueden empeorar por huecos de primer plano → NO refuta la teoría."
echo "════════════════════════════════════════════════════════════════════"
