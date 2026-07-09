#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run61 — A/B del CLAMP FIX a cap 7.5M + BARRIDO scale_reg = [0.06, 0.01] (flowers)
#
# QUÉ PRUEBA (dos cosas de una sola tanda):
#   1) ¿El clamp fix del backward (forward↔backward consistente, backward.cu:334
#      `alpha = min(0.99f, opa*kernel)`) también MEJORA a cap 7.5M, como mejoró
#      marginalmente a cap 9M (run60: +0.10 PSNR / +0.0005 SSIM / +0.0006 LPIPS vs
#      run59, sin regresión)? El punto scale_reg=0.01 de este barrido es el A/B
#      LIMPIO: = run56 EXACTO (20.9453/0.5932/0.3552) + el clamp fix. Único delta.
#   2) ¿Mueve algo scale_reg=0.06 sobre esta base? (2º punto del dial; el resto del
#      historial dice que scale_reg falla 7×, pero se prueba gratis en la misma tanda.)
#
# ESTADO DEL RASTERIZER COMPILADO (commit "bug is solved"):
#   - CULL_SUBPIXEL = 1  (ON, forward.cu:21 — igual que run56)
#   - clamp fix       = SÍ (backward.cu:334) ← el delta nuevo vs run56
#   Ambos son COMPILE-TIME (baked en el .so). Este script asume que el .so YA está
#   compilado así en el servidor. Si no, recompilar ANTES:
#     docker compose exec surfel_env pip install --force-reinstall --no-deps \
#         /workspace/submodules/diff-surfel-rasterization
#
# CONFIG = run56 EXACTA (= run51 + CULL), solo se mueve scale_reg:
#   MCMC, cap 7.5M, opacity_reg 0.06, gate 5, lambda_dist 10, ruido híbrido
#   (--cov_noise) 100% en el plano (--cov_noise_normal 0.0), mcmc_error_weight 3.5,
#   mcmc_jitter_scale 1.5, ruido run9, reset OFF, jitter IN-PLANE de add_new_gs
#   (scene/gaussian_model.py, Python → asume activo, igual que run51/run56).
#
# ANCLAS a comparar:
#   run56 = 20.9453 / 0.5932 / 0.3552   (MISMA config, scale_reg 0.01, SIN clamp fix, CON CULL)
#   run36 = 21.16   / 0.5926 / 0.3555   (opacity_reg 0.06, sin jitter/CULL/clamp — techo PSNR)
#   run60 = 20.902  / 0.5934 / 0.3521   (clamp fix pero a cap 9M — subóptimo de cap)
#
# CÓMO LEER:
#   - run61 (scale_reg 0.01) vs run56 → aísla el clamp fix a 7.5M. Esperado: neutro-a-
#     levemente-positivo (fix de CORRECCIÓN, no palanca; el impacto práctico es pequeño).
#   - run62 (scale_reg 0.06) vs run61 → ¿scale_reg alto ayuda con velo ya adelgazado?
#     Vigilar s(max) en el log: si sigue CLAVADO en el clamp (≈4.816e-01=extent), el
#     rasterizer vuelve a topar los gigantes del fondo (mecanismo run21) → dial muerto.
#
# EJECUTAR EN EL SERVIDOR, dentro del contenedor:
#   docker compose exec surfel_env bash run61_scale_reg.sh
# RESUMIBLE: si una run ya tiene results.json, se salta (no reentrena).
# TIEMPO: cada run flowers @30k/7.5M ≈ 1–2 h → 2 valores ≈ 2–4 h.
# ─────────────────────────────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_NOISE=20
export DEBUG_MEM=1000

# ───────────────────────── BLOQUE EDITABLE ──────────────────────────────────
DATASET=flowers
DEAD_SUSTAIN=5
CAP_MAX=7500000
OPACITY_REG=0.06         # FIJO = run56/run36 (palanca del velo). NO se mueve aquí.
COV_NOISE_NORMAL=0.0     # = run56/run51 (ruido MCMC 100% en el plano)

# Valores de scale_reg a barrer y la run asignada a cada uno (índices alineados).
# 0.01 (run62) = A/B LIMPIO del clamp fix vs run56 (misma config, +clamp).
VALUES=(0.06 0.01)
RUNS=(61   62)
# ─────────────────────────────────────────────────────────────────────────────

CSV=logs/run61_scale_reg.csv
mkdir -p logs
[ -f "$CSV" ] || echo "run,scale_reg,opacity_reg,cap_max,psnr,ssim,lpips,dmean_render_minus_gt" > "$CSV"

# Extrae "psnr ssim lpips" de un results.json (clave ours_30000). Vacío si falta.
read_metrics () {  # $1 = model dir
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
# >0 = render sobre-brillante (velo). Escala 0..255 como el "dmean" del historial.
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
    if R.shape != G.shape:
        G = np.asarray(Image.open(g).convert("RGB").resize((R.shape[1],R.shape[0])), np.float32)/255.0
    diffs.append(float((R-G).mean())*255.0)
if diffs:
    print(f'{np.mean(diffs):.4f}')
PY
}

for i in "${!VALUES[@]}"; do
  SCALE_REG="${VALUES[$i]}"
  RUN="${RUNS[$i]}"
  MODEL="output/m360/${DATASET}_beta_run${RUN}"
  LOG="logs/${DATASET}${RUN}.log"

  echo ""
  echo "════════════════════════════════════════════════════════════════════"
  echo " run${RUN}  scale_reg=${SCALE_REG}  (opacity_reg=${OPACITY_REG}, cap ${CAP_MAX}, CULL+clamp fix, jitter IN-PLANE)"
  echo " ANCLA run56 (scale_reg 0.01, SIN clamp) = 20.9453 / 0.5932 / 0.3552"
  echo "════════════════════════════════════════════════════════════════════"

  if [ -f "$MODEL/results.json" ]; then
    echo "  [skip] ya existe $MODEL/results.json — no reentreno (borra ese json para forzar)."
  else
    # ── 1) TRAIN (= run56 EXACTO; solo cambia scale_reg; CULL+clamp = rasterizer) ──
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

    # ── 2) RENDER (solo test: lo que metrics.py necesita; sin vídeo ni malla) ──
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
    echo "run${RUN},${SCALE_REG},${OPACITY_REG},${CAP_MAX},${PSNR},${SSIM},${LPIPS},${D:-NA}" >> "$CSV"
    echo "  → run${RUN}  scale_reg=${SCALE_REG}  PSNR=${PSNR}  SSIM=${SSIM}  LPIPS=${LPIPS}  dmean(render−gt)=${D:-NA}"
  else
    echo "  !! run${RUN} sin results.json — revisar $LOG"
  fi
done

# ───────────────────── RESUMEN ──────────────────
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo " RESUMEN run61 (clamp fix @7.5M + scale_reg [0.06, 0.01])   (PSNR↑ / SSIM↑ / LPIPS↓)"
echo "════════════════════════════════════════════════════════════════════"
printf "%-8s %-10s %8s %8s %8s\n" "run" "scale_reg" "PSNR" "SSIM" "LPIPS"
print_row () {  # $1=etiqueta  $2=scale_reg  $3=model dir
  local M; M=$(read_metrics "$3")
  if [ -n "$M" ]; then
    read -r P S L <<< "$M"
    printf "%-8s %-10s %8s %8s %8s\n" "$1" "$2" "$P" "$S" "$L"
  fi
}
ROWS=(
  "run62 0.01 output/m360/${DATASET}_beta_run62"
  "run61 0.06 output/m360/${DATASET}_beta_run61"
)
for r in "${ROWS[@]}"; do print_row $r; done
echo ""
echo "Referencias (NO en la tabla):"
echo "  run56 = 20.9453 / 0.5932 / 0.3552  (scale_reg 0.01, SIN clamp fix)  ← A/B de run62"
echo "  run36 = 21.16   / 0.5926 / 0.3555  (techo PSNR, sin jitter/CULL/clamp)"
echo "  run60 = 20.902  / 0.5934 / 0.3521  (clamp fix pero a cap 9M)"
echo "CLAVE run62 vs run56: aísla el clamp fix a 7.5M (esperado neutro-a-leve-positivo)."
echo "CLAVE run61: vigilar s(max) en el log — si sigue CLAVADO en ≈4.816e-01 el dial muere (run21)."
echo "CSV: $CSV"
