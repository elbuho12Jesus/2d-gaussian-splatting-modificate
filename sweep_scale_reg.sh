#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# BARRIDO AUTOMÁTICO de scale_reg (0.02 → 0.10) sobre la base run36.
#
# BASE = run36 (HITO): opacity_reg=0.06 (óptimo de PSNR del barrido anterior, que
# DOMINA al original en las 3 métricas → 21.16/0.5926/0.3555), todo lo demás = run16
# EXACTO (MCMC, cap 7.5M, gate 5, lambda_dist 10, ruido run9, reset OFF). El ÚNICO
# dial que se mueve aquí es scale_reg. Ancla del barrido: run36 = scale_reg 0.01.
#
# POR QUÉ RE-BARRER scale_reg (ya falló en la base vieja):
#   - run21 (scale_reg 0.05, base opacity_reg 0.01): FALLÓ. s(mean) −16% pero
#     s(max) CLAVADO en extent=4.816 → los surfels GIGANTES del fondo NO encogieron
#     (topados en el clamp del rasterizer) → fondo NO cambió.
#   - run22 (scale_reg 0.03 + dead_sustain 15): PEOR de su tanda + más velo.
#   NOVEDAD: la base run36 ya adelgazó el velo translúcido (opacity_reg 0.06,
#   dmean render−gt +0.999→ hacia +0.6). Con el velo más fino, la pregunta legítima
#   es si scale_reg rinde DISTINTO ahora (¿complementa la presión L1 de opacidad?).
#   HIPÓTESIS NULA a batir: el clamp de escala sigue topando los gigantes → scale_reg
#   seguirá sin tocar el fondo y solo redistribuirá el primer plano.
#
# Cada valor = una run independiente. Para cada valor:
#   1) train.py  → output/m360/${DATASET}_beta_run${RUN}
#   2) render.py (SOLO test, sin vídeo de traj ni malla → más rápido)
#   3) metrics.py → results.json (PSNR/SSIM/LPIPS honestos)
#   4) fila al CSV resumen logs/sweep_scale_reg.csv
# Al final imprime la curva completa de scale_reg (incluye run36 como ancla 0.01).
#
# EJECUTAR EN EL SERVIDOR, dentro del contenedor:
#   docker compose exec surfel_env bash sweep_scale_reg.sh
#
# RESUMIBLE: si una run ya tiene results.json, se salta (no reentrena). Borra ese
# results.json (o el dir entero) para forzar repetición.
#
# TIEMPO: cada run flowers @30k/7.5M ≈ 1–2 h en la Blackwell → 4 valores ≈ 4–8 h.
# ─────────────────────────────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_NOISE=20
export DEBUG_MEM=1000

# ───────────────────────── BLOQUE EDITABLE ──────────────────────────────────
DATASET=flowers
DEAD_SUSTAIN=5
CAP_MAX=7500000
OPACITY_REG=0.06         # FIJO = run36 (óptimo de PSNR; domina al original). NO se mueve aquí.

# Valores de scale_reg a barrer y la run asignada a cada uno (índices alineados).
# 0.01 ya está medido (run36) → no se repite; el resumen final lo recupera de su
# results.json como ancla del barrido.
VALUES=(0.02 0.03 0.05 0.10)
RUNS=(38   39   40   41)
# ─────────────────────────────────────────────────────────────────────────────

CSV=logs/sweep_scale_reg.csv
mkdir -p logs
[ -f "$CSV" ] || echo "run,scale_reg,psnr,ssim,lpips" > "$CSV"

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

for i in "${!VALUES[@]}"; do
  SCALE_REG="${VALUES[$i]}"
  RUN="${RUNS[$i]}"
  MODEL="output/m360/${DATASET}_beta_run${RUN}"
  LOG="logs/${DATASET}${RUN}.log"

  echo ""
  echo "════════════════════════════════════════════════════════════════════"
  echo " SWEEP  scale_reg=${SCALE_REG}  (opacity_reg=${OPACITY_REG} fijo)  →  run${RUN}  (${MODEL})"
  echo "════════════════════════════════════════════════════════════════════"

  if [ -f "$MODEL/results.json" ]; then
    echo "  [skip] ya existe $MODEL/results.json — no reentreno."
  else
    # ── 1) TRAIN (config = run36 EXACTO, solo cambia scale_reg) ──
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
        --cov_noise_normal 1.0 \
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

  # ── 4) Fila al CSV ──
  M=$(read_metrics "$MODEL")
  if [ -n "$M" ]; then
    read -r PSNR SSIM LPIPS <<< "$M"
    echo "run${RUN},${SCALE_REG},${PSNR},${SSIM},${LPIPS}" >> "$CSV"
    echo "  → run${RUN}  scale_reg=${SCALE_REG}  PSNR=${PSNR}  SSIM=${SSIM}  LPIPS=${LPIPS}"
  else
    echo "  !! run${RUN} sin results.json — revisar $LOG"
  fi
done

# ───────────────────── RESUMEN: curva completa de scale_reg ──────────────────
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo " RESUMEN BARRIDO scale_reg   (opacity_reg=0.06 fijo)   (PSNR↑ / SSIM↑ / LPIPS↓)"
echo "════════════════════════════════════════════════════════════════════"
printf "%-8s %-10s %8s %8s %8s\n" "run" "scale_reg" "PSNR" "SSIM" "LPIPS"
print_row () {  # $1=etiqueta run  $2=scale_reg  $3=model dir
  local M; M=$(read_metrics "$3")
  if [ -n "$M" ]; then
    read -r P S L <<< "$M"
    printf "%-8s %-10s %8s %8s %8s\n" "$1" "$2" "$P" "$S" "$L"
  fi
}
# Curva en orden ascendente de scale_reg. run36 = ancla (0.01, base del barrido).
# OJO: run21/run22 NO son comparables aquí (base opacity_reg 0.01, no 0.06) → no se listan.
# label  scale_reg  model-dir
ROWS=(
  "run36 0.01 output/m360/${DATASET}_beta_run36"
  "run38 0.02 output/m360/${DATASET}_beta_run38"
  "run39 0.03 output/m360/${DATASET}_beta_run39"
  "run40 0.05 output/m360/${DATASET}_beta_run40"
  "run41 0.10 output/m360/${DATASET}_beta_run41"
)
for r in "${ROWS[@]}"; do print_row $r; done
echo ""
echo "CSV: $CSV"
echo "Recuerda: en el log vigilar s(mean) y sobre todo s(max) — si sigue CLAVADO en"
echo "extent (≈4.816), el clamp del rasterizer vuelve a topar los gigantes (mecanismo"
echo "run21) y scale_reg no toca el fondo. Añadir filas ganadoras al historial (CLAUDE.md)"
echo "y a docs/comparativa_runs.html."
