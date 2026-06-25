#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# BARRIDO AUTOMÁTICO de opacity_reg (0.03 → 0.07) sobre la base run16/run32.
# Localiza el óptimo del trade-off PSNR↔perceptual que abrió run32 (opacity_reg
# 0.05 → 20.62/0.5945/0.3525, primer dial que sube PSNR; mecanismo confirmado:
# dmean(render−gt) +0.999→+0.627 = adelgaza el velo translúcido).
#
# Cada valor = una run independiente. Todo lo demás = run16 EXACTO (MCMC, cap 7.5M,
# gate 5, lambda_dist 10, scale_reg 0.01, ruido run9, reset OFF). Para cada valor:
#   1) train.py  → output/m360/${DATASET}_beta_run${RUN}
#   2) render.py (SOLO test, sin vídeo de traj ni malla → más rápido)
#   3) metrics.py → results.json (PSNR/SSIM/LPIPS honestos)
#   4) fila al CSV resumen logs/sweep_opacity_reg.csv
# Al final imprime la curva completa 0.01→0.07 (incluye run16/run20/run32 si existen).
#
# EJECUTAR EN EL SERVIDOR, dentro del contenedor:
#   docker compose exec surfel_env bash sweep_opacity_reg.sh
# (o desde una shell interactiva del contenedor: bash sweep_opacity_reg.sh)
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
SCALE_REG=0.01

# Valores de opacity_reg a barrer y la run asignada a cada uno (índices alineados).
# 0.05 ya está medido (run32) → no se repite; el resumen final lo recupera de su
# results.json. Si quieres re-medirlo, añade  0.05  con su propia run.
VALUES=(0.03 0.04 0.06 0.07)
RUNS=(33   34   36   37)
# ─────────────────────────────────────────────────────────────────────────────

CSV=logs/sweep_opacity_reg.csv
mkdir -p logs
[ -f "$CSV" ] || echo "run,opacity_reg,psnr,ssim,lpips" > "$CSV"

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
  OPACITY_REG="${VALUES[$i]}"
  RUN="${RUNS[$i]}"
  MODEL="output/m360/${DATASET}_beta_run${RUN}"
  LOG="logs/${DATASET}${RUN}.log"

  echo ""
  echo "════════════════════════════════════════════════════════════════════"
  echo " SWEEP  opacity_reg=${OPACITY_REG}  →  run${RUN}  (${MODEL})"
  echo "════════════════════════════════════════════════════════════════════"

  if [ -f "$MODEL/results.json" ]; then
    echo "  [skip] ya existe $MODEL/results.json — no reentreno."
  else
    # ── 1) TRAIN (config = run16 EXACTO, solo cambia opacity_reg) ──
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
    echo "run${RUN},${OPACITY_REG},${PSNR},${SSIM},${LPIPS}" >> "$CSV"
    echo "  → run${RUN}  opacity_reg=${OPACITY_REG}  PSNR=${PSNR}  SSIM=${SSIM}  LPIPS=${LPIPS}"
  else
    echo "  !! run${RUN} sin results.json — revisar $LOG"
  fi
done

# ───────────────────── RESUMEN: curva completa 0.01 → 0.07 ───────────────────
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo " RESUMEN BARRIDO opacity_reg   (PSNR↑ / SSIM↑ / LPIPS↓)"
echo "════════════════════════════════════════════════════════════════════"
printf "%-8s %-12s %8s %8s %8s\n" "run" "opacity_reg" "PSNR" "SSIM" "LPIPS"
print_row () {  # $1=etiqueta run  $2=opacity_reg  $3=model dir
  local M; M=$(read_metrics "$3")
  if [ -n "$M" ]; then
    read -r P S L <<< "$M"
    printf "%-8s %-12s %8s %8s %8s\n" "$1" "$2" "$P" "$S" "$L"
  fi
}
# Curva completa en orden ascendente de opacity_reg (anclas conocidas + barrido).
# label  opacity_reg  model-dir
ROWS=(
  "run16 0.01 output/m360/${DATASET}_beta_run16"
  "run20 0.02 output/m360/${DATASET}_beta_run20"
  "run33 0.03 output/m360/${DATASET}_beta_run33"
  "run34 0.04 output/m360/${DATASET}_beta_run34"
  "run32 0.05 output/m360/${DATASET}_beta_run32"
  "run36 0.06 output/m360/${DATASET}_beta_run36"
  "run37 0.07 output/m360/${DATASET}_beta_run37"
)
for r in "${ROWS[@]}"; do print_row $r; done
echo ""
echo "CSV: $CSV"
echo "Recuerda: añadir las filas ganadoras al historial (CLAUDE.md) y a docs/comparativa_runs.html."
