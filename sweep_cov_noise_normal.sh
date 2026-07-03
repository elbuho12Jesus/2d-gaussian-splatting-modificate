#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# BARRIDO de cov_noise_normal (0.0 / 0.2 / 0.5 / 1.0) sobre base JITTER IN-PLANE + opacity_reg 0.06.
#
# QUÉ MIDE: la dirección del RUIDO MCMC posicional (train.py, bloque [NOISE]). El ruido
# híbrido (--cov_noise) mueve cada splat casi-muerto con un elipsoide 3D alineado al marco
# del surfel: 2 ejes EN EL PLANO (∝ escalas reales) + 1 eje en la NORMAL con std =
# cov_noise_normal · media(escalas_plano). El dial cov_noise_normal (=pad) controla cuánto
# se sale el ruido DEL PLANO:
#     pad = 1.0 (default): la normal explora a la tasa media del plano (ruido 3D). ANCLA=run54.
#     pad = 0.5:           media componente normal.
#     pad = 0.2:           casi confinado al plano.
#     pad = 0.0:           ruido 100% EN EL PLANO (rango 2, sin salir del surfel).
#
# HIPÓTESIS: la "pura luz" = velo translúcido de floaters sobre-brillantes en el fondo. Parte
# de esos floaters puede venir de splats empujados FUERA de su plano por la componente normal
# del ruido (pad=1.0 los saca del surfel → flotan). Confinar el ruido al plano (pad↓) podría
# reducir esa fuga y adelgazar el velo AÚN MÁS, encima de la presión L1 de opacity_reg 0.06.
# HIPÓTESIS NULA a batir: la normal del ruido es irrelevante para el velo → pad no mueve las
# métricas y el ancla run54 (pad 1.0) gana.
#
# BASE = run36 (opacity_reg 0.06, la palanca PROBADA del velo → DOMINA al original,
# 21.16/0.5926/0.3555) + JITTER IN-PLANE de add_new_gs (código, scene/gaussian_model.py).
# Resto = run16 EXACTO (MCMC, cap 7.5M, gate 5, lambda_dist 10, scale_reg 0.01,
# mcmc_error_weight 3.5, mcmc_jitter_scale 1.5, ruido run9, reset OFF). ÚNICO dial que se
# mueve = cov_noise_normal.
#
# ANCLA = run54 (pad 1.0) = "run36 + jitter in-plane" (opacity_reg 0.06, scale_reg 0.01,
# jitter, ruido 3D). NOTA: esta config es IDÉNTICA a run46 del sweep_scale_reg_jitter.sh
# (scale_reg=0.01) → si run46 ya está entrenado, puedes copiar su output a run54 y saltarte
# el reentreno (o borrar run54 de VALUES/RUNS y usar run46 como ancla en el resumen).
# ⚠️ run45 NO sirve de ancla aquí: usa opacity_reg 0.01, otra base.
#
# ⚠️ VALIDEZ: el jitter in-plane está en el CÓDIGO (Python puro). Este barrido SOLO es
# coherente mientras ese código siga activo. NO requiere recompilar el rasterizer.
#
# QUÉ VIGILAR EN EL LOG [NOISE]:
#   - normal_std(mean): debe escalar con pad (pad=0 → normal_std=0 exacto).
#   - |disp|(mean/max): con pad↓ el desplazamiento total baja algo (menos componente normal).
#   - activos %: la puerta (1−o)^100 no cambia con pad → % activos ≈ igual entre runs.
#
# Cada valor = una run independiente. Para cada valor:
#   1) train.py → output/m360/${DATASET}_beta_run${RUN}
#   2) render.py (SOLO test) → 3) metrics.py → results.json → 4) fila al CSV.
# Al final imprime la curva completa (incluye run45 como ancla 1.0).
#
# EJECUTAR EN EL SERVIDOR, dentro del contenedor:
#   docker compose exec surfel_env bash sweep_cov_noise_normal.sh
#
# RESUMIBLE: si una run ya tiene results.json, se salta. Borra ese results.json para forzar.
# TIEMPO: cada run flowers @30k/7.5M ≈ 1–2 h → 4 valores ≈ 4–8 h (3 si reusas run46 como run54).
# ─────────────────────────────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_NOISE=20
export DEBUG_MEM=1000

# ───────────────────────── BLOQUE EDITABLE ──────────────────────────────────
DATASET=flowers
DEAD_SUSTAIN=5
CAP_MAX=7500000
OPACITY_REG=0.06         # FIJO = run36 (palanca PROBADA del velo). NO se mueve aquí.
SCALE_REG=0.01           # FIJO = run16. NO se mueve aquí.

# Valores de cov_noise_normal a barrer y la run asignada (índices alineados).
# 1.0 (run54) = ancla = "run36 + jitter in-plane" (config ≡ run46 del sweep_scale_reg_jitter).
# Si run46 ya existe, puedes quitar 1.0/run54 de aquí y usar run46 como ancla en el resumen.
VALUES=(0.0 0.2 0.5 1.0)
RUNS=(51  52  53  54)
# ─────────────────────────────────────────────────────────────────────────────

CSV=logs/sweep_cov_noise_normal.csv
mkdir -p logs
[ -f "$CSV" ] || echo "run,cov_noise_normal,psnr,ssim,lpips" > "$CSV"

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
  PAD="${VALUES[$i]}"
  RUN="${RUNS[$i]}"
  MODEL="output/m360/${DATASET}_beta_run${RUN}"
  LOG="logs/${DATASET}${RUN}.log"

  echo ""
  echo "════════════════════════════════════════════════════════════════════"
  echo " SWEEP  cov_noise_normal=${PAD}  (opacity_reg=${OPACITY_REG}, jitter IN-PLANE)  →  run${RUN}  (${MODEL})"
  echo "════════════════════════════════════════════════════════════════════"

  if [ -f "$MODEL/results.json" ]; then
    echo "  [skip] ya existe $MODEL/results.json — no reentreno."
  else
    # ── 1) TRAIN (base run36 + jitter in-plane, solo cambia cov_noise_normal) ──
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
        --cov_noise_normal "$PAD" \
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

  # ── 4) Fila al CSV ──
  M=$(read_metrics "$MODEL")
  if [ -n "$M" ]; then
    read -r PSNR SSIM LPIPS <<< "$M"
    echo "run${RUN},${PAD},${PSNR},${SSIM},${LPIPS}" >> "$CSV"
    echo "  → run${RUN}  cov_noise_normal=${PAD}  PSNR=${PSNR}  SSIM=${SSIM}  LPIPS=${LPIPS}"
  else
    echo "  !! run${RUN} sin results.json — revisar $LOG"
  fi
done

# ───────────────────── RESUMEN: curva completa de cov_noise_normal ───────────
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo " RESUMEN BARRIDO cov_noise_normal   (opacity_reg 0.06, jitter IN-PLANE)   (PSNR↑ / SSIM↑ / LPIPS↓)"
echo "════════════════════════════════════════════════════════════════════"
printf "%-8s %-16s %8s %8s %8s\n" "run" "cov_noise_norm" "PSNR" "SSIM" "LPIPS"
print_row () {  # $1=etiqueta run  $2=pad  $3=model dir
  local M; M=$(read_metrics "$3")
  if [ -n "$M" ]; then
    read -r P S L <<< "$M"
    printf "%-8s %-16s %8s %8s %8s\n" "$1" "$2" "$P" "$S" "$L"
  fi
}
# Curva en orden ascendente de pad. run54 = ancla (pad 1.0 = ruido 3D, = "run36 + jitter").
# label  cov_noise_normal  model-dir
ROWS=(
  "run51 0.0 output/m360/${DATASET}_beta_run51"
  "run52 0.2 output/m360/${DATASET}_beta_run52"
  "run53 0.5 output/m360/${DATASET}_beta_run53"
  "run54 1.0 output/m360/${DATASET}_beta_run54"
)
for r in "${ROWS[@]}"; do print_row $r; done
echo ""
echo "CSV: $CSV"
echo "Referencia externa: run36 = opacity_reg 0.06 SIN jitter ni confinamiento = 21.16/0.5926/0.3555"
echo "→ comparar run54 vs run36 aísla el jitter in-plane; run51/52/53 vs run54 aísla el pad."
echo "En el log [NOISE] confirmar normal_std↓ con pad↓ (pad=0 → normal_std=0). Añadir filas"
echo "ganadoras al historial (CLAUDE.md) y a docs/comparativa_runs.html."
