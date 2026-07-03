#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# BARRIDO AUTOMÁTICO de scale_reg (0.01 → 0.10) sobre la base JITTER IN-PLANE + opacity_reg 0.06.
#
# BASE = run36 (HITO: opacity_reg=0.06, la palanca PROBADA contra el velo translúcido —
# adelgazó el sobre-brillo dmean +0.999→+0.627 y DOMINA al original en las 3 métricas,
# 21.16/0.5926/0.3555) + el JITTER IN-PLANE nuevo de add_new_gs (scene/gaussian_model.py:
# el clon se desplaza por el EJE IN-PLANE DOMINANTE del surfel, estilo densify_and_split,
# en vez de en 3D isótropo). Resto = run16 EXACTO (MCMC, cap 7.5M, gate 5, lambda_dist 10,
# mcmc_error_weight 3.5, mcmc_jitter_scale 1.5, ruido run9, reset OFF). ÚNICO dial que se
# mueve en el barrido = scale_reg.
#
# POR QUÉ opacity_reg=0.06 (y no 0.01): la "pura luz" que se ve en run45 = velo translúcido
# sobre-brillante. La palanca que lo ADELGAZA es opacity_reg (L1 sobre opacidad → empuja
# opacidad ↓ → poda floaters del velo). scale_reg NO lo toca (los gigantes del fondo están
# topados en el clamp del rasterizer). Aquí combinamos la palanca del velo (opacity_reg 0.06)
# con la dirección nueva del jitter, y barremos scale_reg encima.
#
# PUNTO CLAVE del barrido: scale_reg=0.01 (run46) = "run36 + jitter in-plane" → test directo
# de si la dirección in-plane del jitter mejora el HITO run36 (21.16/0.5926/0.3555). Ese es
# el ancla real de este barrido (run45 NO sirve de ancla: usa opacity_reg 0.01, otra base).
#
# ⚠️ VALIDEZ: el jitter in-plane está en el CÓDIGO (Python puro, add_new_gs). Este barrido
# SOLO es coherente mientras ese código siga activo. Si se revierte el jitter a 3D isótropo,
# estas runs dejan de compartir base. NO requiere recompilar el rasterizer.
#
# POR QUÉ RE-BARRER scale_reg (ha FALLADO 6 veces):
#   - run21 (scale_reg 0.05, base run16): s(mean) −16% pero s(max) CLAVADO en el clamp del
#     rasterizer → los surfels GIGANTES del fondo NO encogieron → fondo NO cambió.
#   - run22 (scale_reg 0.03 + dead_sustain 15): PEOR de su tanda + más velo.
#   - run38/39/40/41 (scale_reg 0.02/0.03/0.05/0.10, base run36 SIN jitter): NINGUNO superó
#     al ancla; mismo mecanismo (s(max) topado, scale_reg solo muerde el primer plano).
#   NOVEDAD aquí: base = opacity_reg 0.06 (velo YA adelgazado) + jitter IN-PLANE (clones
#   sobre el plano del surfel). La pregunta: ¿con el velo ya fino Y los clones en-plano,
#   scale_reg redistribuye distinto y complementa, o vuelve a toparse en el clamp igual?
#   HIPÓTESIS NULA a batir: s(max) seguirá clavado en el clamp → scale_reg no tocará el fondo
#   y solo redistribuirá el primer plano (enésimo cheap-sweep fallido). Nota: run38-41 ya
#   fallaron con opacity_reg 0.06 SIN jitter; lo único nuevo que se prueba aquí es el jitter.
#
# Cada valor = una run independiente. Para cada valor:
#   1) train.py  → output/m360/${DATASET}_beta_run${RUN}
#   2) render.py (SOLO test, sin vídeo de traj ni malla → más rápido)
#   3) metrics.py → results.json (PSNR/SSIM/LPIPS honestos)
#   4) fila al CSV resumen logs/sweep_scale_reg_jitter.csv
# Al final imprime la curva completa de scale_reg (incluye run45 como ancla 0.01).
#
# EJECUTAR EN EL SERVIDOR, dentro del contenedor:
#   docker compose exec surfel_env bash sweep_scale_reg_jitter.sh
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
OPACITY_REG=0.06         # FIJO = run36 (palanca PROBADA contra el velo). NO se mueve aquí.

# Valores de scale_reg a barrer y la run asignada a cada uno (índices alineados).
# 0.01 (run46) = ancla real = "run36 + jitter in-plane" (test directo vs el HITO run36).
# run45 NO sirve de ancla aquí (usa opacity_reg 0.01, base distinta).
VALUES=(0.01 0.02 0.03 0.05 0.10)
RUNS=(46   47   48   49   50)
# ─────────────────────────────────────────────────────────────────────────────

CSV=logs/sweep_scale_reg_jitter.csv
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
  echo " SWEEP  scale_reg=${SCALE_REG}  (opacity_reg=${OPACITY_REG} fijo, jitter IN-PLANE)  →  run${RUN}  (${MODEL})"
  echo "════════════════════════════════════════════════════════════════════"

  if [ -f "$MODEL/results.json" ]; then
    echo "  [skip] ya existe $MODEL/results.json — no reentreno."
  else
    # ── 1) TRAIN (base run36 + jitter in-plane, solo cambia scale_reg; jitter = código) ──
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
echo " RESUMEN BARRIDO scale_reg (JITTER IN-PLANE)   (opacity_reg=0.06 fijo)   (PSNR↑ / SSIM↑ / LPIPS↓)"
echo "════════════════════════════════════════════════════════════════════"
printf "%-8s %-10s %8s %8s %8s\n" "run" "scale_reg" "PSNR" "SSIM" "LPIPS"
print_row () {  # $1=etiqueta run  $2=scale_reg  $3=model dir
  local M; M=$(read_metrics "$3")
  if [ -n "$M" ]; then
    read -r P S L <<< "$M"
    printf "%-8s %-10s %8s %8s %8s\n" "$1" "$2" "$P" "$S" "$L"
  fi
}
# Curva en orden ascendente de scale_reg. run46 = ancla (0.01 = "run36 + jitter in-plane").
# Referencia externa (SIN jitter, no en la tabla): run36 = 0.01 = 21.16/0.5926/0.3555.
# OJO: run38-41 NO son comparables aquí (base sin jitter in-plane) → no se listan.
# label  scale_reg  model-dir
ROWS=(
  "run46 0.01 output/m360/${DATASET}_beta_run46"
  "run47 0.02 output/m360/${DATASET}_beta_run47"
  "run48 0.03 output/m360/${DATASET}_beta_run48"
  "run49 0.05 output/m360/${DATASET}_beta_run49"
  "run50 0.10 output/m360/${DATASET}_beta_run50"
)
for r in "${ROWS[@]}"; do print_row $r; done
echo ""
echo "CSV: $CSV"
echo "Recuerda: en el log vigilar s(mean) y sobre todo s(max) — si sigue CLAVADO en"
echo "el clamp (≈4.816e-01 en run45), el rasterizer vuelve a topar los gigantes (mecanismo"
echo "run21) y scale_reg no toca el fondo ni la 'pura luz'. Añadir filas ganadoras al"
echo "historial (CLAUDE.md) y a docs/comparativa_runs.html."
