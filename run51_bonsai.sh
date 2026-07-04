#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# RUN55 — "receta run51 ARREGLADA para bonsai".
#
# QUÉ ES run51 (flowers): cov_noise_normal=0.0 (ruido MCMC 100% EN EL PLANO del
# surfel) + jitter IN-PLANE de add_new_gs, sobre base run36 (opacity_reg 0.06,
# cap 7.5M). En flowers FALLÓ (20.9722/0.5930/0.3550 < run36): la dirección del
# ruido no es palanca del velo.
#
# POR QUÉ NO SE PUEDE COPIAR TAL CUAL A BONSAI: bonsai es interior ACOTADA (sin
# velo del fondo rasante) y las lecciones ya medidas dicen que la base run36 de
# flowers ROMPE en bonsai:
#   · opacity_reg 0.06 ESTORBA en bonsai (no hay velo que limpiar → solo vacía
#     detalle). run44 probó 0.06→0.01 = +2.29 dB PSNR / +0.0253 SSIM / −0.0329 LPIPS.
#   · cap 7.5M SOBRE-DENSIFICA la escena acotada (run42: −3.82 dB vs original).
#     El cap SÍ es palanca en bonsai (run43): bajar a 1.5M (oficial) recupera.
#
# POR ESO "arreglar run51 para bonsai" = tomar la BASE CAMPEONA de bonsai (run44:
# cap 1.5M + opacity_reg 0.01) y añadirle SOLO los dos deltas experimentales de
# run51 que run44 no tenía:
#     (1) --cov_noise --cov_noise_normal 0.0   → ruido MCMC confinado al plano
#     (2) jitter IN-PLANE de add_new_gs          → ya activo en el CÓDIGO
#                                                   (scene/gaussian_model.py)
# Nota: run44 usó jitter ISÓTROPO (código viejo) + ruido isótropo. Aquí ambos
# (ruido y jitter) quedan confinados al plano del surfel.
#
# PREGUNTA QUE RESPONDE: en una escena ACOTADA (sin velo), ¿confinar ruido+jitter
# al plano del surfel ayuda sobre la mejor config de bonsai (run44)? HIPÓTESIS
# NULA a batir: no mueve las métricas → run44 (30.67/0.9404/0.1862) sigue siendo
# el techo de bonsai. ANCLA de comparación = run44.
#
# ⚠️ VALIDEZ: el jitter in-plane está en el CÓDIGO (Python puro, add_new_gs). Este
# run SOLO es coherente mientras ese código siga activo. NO recompila rasterizer.
#
# baseline original 2DGS bonsai (honesto): 31.36 / 0.9359 / 0.2042 (798K splats).
# cap oficial bonsai = 1.5M. 37 vistas de test. Escena ACOTADA.
#
# EJECUTAR EN EL SERVIDOR, dentro del contenedor:
#   docker compose exec surfel_env bash run51_bonsai.sh
#
# RESUMIBLE: si el run ya tiene results.json, se salta (no reentrena). Borra ese
# results.json (o el dir entero) para forzar repetición.
# TIEMPO: bonsai @30k / 1.5M ≈ 30–60 min en la Blackwell.
# ─────────────────────────────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_NOISE=20    # estadísticas [NOISE]/[GRAD]/[FLOATER]/[DEADGATE]
export DEBUG_MEM=1000

# ───────────────────────── BLOQUE EDITABLE ──────────────────────────────────
DATASET=bonsai
RUN=55
CAP_MAX=1500000          # oficial bonsai (run44). En bonsai el cap SÍ es palanca.
OPACITY_REG=0.01         # ARREGLADO vs run51: 0.06→0.01 (run44; 0.06 estorba en bonsai).
SCALE_REG=0.01           # = run16/run44.
DEAD_SUSTAIN=5           # óptimo del gate.
COV_NOISE_NORMAL=0.0     # << el DELTA de run51: ruido MCMC 100% en el plano.
MCMC_ERROR_WEIGHT=3.5
MCMC_JITTER_SCALE=1.5    # jitter in-plane (dirección la fija el CÓDIGO de add_new_gs).
NOISE_LR=3e3
LAMBDA_DIST=10
LAMBDA_NORMAL=0.05
ITERATIONS=30000
DENSIFY_UNTIL=25000

MODEL=output/m360/${DATASET}_beta_run${RUN}
LOG=logs/${DATASET}${RUN}.log
CSV=logs/run51_bonsai.csv
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p logs
[ -f "$CSV" ] || echo "run,dataset,cov_noise_normal,opacity_reg,cap_max,psnr,ssim,lpips" > "$CSV"

read_metrics () {  # $1 = model dir → "psnr ssim lpips" (clave ours_30000). Vacío si falta.
  python - "$1/results.json" <<'PY' 2>/dev/null
import json, sys
try:
    d = json.load(open(sys.argv[1]))["ours_30000"]
    print(f'{d["PSNR"]:.4f} {d["SSIM"]:.4f} {d["LPIPS"]:.4f}')
except Exception:
    pass
PY
}

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo " RUN${RUN}  bonsai  =  run51 ARREGLADO (base run44 + cov_noise_normal=${COV_NOISE_NORMAL} + jitter in-plane)"
echo "   cap=${CAP_MAX}  opacity_reg=${OPACITY_REG}  →  ${MODEL}"
echo "════════════════════════════════════════════════════════════════════"

if [ -f "$MODEL/results.json" ]; then
  echo "  [skip] ya existe $MODEL/results.json — no reentreno."
else
  # ── 1) TRAIN (base run44 + los dos deltas de run51) ──
  python train.py -s "Datasets/${DATASET}" \
      -m "$MODEL" \
      --eval \
      --densify_mode mcmc \
      --iterations "$ITERATIONS" \
      --test_iterations 7000 15000 20000 25000 30000 \
      --densify_until_iter "$DENSIFY_UNTIL" \
      --lambda_normal "$LAMBDA_NORMAL" \
      --lambda_dist "$LAMBDA_DIST" \
      --opacity_reset_interval 1000000000 \
      --cap_max "$CAP_MAX" \
      --noise_lr "$NOISE_LR" \
      --scale_reg "$SCALE_REG" \
      --opacity_reg "$OPACITY_REG" \
      --opacity_cull 0.01 \
      --floater_cull_dist 0.2 \
      --mcmc_error_weight "$MCMC_ERROR_WEIGHT" \
      --mcmc_jitter_scale "$MCMC_JITTER_SCALE" \
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

# ── 4) Fila al CSV resumen ──
M=$(read_metrics "$MODEL")
if [ -n "$M" ]; then
  read -r PSNR SSIM LPIPS <<< "$M"
  echo "run${RUN},${DATASET},${COV_NOISE_NORMAL},${OPACITY_REG},${CAP_MAX},${PSNR},${SSIM},${LPIPS}" >> "$CSV"
fi

echo ""
echo "──────────────────────────────────────────────────────────────────"
echo " run${RUN} (bonsai) listo. Comparar vs ANCLA run44: 30.67 / 0.9404 / 0.1862"
echo "                          y vs ORIGINAL 2DGS bonsai: 31.36 / 0.9359 / 0.2042"
echo " Resumen en: $CSV"
echo "──────────────────────────────────────────────────────────────────"
