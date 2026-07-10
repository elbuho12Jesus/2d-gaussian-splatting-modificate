#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# RUN63 — "run61 ADAPTADO a bonsai" (los deltas NUEVOS de run61 sobre el techo bonsai).
#
# QUÉ ES run61 (flowers): base run56 (MCMC, cap 7.5M, opacity_reg 0.06, gate 5,
# ruido híbrido cov_noise_normal 0.0, jitter in-plane) + rasterizer CULL_SUBPIXEL=1
# + clamp fix (backward.cu:334) + scale_reg 0.06. Honesto: 20.9021/0.5920/0.3535
# (≈ run56 dentro del ruido; scale_reg 0.06 no batió el ancla, 8º fallo del dial).
#
# POR QUÉ NO SE COPIA TAL CUAL A BONSAI (choque con lecciones ya MEDIDAS):
#   · opacity_reg 0.06 ESTORBA en bonsai (interior ACOTADA, sin velo que limpiar →
#     solo vacía detalle). run42 lo usó = 27.54 (−3.82 dB). run44 lo bajó a 0.01
#     = +2.29 dB / +0.0253 SSIM / −0.0329 LPIPS → HITO/techo bonsai. NO transferible.
#   · cap 7.5M SOBRE-DENSIFICA la escena acotada (run42). El cap SÍ es palanca en
#     bonsai (run43): 1.5M (oficial) es el sitio (run44).
#
# POR ESO run63 = BASE CAMPEONA de bonsai (run44: cap 1.5M + opacity_reg 0.01) +
# SOLO los deltas que definen a run61 y que bonsai NUNCA ha probado:
#     (1) rasterizer CULL_SUBPIXEL=1   (compile-time; splats sub-píxel radii=0)
#     (2) clamp fix backward.cu:334    (compile-time; alpha=min(0.99f,opa*kernel))
#     (3) scale_reg 0.01 → 0.06        (delta CLI de run61)
#   + los 2 deltas que run55 ya tenía sobre run44 (para aislar SOLO lo nuevo):
#     (4) --cov_noise --cov_noise_normal 0.0  → ruido MCMC 100% en el plano
#     (5) jitter IN-PLANE de add_new_gs        → activo en el CÓDIGO (Python)
#
# ⇒ run63 = run55 EXACTO + {CULL_SUBPIXEL, clamp fix, scale_reg 0.06}.
#   El A/B LIMPIO de "los deltas nuevos de run61 en bonsai" es vs run55.
#
# ⚠️ ESTADO DEL RASTERIZER COMPILADO (compile-time, baked en el .so):
#   - CULL_SUBPIXEL = 1  (ON, forward.cu:21)
#   - clamp fix       = SÍ (backward.cu:334)
#   El servidor ya quedó compilado así tras run56/run60/run61/run62. Si se recompiló
#   sin ellos, RECOMPILAR ANTES (y verificar CULL_SUBPIXEL 1 en forward.cu):
#     docker compose exec surfel_env pip install --force-reinstall --no-deps \
#         /workspace/submodules/diff-surfel-rasterization
#   El jitter in-plane vive en scene/gaussian_model.py (Python) — asumir activo.
#
# PREGUNTA QUE RESPONDE: en una escena ACOTADA, ¿los deltas nuevos de run61
# (cull sub-píxel + clamp fix + scale_reg 0.06) mejoran sobre el techo de bonsai?
# ANCLAS a batir:
#   run44 (techo bonsai) = 30.67   / 0.9404 / 0.1862   (SIN CULL/clamp, scale_reg 0.01)
#   run55                = 30.2295 / 0.9391 / 0.1871   (= run44 + cov_noise 0.0 + jitter in-plane)
#   original 2DGS bonsai = 31.36   / 0.9359 / 0.2042   (798K splats; honesto)
# HIPÓTESIS: scale_reg falló 8× en flowers y el cull fue neutro (run56); esperado
# ≈ run55/run44, sin palanca nueva. Pero bonsai nunca vio CULL+clamp → merece medirlo.
#
# EJECUTAR EN EL SERVIDOR, dentro del contenedor:
#   docker compose exec surfel_env bash run63_bonsai.sh
# RESUMIBLE: si el run ya tiene results.json, se salta (no reentrena).
# TIEMPO: bonsai @30k / 1.5M ≈ 30–60 min en la Blackwell.
# ─────────────────────────────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_NOISE=20    # estadísticas [NOISE]/[GRAD]/[FLOATER]/[DEADGATE]
export DEBUG_MEM=1000

# ───────────────────────── BLOQUE EDITABLE ──────────────────────────────────
DATASET=bonsai
RUN=63
CAP_MAX=1500000          # oficial bonsai (run44). En bonsai el cap SÍ es palanca.
OPACITY_REG=0.01         # run44 (techo bonsai). 0.06 estorba en escena acotada.
SCALE_REG=0.06           # << DELTA de run61 (era 0.01 en run44/run55).
DEAD_SUSTAIN=5           # óptimo del gate.
COV_NOISE_NORMAL=0.0     # = run55/run51: ruido MCMC 100% en el plano.
MCMC_ERROR_WEIGHT=3.5
MCMC_JITTER_SCALE=1.5    # jitter in-plane (dirección la fija el CÓDIGO de add_new_gs).
NOISE_LR=3e3
LAMBDA_DIST=10
LAMBDA_NORMAL=0.05
ITERATIONS=30000
DENSIFY_UNTIL=25000

MODEL=output/m360/${DATASET}_beta_run${RUN}
LOG=logs/${DATASET}${RUN}.log
CSV=logs/run63_bonsai.csv
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p logs
[ -f "$CSV" ] || echo "run,dataset,cov_noise_normal,opacity_reg,scale_reg,cap_max,psnr,ssim,lpips" > "$CSV"

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
echo " RUN${RUN}  bonsai  =  run61 ADAPTADO (base run44 + CULL + clamp + scale_reg=${SCALE_REG})"
echo "   cap=${CAP_MAX}  opacity_reg=${OPACITY_REG}  cov_noise_normal=${COV_NOISE_NORMAL}  →  ${MODEL}"
echo "════════════════════════════════════════════════════════════════════"

if [ -f "$MODEL/results.json" ]; then
  echo "  [skip] ya existe $MODEL/results.json — no reentreno (borra ese json para forzar)."
else
  # ── 1) TRAIN (base run44 + deltas run55 + scale_reg 0.06; CULL+clamp = rasterizer) ──
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

  # ── 2) RENDER (solo test: lo que metrics.py necesita) ──
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
  echo "run${RUN},${DATASET},${COV_NOISE_NORMAL},${OPACITY_REG},${SCALE_REG},${CAP_MAX},${PSNR},${SSIM},${LPIPS}" >> "$CSV"
  echo "  → run${RUN}  PSNR=${PSNR}  SSIM=${SSIM}  LPIPS=${LPIPS}"
fi

echo ""
echo "──────────────────────────────────────────────────────────────────"
echo " run${RUN} (bonsai) listo."
echo "   A/B LIMPIO de los deltas nuevos vs run55: 30.2295 / 0.9391 / 0.1871"
echo "   ANCLA techo bonsai   run44:               30.67   / 0.9404 / 0.1862"
echo "   ORIGINAL 2DGS bonsai:                     31.36   / 0.9359 / 0.2042"
echo " Resumen en: $CSV"
echo "──────────────────────────────────────────────────────────────────"
