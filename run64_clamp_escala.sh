#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run64 — BAJAR EL CLAMP DE ESCALA + SUBIR EL CAP (flowers)
#
# QUÉ PRUEBA (experimento COPLADO, dos deltas por diseño, no confusión):
#   Los "gigantes del fondo" están clavados en el techo del clamp de escala
#   (get_scaling: s <= factor·extent, con factor=0.1 histórico). scale_reg baja la
#   escala MEDIA pero NO mueve a los gigantes topados (mecanismo run21/run41/run61).
#   Este run ATACA a los gigantes DIRECTAMENTE bajando el techo a factor=0.05
#   (techo ≈ 0.24 en vez de ≈ 0.48). Surfels más pequeños → necesitan MÁS para
#   cubrir el fondo rasante → por eso se sube cap 7.5M → 12M (área ∝ escala²).
#   Análisis completo: docs/clamp_escala_gigantes_fondo.html
#
#   Predicción a batir (del doc): bajar el clamp puede cambiar VELO por HUECOS
#   negros (sub-cobertura). Si aparece mejora honesta vs run61 → señal genuina y
#   el clamp deja de ser "dial muerto". Si empeora (huecos / cap saturado) → el
#   clamp queda cerrado y el foco vuelve a orientación + kernel β.
#
# BASE = run61 EXACTO (config), solo se mueven DOS cosas:
#   (1) SCALE_CLAMP_FACTOR  0.1 → 0.05   (env var, se lee en train Y render)
#   (2) CAP_MAX             7.5M → 12M
#   Resto = run61: MCMC, opacity_reg 0.06, scale_reg 0.06, gate 5, lambda_dist 10,
#   ruido híbrido (--cov_noise) 100% en el plano (--cov_noise_normal 0.0),
#   mcmc_error_weight 3.5, mcmc_jitter_scale 1.5, ruido run9, reset OFF,
#   jitter IN-PLANE de add_new_gs (Python → asume activo, igual que run56/run61).
#
# ESTADO DEL RASTERIZER COMPILADO (igual que run61, commit "bug is solved"):
#   - CULL_SUBPIXEL = 1  (ON, forward.cu:21)
#   - clamp fix     = SÍ (backward.cu:334, alpha = min(0.99f, opa*kernel))
#   Ambos COMPILE-TIME (baked en el .so). Este script ASUME que el .so ya está así.
#   Si no, recompilar ANTES:
#     docker compose exec surfel_env pip install --force-reinstall --no-deps \
#         /workspace/submodules/diff-surfel-rasterization
#
# ANCLA a comparar (MISMA config salvo clamp+cap):
#   run61 = 20.9021 / 0.5920 / 0.3535   (clamp 0.1, cap 7.5M)
#   run36 = 21.16   / 0.5926 / 0.3555   (techo PSNR de la familia)
#   run16 = 20.21   / 0.597  / 0.339    (techo perceptual)
#   original 2DGS = 20.89 / 0.556 / 0.402
#
# CÓMO LEER:
#   - Métrica honesta (metrics.py) vs run61.
#   - dmean(render−gt): si SUBE mucho hacia negativo → huecos negros (render más
#     oscuro que GT = sub-cobertura, la predicción pesimista).
#   - DIAGNÓSTICO DE ESCALA (al final): s(max), s(mean), y cuántos surfels quedan
#     pegados al NUEVO techo 0.05·extent vs al VIEJO 0.1·extent. Confirma si el
#     clamp bajó de verdad a los gigantes.
#
# EJECUTAR EN EL SERVIDOR, dentro del contenedor:
#   docker compose exec surfel_env bash run64_clamp_escala.sh
# RESUMIBLE: si ya existe results.json se salta (no reentrena).
# TIEMPO: flowers @30k/12M ≈ 2–3 h (más que run61 por +60% de splats).
# ─────────────────────────────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_NOISE=20
export DEBUG_MEM=1000

# ─── EL DELTA #1: techo del clamp de escala (get_scaling). Se lee en train Y en
#     render mientras esté exportado en este shell (ambos corren aquí). ──────────
export SCALE_CLAMP_FACTOR=0.05

# ───────────────────────── BLOQUE EDITABLE ──────────────────────────────────
DATASET=flowers
RUN=64
DEAD_SUSTAIN=5
CAP_MAX=12000000          # ← EL DELTA #2 (run61 = 7.5M)
OPACITY_REG=0.06         # = run61 (palanca del velo)
SCALE_REG=0.06           # = run61
COV_NOISE_NORMAL=0.0     # = run61 (ruido MCMC 100% en el plano)
# ─────────────────────────────────────────────────────────────────────────────

MODEL="output/m360/${DATASET}_beta_run${RUN}"
LOG="logs/${DATASET}${RUN}.log"
CSV=logs/run64_clamp_escala.csv
mkdir -p logs
[ -f "$CSV" ] || echo "run,scale_clamp_factor,cap_max,opacity_reg,scale_reg,psnr,ssim,lpips,dmean_render_minus_gt,s_max,s_mean,pct_at_new_ceiling" > "$CSV"

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

dmean_render_gt () {  # $1 = model dir → dmean(render−gt) en escala 0..255, o vacío
  python - "$1/test/ours_30000" <<'PY' 2>/dev/null
import sys, os, glob, numpy as np
from PIL import Image
base = sys.argv[1]
rd, gd = os.path.join(base,"renders"), os.path.join(base,"gt")
rs = sorted(glob.glob(os.path.join(rd,"*.png")))
if not rs: sys.exit(0)
diffs = []
for r in rs:
    g = os.path.join(gd, os.path.basename(r))
    if not os.path.exists(g): continue
    R = np.asarray(Image.open(r).convert("RGB"), np.float32)/255.0
    G = np.asarray(Image.open(g).convert("RGB"), np.float32)/255.0
    if R.shape != G.shape:
        G = np.asarray(Image.open(g).convert("RGB").resize((R.shape[1],R.shape[0])), np.float32)/255.0
    diffs.append(float((R-G).mean())*255.0)
if diffs: print(f'{np.mean(diffs):.4f}')
PY
}

# Diagnóstico de escala: lee el ply @30k, aplica exp (scaling_activation) y reporta
# s(max), s(mean) y el % de surfels pegados al techo NUEVO (0.05·extent) vs VIEJO.
# extent≈4.816 en flowers → techo nuevo≈0.2408, viejo≈0.4816.
scale_stats () {  # $1 = model dir → "s_max s_mean pct_at_new_ceiling" o vacío
  python - "$1" <<'PY' 2>/dev/null
import sys, os, glob, numpy as np
from plyfile import PlyData
mdl = sys.argv[1]
plys = glob.glob(os.path.join(mdl, "point_cloud", "iteration_30000", "*.ply"))
if not plys: sys.exit(0)
p = PlyData.read(plys[0])["vertex"]
names = [pr.name for pr in p.properties]
sc = [n for n in names if n.startswith("scale_")]
if not sc: sys.exit(0)
S = np.exp(np.stack([np.asarray(p[n], np.float64) for n in sc], axis=1))  # (N,2) escala real
smax_per = S.max(axis=1)                 # escala máx por surfel
EXTENT = 4.816
new_ceil = 0.05*EXTENT                    # 0.2408
pct = 100.0*float((smax_per >= new_ceil*0.98).mean())  # % topados en el techo nuevo
print(f'{smax_per.max():.4f} {S.mean():.4f} {pct:.2f}')
PY
}

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo " run${RUN}  SCALE_CLAMP_FACTOR=${SCALE_CLAMP_FACTOR}  cap=${CAP_MAX}  (base run61 + clamp↓ + cap↑)"
echo " ANCLA run61 (clamp 0.1, cap 7.5M) = 20.9021 / 0.5920 / 0.3535"
echo "════════════════════════════════════════════════════════════════════"

if [ -f "$MODEL/results.json" ]; then
  echo "  [skip] ya existe $MODEL/results.json — no reentreno (borra ese json para forzar)."
else
  # ── 1) TRAIN (= run61 EXACTO; deltas = SCALE_CLAMP_FACTOR env + cap 12M) ──
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

  # ── 2) RENDER (solo test; SCALE_CLAMP_FACTOR sigue exportado → consistente) ──
  python render.py -s "Datasets/${DATASET}" \
      -m "$MODEL" \
      --iteration 30000 \
      --skip_train --skip_mesh \
      2>&1 | tee "logs/${DATASET}${RUN}_test.log"

  # ── 3) METRICS (honesto, genera results.json) ──
  python metrics.py -m "$MODEL" 2>&1 | tee "logs/${DATASET}${RUN}_metrics.log"
fi

# ── 4) Fila al CSV + diagnósticos ──
M=$(read_metrics "$MODEL")
D=$(dmean_render_gt "$MODEL")
ST=$(scale_stats "$MODEL")
read -r SMAX SMEAN PCT <<< "${ST:-NA NA NA}"
if [ -n "$M" ]; then
  read -r PSNR SSIM LPIPS <<< "$M"
  echo "run${RUN},${SCALE_CLAMP_FACTOR},${CAP_MAX},${OPACITY_REG},${SCALE_REG},${PSNR},${SSIM},${LPIPS},${D:-NA},${SMAX},${SMEAN},${PCT}" >> "$CSV"
else
  echo "  !! run${RUN} sin results.json — revisar $LOG"
fi

# ───────────────────── RESUMEN ──────────────────
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo " RESUMEN run${RUN}   (PSNR↑ / SSIM↑ / LPIPS↓)"
echo "════════════════════════════════════════════════════════════════════"
if [ -n "$M" ]; then
  printf "  run%-4s  PSNR=%-8s SSIM=%-8s LPIPS=%-8s  dmean(render−gt)=%s\n" "$RUN" "$PSNR" "$SSIM" "$LPIPS" "${D:-NA}"
  printf "  escala:  s(max)=%-8s s(mean)=%-8s  %%surfels_en_techo_nuevo(0.05·extent)=%s%%\n" "$SMAX" "$SMEAN" "$PCT"
fi
echo ""
echo "  ANCLA run61 = 20.9021 / 0.5920 / 0.3535   (clamp 0.1, cap 7.5M)"
echo "  run36       = 21.16   / 0.5926 / 0.3555   (techo PSNR)"
echo "  run16       = 20.21   / 0.597  / 0.339    (techo perceptual)"
echo "  original    = 20.89   / 0.556  / 0.402"
echo ""
echo "  LECTURA CLAVE:"
echo "   • s(max) debe BAJAR de ~0.48 (viejo techo) a ~0.24 (nuevo) → los gigantes SÍ encogieron."
echo "   • Si dmean(render−gt) se vuelve MUY negativo → render más oscuro que GT = HUECOS (predicción pesimista)."
echo "   • Si PSNR/SSIM/LPIPS baten a run61 → clamp deja de ser dial muerto (señal genuina)."
echo "  CSV: $CSV"
