#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# BARRIDO de scale_reg (0.01 → 0.06) sobre la base run67 — CAMINO CLÁSICO.
#
# BASE = run67 (mejor clásico del historial: 20.6684 / 0.5811 / 0.3675, train@30k
# 23.63, 4.875M splats) = ancla run26 + FIX DEL TRINQUETE DE β. Config clásica:
# clone/split + opacity_reset 3000, opacity_reg=0, prune_sustain=25,
# opacity_cull=0.005, densify 500→15000, lambda_dist=0, SCALE_CLAMP_FACTOR=0.1.
# El ÚNICO dial que se mueve aquí es scale_reg. Ancla del barrido: run67 = 0.
#
# ═══ POR QUÉ ESTE BARRIDO NO ES "EL 9º FALLO DE scale_reg" ═══
# Los 8 fallos previos del dial son TODOS en MCMC (run21/22/38–41/46–50/61). En
# CLÁSICO scale_reg solo se ha probado DOS veces, y las dos están CONTAMINADAS:
#   · run65 (scale_reg 0.06 + opacity_reg 0.06): colapso a 15.59.
#   · run66 (scale_reg 0.06 + opacity_reg 0.02): colapso a 16.10.
# En ambas, scale_reg 0.06 encogía los surfels (s(mean) 0.0184 / 0.0169) y se leyó
# que "el optimizador aplanaba β para recuperar cobertura" → 55.9% / 48.71% de
# splats con β<0.1. run67 DEMOSTRÓ que esa lectura era incorrecta: el colapso de β
# lo causaba el TRINQUETE de densify_and_split (restaba log(N) → dividía β entre 2
# por generación hasta el suelo del clamp), un mecanismo DETERMINISTA en el que el
# optimizador no intervenía. Con el trinquete arreglado, β<0.1 cae a 0.000%.
#
# ⇒ NO SABEMOS qué hace scale_reg en clásico. Todo lo que creíamos saber (incluido
#   el A/B local que le atribuía −1.68 dB) se midió con el trinquete VIVO, que
#   amplificaba cualquier encogimiento en un colapso de β. Este barrido lo mide por
#   primera vez limpio. Es una pregunta ABIERTA, no una repetición.
#
# ═══ QUÉ SE ESPERA / QUÉ DISCRIMINA ═══
# HIPÓTESIS A (scale_reg era inocente): el daño de run65/66 era ~todo el trinquete
#   (+ los 3 sumideros de opacidad). Sin ellos, la curva será suave y el coste de
#   0.06 mucho menor que 1.68 dB. → scale_reg pasa a ser un dial usable en clásico.
# HIPÓTESIS B (scale_reg era culpable de verdad): incluso limpio, subirlo encoge los
#   surfels, abre HUECOS en el fondo rasante (modo de error propio del clásico) y el
#   PSNR cae monótono. → dial muerto también aquí, pero por fin medido sin confusor.
# El valor 0.06 está en la lista A PROPÓSITO: es exactamente el de run65/66, así que
# su resultado es directamente comparable y cuantifica cuánto de aquel desastre era
# scale_reg y cuánto el trinquete.
#
# ⚠ OJO al leer el mecanismo (por eso el CSV lleva columnas de diagnóstico):
#   · β: si β<0.1 vuelve a dispararse SIN el trinquete, entonces sí existe la vía
#     "encoger → el optimizador aplana β" que run65 postuló. Si se queda en ~0%, esa
#     hipótesis queda descartada del todo y el trinquete era el 100% de la historia.
#   · s(mean) y % topados: si el clamp de escala empieza a morder, ojo — el clásico
#     NO clampa en render (load_ply deja spatial_lr_scale=0) → inconsistencia
#     train↔render ya vista en run64/65 (los "haces de luz").
#   · N splats: el clásico NO tiene cap. Surfels más chicos pueden disparar el
#     clone/split → VIGILAR OOM y disco (run67 = 4.875M y 5.2 GB de salida).
#
# Cada valor = una run independiente. Para cada valor:
#   1) train.py  → output/m360/${DATASET}_beta_run${RUN}
#   2) render.py (SOLO test, sin vídeo de traj ni malla → más rápido)
#   3) metrics.py → results.json (PSNR/SSIM/LPIPS honestos)
#   4) fila al CSV con métricas + DIAGNÓSTICO (β, escala, N, train@30k)
# Al final imprime la curva completa (incluye run67 como ancla 0).
#
# EJECUTAR EN EL SERVIDOR, dentro del contenedor:
#   docker compose exec surfel_env bash sweep_scale_reg_classic.sh
#
# ⚠ El rasterizer del servidor está compilado con CULL_SUBPIXEL=1 (igual que run67).
#   Se mantiene → es común a todo el barrido y al ancla, así que no confunde el A/B.
#
# RESUMIBLE: si una run ya tiene results.json, se salta (no reentrena). Borra ese
# results.json (o el dir entero) para forzar repetición.
#
# TIEMPO/DISCO: cada run flowers @30k ≈ 1–2 h en la Blackwell y ~5 GB de salida
#   → 4 valores ≈ 4–8 h y ~21 GB. Comprobar espacio antes de lanzar.
# ─────────────────────────────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_MEM=1000    # pico de memoria + dev_free (delata zombies/OOM)
export DEBUG_DENSIFY=1   # [DENSIFY]/[RESET] — clave en clásico
# (DEBUG_NOISE NO aplica: el ruido posicional MCMC está OFF en el camino clásico)

# ⚠ run65 se tituló "small clamp" pero NUNCA exportó esta env var → corrió con el
# default sin avisar. Explícita aquí + el print [CLAMP] evita repetir el fallo.
export SCALE_CLAMP_FACTOR=0.1  # 0.1 = default 3DGS/2DGS (= run67)

# ───────────────────────── BLOQUE EDITABLE ──────────────────────────────────
DATASET=flowers

# FIJOS = run67 EXACTO. Nada de esto se mueve en el barrido.
OPACITY_REG=0                 # OFF = run25/26/67. La L1 sin reciclado de masa hunde el clásico (run65/66)
PRUNE_SUSTAIN=25              # = run26/run67 (ancla)
OPACITY_RESET_INTERVAL=3000   # SEGURO en clásico (del 2DGS original)
DENSIFY_FROM=500
DENSIFY_UNTIL=15000
DENSIFICATION_INTERVAL=100
DENSIFY_GRAD_THRESHOLD=0.0002
PERCENT_DENSE=0.01
OPACITY_CULL=0.005            # 2DGS original
LAMBDA_DIST=0                 # receta 2DGS original (el 10 era nuestro; run27 lo descartó en clásico)
LAMBDA_NORMAL=0.05
ITERATIONS=30000

# Valores de scale_reg a barrer y la run asignada (índices alineados).
# 0 ya está medido (run67) → no se repite; el resumen lo recupera como ancla.
# 0.06 = el valor de run65/66, incluido a propósito para cuantificar el confusor.
VALUES=(0.01 0.02 0.03 0.06)
RUNS=(68   69   70   71)
# ─────────────────────────────────────────────────────────────────────────────

CSV=logs/sweep_scale_reg_classic.csv
mkdir -p logs
[ -f "$CSV" ] || echo "run,scale_reg,psnr,ssim,lpips,train30k,n_splats,beta_pct_lt01,beta_min,beta_mean,s_mean,clamp_topados_pct" > "$CSV"

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

# Extrae el DIAGNÓSTICO del log de train (última aparición de cada print).
# Devuelve: "train30k n_splats beta_pct_lt01 beta_min beta_mean s_mean clamp_pct"
# Campos no encontrados salen como "-" (nunca rompe el CSV).
read_diag () {  # $1 = log file
  python - "$1" <<'PY' 2>/dev/null
import re, sys
try:
    txt = open(sys.argv[1], errors="ignore").read()
except Exception:
    print("- - - - - - -"); raise SystemExit

def last(pat, *groups, default="-"):
    m = re.findall(pat, txt)
    if not m:
        return ["-"] * len(groups) if len(groups) > 1 else default
    g = m[-1]
    if isinstance(g, str):
        g = (g,)
    return [g[i] for i in groups] if len(groups) > 1 else g[groups[0]]

# train PSNR @30k = último "Evaluating train: ... PSNR <x>"
train30k = last(r"Evaluating train: L1 [\d.]+ PSNR ([\d.]+)", 0)
if train30k != "-":
    train30k = f"{float(train30k):.2f}"

# [BETA] total=N | beta<0.1: K (P%) | beta min/mean/max = a/b/c
n, pct, bmin, bmean = last(
    r"\[BETA\] total=(\d+) \| beta<0\.1: \d+ \(([\d.]+)%\) \| "
    r"beta min/mean/max = ([\d.]+)/([\d.]+)/[\d.]+",
    0, 1, 2, 3)

# [CLAMP] ... topados: X/Y (P%) ... s_post max/mean=.../<s_mean>
cpct, smean = last(
    r"\[CLAMP\] factor=[\d.]+ techo=[\d.]+ \| topados: \d+/\d+ \(([\d.]+)%\).*?"
    r"s_post max/mean=[\d.]+/([\d.]+)",
    0, 1)

print(" ".join([train30k, n, pct, bmin, bmean, smean, cpct]))
PY
}

for i in "${!VALUES[@]}"; do
  SCALE_REG="${VALUES[$i]}"
  RUN="${RUNS[$i]}"
  MODEL="output/m360/${DATASET}_beta_run${RUN}"
  LOG="logs/${DATASET}${RUN}.log"

  echo ""
  echo "════════════════════════════════════════════════════════════════════"
  echo " SWEEP CLÁSICO  scale_reg=${SCALE_REG}  →  run${RUN}  (${MODEL})"
  echo " base run67 (20.6684/0.5811/0.3675, train 23.63) · opacity_reg=${OPACITY_REG} fijo"
  echo "════════════════════════════════════════════════════════════════════"

  if [ -f "$MODEL/results.json" ]; then
    echo "  [skip] ya existe $MODEL/results.json — no reentreno."
  else
    # ── 1) TRAIN (config = run67 EXACTO, solo cambia scale_reg) ──
    python train.py -s "Datasets/${DATASET}" \
        -m "$MODEL" \
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
        --scale_reg "$SCALE_REG" \
        --classic_prune_sustain $PRUNE_SUSTAIN \
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

  # ── 4) Fila al CSV: métricas + diagnóstico del mecanismo ──
  M=$(read_metrics "$MODEL")
  if [ -n "$M" ]; then
    read -r PSNR SSIM LPIPS <<< "$M"
    D=$(read_diag "$LOG")
    read -r TR30 NSPL BPCT BMIN BMEAN SMEAN CPCT <<< "${D:-- - - - - - -}"
    echo "run${RUN},${SCALE_REG},${PSNR},${SSIM},${LPIPS},${TR30},${NSPL},${BPCT},${BMIN},${BMEAN},${SMEAN},${CPCT}" >> "$CSV"
    echo "  → run${RUN}  scale_reg=${SCALE_REG}  PSNR=${PSNR}  SSIM=${SSIM}  LPIPS=${LPIPS}"
    echo "     train@30k=${TR30}  N=${NSPL}  β<0.1=${BPCT}%  β min=${BMIN}  s(mean)=${SMEAN}  topados=${CPCT}%"
  else
    echo "  !! run${RUN} sin results.json — revisar $LOG"
  fi
done

# ───────────────── RESUMEN: curva completa de scale_reg en CLÁSICO ───────────
echo ""
echo "════════════════════════════════════════════════════════════════════════════════════════"
echo " RESUMEN BARRIDO scale_reg — CLÁSICO (base run67, opacity_reg=0)   PSNR↑ SSIM↑ LPIPS↓"
echo "════════════════════════════════════════════════════════════════════════════════════════"
printf "%-7s %-10s %8s %7s %7s %8s %10s %8s %8s %8s\n" \
       "run" "scale_reg" "PSNR" "SSIM" "LPIPS" "train30k" "N" "b<0.1%" "s(mean)" "topad%"
print_row () {  # $1=run  $2=scale_reg  $3=model dir  $4=log
  local M; M=$(read_metrics "$3")
  [ -z "$M" ] && return
  read -r P S L <<< "$M"
  local D; D=$(read_diag "$4")
  read -r TR NS BP BMI BME SM CP <<< "${D:-- - - - - - -}"
  printf "%-7s %-10s %8s %7s %7s %8s %10s %8s %8s %8s\n" "$1" "$2" "$P" "$S" "$L" "$TR" "$NS" "$BP" "$SM" "$CP"
}
# Curva en orden ascendente. run67 = ancla (scale_reg 0, base del barrido).
# OJO: run65/66 NO son comparables aquí (llevaban opacity_reg alto Y el trinquete vivo).
ROWS=(
  "run67 0.00 output/m360/${DATASET}_beta_run67 logs/${DATASET}67.log"
  "run68 0.01 output/m360/${DATASET}_beta_run68 logs/${DATASET}68.log"
  "run69 0.02 output/m360/${DATASET}_beta_run69 logs/${DATASET}69.log"
  "run70 0.03 output/m360/${DATASET}_beta_run70 logs/${DATASET}70.log"
  "run71 0.06 output/m360/${DATASET}_beta_run71 logs/${DATASET}71.log"
)
for r in "${ROWS[@]}"; do print_row $r; done

cat <<'NOTA'

────────────────────────────────────────────────────────────────────────────────
CÓMO LEER ESTA TABLA (lo importante no es solo el PSNR):

· b<0.1%  → si SIGUE en ~0.000% al subir scale_reg, queda DESCARTADA del todo la
            hipótesis de run65 ("encoger surfels hace que el optimizador aplane β"):
            el trinquete era el 100% de aquel colapso. Si se DISPARA, esa vía existe
            de verdad y hay que documentarla como mecanismo propio de scale_reg.
· s(mean) → confirma que el dial realmente muerde (si no baja, el barrido es un NO-OP
            y ningún resultado concluye nada — mismo fallo que run65 con el clamp).
· topad%  → si sube mucho, el clamp de escala empieza a morder. Ojo: el render NO
            clampa (load_ply deja spatial_lr_scale=0) → inconsistencia train↔render
            ya vista en run64/65 = "haces de luz" en el vídeo.
· N       → el clásico NO tiene cap. Comparar con los 4.875M de run67: si se dispara,
            vigilar OOM/disco; si se hunde, la densificación se está muriendo de
            hambre otra vez (patrón run65/66).
· train30k→ el chivato más rápido de "el modelo ni ajusta el train" (run65 17.48,
            run66 17.74 vs run67 23.63). Si cae de ~22, algo estructural va mal.

ANCLAS: run67 = 20.6684 / 0.5811 / 0.3675 (train 23.63, 4.875M, β<0.1 0.039%).
        2DGS original = 20.89 / 0.556 / 0.402.  run36 (techo global, MCMC) = 21.16 / 0.5926 / 0.3555.
Añadir las filas al historial (CLAUDE.md), historial_runs.csv y docs/comparativa_runs.html.
────────────────────────────────────────────────────────────────────────────────
NOTA
echo "CSV: $CSV"
