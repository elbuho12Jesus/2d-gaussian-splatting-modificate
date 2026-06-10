export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ───────────────────────────────────────────────────────────────────────────
# ÚNICO bloque a editar entre runs. Deben coincidir con train_server.sh.
DATASET=flowers          # carpeta en Datasets/
RUN=11                    # número de run
ITER=50000               # iteración (checkpoint) a renderizar

MODEL=output/m360/${DATASET}_beta_run${RUN}
# ───────────────────────────────────────────────────────────────────────────

# 1) Vídeo de trayectoria (vistas nuevas interpoladas).
# --skip_train --skip_test --skip_mesh + --render_path => SOLO genera el vídeo,
# sin re-exportar train/test ni extraer malla (marching cubes).
# Salida: $MODEL/traj/ours_50000/render_traj_color.mp4
python render.py -s Datasets/${DATASET} \
    -m $MODEL \
    --iteration $ITER \
    --skip_train --skip_test --skip_mesh \
    --render_path \
    2>&1 | tee logs/${DATASET}${RUN}_render.log

# 2) Comparativas render|GT lado a lado de las vistas de test.
# --skip_train --skip_mesh => exporta SOLO test (sin vídeo ni malla).
# Salida: $MODEL/test/ours_50000/vis/  (render|GT) y .../renders, .../gt
# Sirve para separar hueco real en vista observada vs artefacto de extrapolación.
python render.py -s Datasets/${DATASET} \
    -m $MODEL \
    --iteration $ITER \
    --skip_train --skip_mesh \
    2>&1 | tee logs/${DATASET}${RUN}_test.log
