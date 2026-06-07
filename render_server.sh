export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Iteración a renderizar (checkpoint del modelo).
ITER=50000
MODEL=output/m360/flowers_beta_run7

# 1) Vídeo de trayectoria (vistas nuevas interpoladas).
# --skip_train --skip_test --skip_mesh + --render_path => SOLO genera el vídeo,
# sin re-exportar train/test ni extraer malla (marching cubes).
# Salida: $MODEL/traj/ours_50000/render_traj_color.mp4
python render.py -s Datasets/flowers \
    -m $MODEL \
    --iteration $ITER \
    --skip_train --skip_test --skip_mesh \
    --render_path \
    2>&1 | tee logs/flowers7_render.log

# 2) Comparativas render|GT lado a lado de las vistas de test.
# --skip_train --skip_mesh => exporta SOLO test (sin vídeo ni malla).
# Salida: $MODEL/test/ours_50000/vis/  (render|GT) y .../renders, .../gt
# Sirve para separar hueco real en vista observada vs artefacto de extrapolación.
python render.py -s Datasets/flowers \
    -m $MODEL \
    --iteration $ITER \
    --skip_train --skip_mesh \
    2>&1 | tee logs/flowers7_test.log
