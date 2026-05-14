# Imagen base con CUDA
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Dependencias del sistema
RUN apt-get update && apt-get install -y \
    wget git build-essential ninja-build \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Miniconda
ENV PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm Miniconda3-latest-Linux-x86_64.sh

WORKDIR /workspace

# ✅ COPIAMOS EL PROYECTO LOCAL (incluye submodules ya editados)
COPY . /workspace/

# CUDA build vars
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

# Aceptar ToS de Conda
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Crear entorno conda (SIN descargar submódulos)
RUN conda env create -f environment.yml

# Instalar extensiones CUDA DESDE RUTAS LOCALES
RUN /root/miniconda3/envs/surfel_splatting/bin/pip install ./submodules/simple-knn
RUN /root/miniconda3/envs/surfel_splatting/bin/pip install ./submodules/diff-surfel-rasterization

# Auto-activar entorno
RUN echo "conda activate surfel_splatting" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

ENV PATH=/root/miniconda3/envs/surfel_splatting/bin:$PATH

CMD ["/bin/bash"]
