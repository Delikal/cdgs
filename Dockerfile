FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH
ENV TCNN_CUDA_ARCHITECTURES=89
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV COLMAP_USE_GPU=1
ENV COLMAP_CUDA_ARCHITECTURES=89
ENV QT_QPA_PLATFORM=offscreen
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=10
ARG COLMAP_VERSION=3.13.0
ARG COLMAP_CUDA_ARCHITECTURES=89

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates build-essential cmake ninja-build ffmpeg \
    sqlite3 rsync \
    libboost-program-options-dev libboost-graph-dev libboost-system-dev \
    libboost-filesystem-dev libboost-regex-dev \
    libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev \
    libopenimageio-dev openimageio-tools libblas-dev liblapack-dev \
    libgoogle-glog-dev libgflags-dev libgtest-dev libgmock-dev libsqlite3-dev libglew-dev \
    qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev libsuitesparse-dev \
    libcurl4-openssl-dev libssl-dev \
    libgl1 libglib2.0-0 libxrender1 libxext6 libsm6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --branch "$COLMAP_VERSION" --depth 1 https://github.com/colmap/colmap.git /tmp/colmap && \
    cmake -S /tmp/colmap -B /tmp/colmap/build -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCUDA_ENABLED=ON \
      -DGUI_ENABLED=OFF \
      -DCMAKE_CUDA_ARCHITECTURES="$COLMAP_CUDA_ARCHITECTURES" && \
    cmake --build /tmp/colmap/build --target install --parallel "$(nproc)" && \
    rm -rf /tmp/colmap && \
    colmap -h >/dev/null

RUN wget -qO /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

RUN mamba create -n nerfstudio -y python=3.8 && \
    mamba clean -afy

SHELL ["conda", "run", "-n", "nerfstudio", "/bin/bash", "-c"]

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
      --extra-index-url https://download.pytorch.org/whl/cu118

RUN pip install ninja

RUN pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

RUN pip install nerfstudio

RUN git clone https://github.com/maturk/dn-splatter /opt/dn-splatter && \
    cd /opt/dn-splatter && \
    pip install setuptools==69.5.1 && \
    pip install -e .

RUN pip install opencv-python pillow matplotlib timm huggingface_hub

RUN git clone https://github.com/DepthAnything/Depth-Anything-V2 /opt/depth-anything && \
    cd /opt/depth-anything && \
    pip install -r requirements.txt

RUN mkdir -p /opt/depth-anything/checkpoints && \
    wget -O /opt/depth-anything/checkpoints/depth_anything_v2_vits.pth \
      "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true" && \
    wget -O /opt/depth-anything/checkpoints/depth_anything_v2_vitb.pth \
      "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true"

COPY generate_depth.py /generate_depth.py
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
