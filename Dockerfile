FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH
ENV TCNN_CUDA_ARCHITECTURES=89
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV COLMAP_USE_GPU=1
ENV QT_QPA_PLATFORM=offscreen

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates build-essential cmake ninja-build ffmpeg \
    colmap sqlite3 rsync \
    libgl1 libglib2.0-0 libxrender1 libxext6 libsm6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

RUN mamba create -n nerfstudio -y python=3.8 && \
    mamba clean -afy

SHELL ["conda", "run", "-n", "nerfstudio", "/bin/bash", "-c"]

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
      --extra-index-url https://download.pytorch.org/whl/cu118 && \
    pip install ninja && \
    pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch && \
    pip install nerfstudio

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
