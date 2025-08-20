FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Target SMs
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV FORCE_CUDA=1
ENV MAX_JOBS=4
ENV CUDACXX=/usr/local/cuda/bin/nvcc

# OS deps (runtime + build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git curl ffmpeg \
    build-essential cmake ninja-build \
    libgl1 libglib2.0-0 libxext6 libxrender1 libsm6 libxi6 libxxf86vm1 libosmesa6 \
    libjpeg-turbo8 zlib1g libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Make python/pip point to python3/pip3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
 && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app
COPY . /app
RUN git submodule update --init --recursive

# --- Torch pinned to 2.6.0 cu124 (matches FlashAttention wheel) ---
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
      torch==2.6.0 torchvision==0.21.0

# nvdiffrast
RUN python -m pip install ./submodules/nvdiffrast

# simple-knn via PEP 517
RUN python -m pip install --no-build-isolation ./submodules/ODGS/submodules/simple-knn

# diff-gaussian-rasterization w/ pose
RUN python -m pip install git+https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose.git

# ODGS rasterization
RUN python -m pip install submodules/ODGS/submodules/odgs-gaussian-rasterization

# Project package (editable)
RUN python -m pip install -e code/DiffSynth-Studio

# --- FlashAttention (prebuilt wheel only) ---
# Place your wheel next to the Dockerfile (example name below).
# Must match: Python 3.10 (cp310), Torch 2.6, CUDA 12.x.
# Example: flash_attn-2.7.4.post1+cu12torch2.6cp310-cp310-linux_x86_64.whl
COPY flash_attn-*.whl /tmp/flash-attn.whl
RUN python -m pip install /tmp/flash-attn.whl --no-build-isolation \
 && rm -f /tmp/flash-attn.whl

# Remaining deps
RUN python -m pip install \
      plyfile decord ffmpeg-python trimesh pyrender xfuser diffusers open3d py360convert \
  && python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7" \
  && python -m pip install peft easydict torchsde open-clip-torch==2.7.0 fairscale natsort \
  && python -m pip install realesrgan \
  && python -m pip install "git+https://github.com/EasternJournalist/utils3d.git#egg=utils3d" \
  && python -m pip install xformers==0.0.31 \
  && python -m pip install jaxtyping==0.3.2 modelscope==1.28.2 diffusers==0.34.0 matplotlib==3.8.4 transformers==4.51.0 \
  && python -m pip install torchmetrics==0.7.0 OmegaConf==2.1.1 imageio-ffmpeg==0.6.0 pytorch-lightning==1.4.2 omegaconf==2.1.1 \
  && python -m pip install webdataset==0.2.5 kornia==0.6 streamlit==1.12.1 einops==0.8.0 open_clip_torch \
  && python -m pip install SwissArmyTransformer==0.4.12 wandb==0.21.1 \
  && python -m pip install -e "git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers" \
  && python -m pip uninstall -y basicsr || true \
  && python -m pip install openai-clip

CMD ["bash"]

