# ============================================================
# Base (PyTorch 2.7.0 + CUDA 12.8 + cuDNN 9, with build toolchain)
# ============================================================
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

# --- Environment ---
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    MAX_JOBS=4 \
    PYGLET_HEADLESS=1 \
    PYOPENGL_PLATFORM=egl \
    EGL_PLATFORM=surfaceless

# --- System deps (build + headless GL/EGL runtime bits) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates tini \
    build-essential cmake pkg-config \
    libegl1 libgles2 libgl1 libglvnd0 \
    libxrender1 libxext6 libsm6 libx11-6 libxi6 libxxf86vm1 \
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

# ============================================================
# App
# ============================================================
WORKDIR /app/Matrix-3D
COPY . .

# Torch/vision are already present from the base image — do not (re)install.
# Upgrade packaging tools once.
RUN python -m pip install --upgrade pip setuptools wheel

# ============================================================
# install.sh (broken into Dockerfile steps)
# ============================================================

# 1) "✅ Installing Submodules..." — nvdiffrast & simple-knn
RUN set -eux; \
    cd ./submodules/nvdiffrast && \
    pip install . && \
    cd ../simple-knn && \
    python setup.py install

# 2) Diff Gaussian rasterizers (ODGS, forks)
RUN set -eux; \
    pip install "git+https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose.git"; \
    cd ./submodules && \
    git clone https://github.com/esw0116/ODGS.git && \
    cd ODGS && \
    pip install submodules/odgs-gaussian-rasterization

# 3) DiffSynth-Studio (editable)
RUN set -eux; \
    cd /app/Matrix-3D/code/DiffSynth-Studio && \
    pip install -e .

# 4) Python dependencies (kept in the same order as your script)
#    NOTE: If any of these pull in torch/torchvision, pip may try to resolve them.
#    We rely on the base image’s torch 2.7.0; pin extra deps carefully if needed.
ARG FLASH_ATTN_WHEEL="flash_attn-2.7.4.post1+cu12torch2.6cp310-cp310-linux_x86_64.whl"
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# IMPORTANT: The example filename you gave is built for *torch 2.6*. For torch 2.7.0
# you must supply a matching wheel (name will differ). Override at build time:
#   podman build --build-arg FLASH_ATTN_WHEEL=<matching_whl> -t your/image .
RUN set -eux; \
    pip install \
        plyfile decord ffmpeg trimesh pyrender xfuser diffusers open3d py360convert && \
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7" && \
    pip install peft easydict torchsde open-clip-torch==2.7.0 fairscale natsort && \
    pip install realesrgan && \
    pip install "/app/Matrix-3D/${FLASH_ATTN_WHEEL}" --no-build-isolation && \
    pip install "git+https://github.com/EasternJournalist/utils3d.git#egg=utils3d" && \
    pip install xformers==0.0.31 && \
    pip install jaxtyping==0.3.2 && \
    pip install modelscope==1.28.2 && \
    pip install diffusers==0.34.0 && \
    pip install matplotlib==3.8.4 && \
    pip install transformers==4.51.0 && \
    pip install torchmetrics==0.7.0 && \
    pip install OmegaConf==2.1.1 && \
    pip install imageio-ffmpeg==0.6.0 && \
    pip install pytorch-lightning==1.4.2 && \
    pip install omegaconf==2.1.1 && \
    pip install webdataset==0.2.5 && \
    pip install kornia==0.6 && \
    pip install streamlit==1.12.1 && \
    pip install einops==0.8.0 && \
    pip install open_clip_torch && \
    pip install SwissArmyTransformer==0.4.12 && \
    pip install wandb==0.21.1 && \
    pip install -e "git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers" && \
    pip uninstall -y basicsr && \
    pip install openai-clip

# (Optional) prove imports resolve at build time for faster failures:
# RUN python - <<'PY'
# import torch, diffusers, pytorch_lightning, xformers, kornia, open_clip
# import trimesh, pyrender
# print("Imports OK")
# PY

ENTRYPOINT ["/usr/bin/tini", "-g", "--"]
CMD ["/bin/bash"]

