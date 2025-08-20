FROM nvidia/cuda:12.4.0-cudnn-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and uv (fast pip replacement)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git curl ffmpeg \
    build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -s /root/.local/bin/uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy repository contents
COPY . /app

# Fetch submodules
RUN git submodule update --init --recursive

# Install core packages
RUN uv pip install --system torch==2.7.1 torchvision==0.22.1

# Install submodule dependencies
RUN uv pip install --system ./submodules/nvdiffrast && \
    (cd submodules/simple-knn && python setup.py install) && \
    uv pip install --system git+https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose.git && \
    git clone https://github.com/esw0116/ODGS.git submodules/ODGS && \
    uv pip install --system submodules/ODGS/submodules/odgs-gaussian-rasterization

# Install project package
RUN uv pip install --system -e code/DiffSynth-Studio

# Install remaining Python dependencies
RUN uv pip install --system plyfile decord ffmpeg trimesh pyrender xfuser diffusers open3d py360convert && \
    uv pip install --system "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7" && \
    uv pip install --system peft easydict torchsde open-clip-torch==2.7.0 fairscale natsort && \
    uv pip install --system realesrgan && \
    uv pip install --system flash_attn-2.7.4.post1+cu12torch2.6cp310-cp310-linux_x86_64.whl --no-build-isolation && \
    uv pip install --system git+https://github.com/EasternJournalist/utils3d.git#egg=utils3d && \
    uv pip install --system xformers==0.0.31 && \
    uv pip install --system jaxtyping==0.3.2 modelscope==1.28.2 diffusers==0.34.0 matplotlib==3.8.4 transformers==4.51.0 && \
    uv pip install --system torchmetrics==0.7.0 OmegaConf==2.1.1 imageio-ffmpeg==0.6.0 pytorch-lightning==1.4.2 omegaconf==2.1.1 && \
    uv pip install --system webdataset==0.2.5 kornia==0.6 streamlit==1.12.1 einops==0.8.0 open_clip_torch && \
    uv pip install --system SwissArmyTransformer==0.4.12 wandb==0.21.1 && \
    uv pip install --system -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers && \
    uv pip uninstall -y basicsr && \
    uv pip install --system openai-clip

CMD ["bash"]
