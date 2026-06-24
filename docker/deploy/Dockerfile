FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/workspace/.hf_home
ENV PYTHONPATH=/workspace

RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    vim \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies (torch/torchvision/torchaudio already in base image)
RUN pip install --no-cache-dir \
    pandas==2.2.3 \
    scipy==1.12.0 \
    biopython==1.87 \
    scikit-learn==1.4.0 \
    xgboost==2.0.3 \
    matplotlib==3.11.0 \
    transformers==4.49.0 \
    peft==0.14.0 \
    datasets==3.3.2 \
    accelerate==1.14.0 \
    safetensors==0.8.0 \
    git+https://github.com/dridk/PyVCF3.git@1.0.4 \
    fastapi==0.137.1 \
    uvicorn[standard]==0.49.0 \
    loguru>=0.7.0 \
    fire==0.7.0 \
    tqdm \
    einops==0.8.2

# mamba-ssm requires CUDA build tools (ninja) and --no-build-isolation
RUN pip install --no-cache-dir --no-build-isolation \
    causal-conv1d==1.5.0.post8 \
    mamba-ssm==2.2.4

# Copy source code and models
COPY app /workspace/app
COPY modules /workspace/modules
COPY models /workspace/models

EXPOSE 8005

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"]
