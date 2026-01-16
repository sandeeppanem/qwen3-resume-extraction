FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone llama-cpp-python WITH llama.cpp submodule
# This gets the latest llama.cpp that llama-cpp-python supports
# If the submodule has Qwen3 support, this will work!
WORKDIR /tmp
RUN echo "Cloning llama-cpp-python with submodules..." && \
    git clone --depth 1 --recurse-submodules https://github.com/abetlen/llama-cpp-python.git && \
    echo "✓ Cloned successfully"

WORKDIR /tmp/llama-cpp-python

# HF Spaces–safe build flags (single job, no optimizations)
# Pass these as CMake args, not just env vars
ENV CMAKE_BUILD_PARALLEL_LEVEL=1
ENV CMAKE_ARGS="-DLLAMA_AVX=OFF -DLLAMA_AVX2=OFF -DLLAMA_FMA=OFF -DLLAMA_CUBLAS=OFF -DLLAMA_CUDA=OFF -DLLAMA_METAL=OFF"

# Build and install llama-cpp-python
RUN echo "Building llama-cpp-python (this may take 15-20 minutes)..." && \
    pip install --no-cache-dir . && \
    echo "✓ llama-cpp-python installed successfully"

# App
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY qwen3-resume-parser-Q5_K_M.gguf .

EXPOSE 7860
CMD ["python", "app.py"]
