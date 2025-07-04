FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install dependencies of tflite: Python 3.10, Bazel 7.4.1, OpenCL
RUN apt-get update && apt-get install -y \
    git curl \
    ocl-icd-libopencl1 \
    ocl-icd-opencl-dev \
    python3.10 python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

# Build TFlite GPU delegate
RUN curl -LO "https://github.com/bazelbuild/bazel/releases/download/7.4.1/bazel-7.4.1-linux-x86_64" \
  && chmod +x bazel-7.4.1-linux-x86_64 \
  && mv bazel-7.4.1-linux-x86_64 /usr/local/bin/bazel

RUN git clone --depth=1 https://github.com/tensorflow/tensorflow.git /app/tensorflow

WORKDIR /app/tensorflow

RUN bazel build -c opt \
    --copt=-DTFLITE_GPU_USE_OPENCL \
    --copt=-DCL_DELEGATE_NO_GL \
    //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so

RUN cp \
    bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so \
    /usr/lib/

RUN rm -rf /app/tensorflow

# Run main code
WORKDIR /app

COPY worker.py .

COPY libs/* /usr/lib/x86_64-linux-gnu/

VOLUME /models

CMD ["python3.10", "worker.py"]

