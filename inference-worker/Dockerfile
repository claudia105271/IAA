FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install dependencies including Python 3.10, Bazel, OpenCL, and build tools
RUN apt-get update && apt-get install -y \
    wget git cmake curl \
    gnupg \
    ocl-icd-libopencl1 \
    ocl-icd-opencl-dev \
    python3.10 python3.10-venv python3.10-distutils python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LO "https://github.com/bazelbuild/bazel/releases/download/7.4.1/bazel-7.4.1-linux-x86_64" \
  && chmod +x bazel-7.4.1-linux-x86_64 \
  && mv bazel-7.4.1-linux-x86_64 /usr/local/bin/bazel

COPY requirements.txt .

RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

# Build TFlite GPU delegate
RUN git clone --depth=1 https://github.com/tensorflow/tensorflow.git /app/tensorflow

WORKDIR /app/tensorflow

RUN bazel build -c opt \
    --copt=-DTFLITE_GPU_USE_OPENCL \
<<<<<<< HEAD
    //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so

=======
    --copt=-DCL_DELEGATE_NO_GL \
    //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so

RUN cp \
    bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so \
    /usr/lib/

>>>>>>> 9440a547f3e543450f521f982f07aff066ff9320
RUN rm -rf /app/tensorflow

WORKDIR /app

COPY worker.py .

VOLUME /models

CMD ["python3.10", "worker.py"]

