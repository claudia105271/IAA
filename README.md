## Project Overview

This project features a lightweight Docker-based application composed of two containers:

- **Application Container**  
  Provides a user interface for uploading images to be classified.

- **Inference Container**  
  Runs the selected model for image classification, supporting execution on either CPU or GPU.  
  This container has access to the GPU device and the necessary drivers.
## How to run
Run the application on a GPU-enabled host:
```bash
./inference-worker/gather_libs.sh
docker-compose up --build
```

![image](https://github.com/user-attachments/assets/f17aa06a-8ee4-40c2-81c1-a6ec258cadbb)


### Assumptions
- All the containers are run on the target machine.
- Models were trained on the cloud. This was simulated by retrieving pre-trained tflite models with download_trained_models.py.

### Restrictions
- For non-Intel backends, the `--gpus all` option cannot be used when launching Docker containers.
- GPU driver libraries must be copied into the Docker image ahead of time using the `gather_libs.sh` script.
