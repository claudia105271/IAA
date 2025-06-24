from kafka import KafkaConsumer, KafkaProducer
from collections import Counter, defaultdict
from PIL import Image
import tflite_runtime.interpreter as tflite
import base64
import json
import time
import subprocess
import numpy as np
import io
import os

# Function to pre-process images for models
def preprocess_image(img_data, model):
    sizes = {
        "MobileNetV2": (224, 224),
        "Resnet50": (224, 224),
        "InceptionV3": (299, 299)
    }
    
    # ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = Image.open(io.BytesIO(img_data)).convert('RGB')
    img = img.resize(sizes.get(model))

    # Normalize image
    img = np.array(img).astype(np.float32) / 255.0
    img = (img - mean) / std
    img = np.expand_dims(img, axis=0)  # add batch dimension

    return img

if __name__ == "__main__":
    n_inferences = 10
    label_path = "/models/imagenet_labels.json"
    model_paths = {
        "MobileNetV2": "/models/mobilenet_v2.tflite",
        "Resnet50": "/models/resnet50.tflite",
        "InceptionV3": "/models/inception_v3.tflite"
    }

    # Set up Kafka Producer flow (inference worker -> user pp)
    producer = KafkaProducer(
        bootstrap_servers='kafka:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        api_version=(3, 5, 1)
    )

    # Set up Kafka Consumer flow (user app -> inference worker)
    consumer = KafkaConsumer(
        'inference_jobs',
        bootstrap_servers='kafka:9092',
        auto_offset_reset='earliest',
        group_id='worker1',
        api_version=(3, 5, 1)
    )

    # Poll until models are found (wait for download)
    for i in range(120):
        if os.path.exists(label_path):
            print(f"Found model file at {label_path}", flush=True)
            break
        print(f"Waiting for models to be downloaded...", flush=True)
        time.sleep(5)
    else:
        raise FileNotFoundError(f"Model file not found after {120 * 5} seconds")

    with open(label_path, 'r') as f:
        labels = json.load(f)

    # Wait for messages from user application
    for msg in consumer:
        decoded_value = msg.value.decode('utf-8')  # decode bytes to string
        job = json.loads(decoded_value)            # parse JSON from string

        model_name = job['model']
        device = job['device']
        img_data = base64.b64decode(job['image'])
        img_array = preprocess_image(img_data, model_name)
        model_path = model_paths[model_name]

        try:
            if device == 'GPU':
               delegate = tflite.load_delegate('libtensorflowlite_gpu_delegate.so')
               interpreter = tflite.Interpreter(model_path=model_path, experimental_delegates=[delegate])
            else:
               interpreter = tflite.Interpreter(model_path=model_path)
        except Exception as e:
            print(f"Error loading tflite interpreter: {e}", flush=True)
            raise

        try:
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]['index'], img_array)
        except Exception as e:
            print(f"Error setting up tflite model inputs and outputs: {e}", flush=True)
            raise

        try:
            inference_times = []
            predictions = []

            for _ in range(n_inferences):
                start_time = time.time()
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])[0]
                end_time = time.time()

                inference_times.append(end_time - start_time)

                top_idx = output_data.argmax()
                confidence = output_data[top_idx]
                predictions.append((top_idx, confidence))

        except Exception as e:
            print(f"Error during model inference (invoke): {e}", flush=True)
            raise

        arr = np.array(inference_times)
        avg_inference_time = np.mean(arr)
        sdv_inference_time = np.std(arr)

        # Find the label with the highest frequency among the top predictions,
        # breaking ties by highest confidence
        counter = Counter()
        confidences = defaultdict(float)

        for idx, conf in predictions:
            counter[idx] += 1
            # Store max confidence per label
            if conf > confidences[idx]:
                confidences[idx] = conf

        # Find label(s) with max frequency
        max_freq = max(counter.values())
        candidates = [idx for idx, freq in counter.items() if freq == max_freq]

        # Among candidates, pick the one with highest confidence
        best_idx = max(candidates, key=lambda i: confidences[i])
        best_confidence = confidences[best_idx]
        label_frequency = round((max_freq / n_inferences) * 100, 2)

        result_payload = {
            "job_id": job.get("job_id", f"job_{int(time.time())}"),
            "label": labels[best_idx],
            "confidence": float(best_confidence),
            "avg_inference_time": float(avg_inference_time),
            "sdv_inference_time": float(sdv_inference_time),
            "label_frequency": label_frequency,
            "model": model_name
        }
        producer.send("inference_results", value=result_payload)
        producer.flush()

