import tensorflow as tf
import json
import urllib.request
import os

MODEL_DIR = "/models"
os.makedirs(MODEL_DIR, exist_ok=True)

def convert_and_save(model_name):
    filename = os.path.join(MODEL_DIR, f"{model_name}.tflite")
    if os.path.exists(filename):
        return

    if model_name == "mobilenet_v2":
        keras_model = tf.keras.applications.MobileNetV2(weights='imagenet')
    elif model_name == "inception_v3":
        keras_model = tf.keras.applications.InceptionV3(weights='imagenet')
    elif model_name == "resnet50":
        keras_model = tf.keras.applications.ResNet50(weights='imagenet')
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    with open(filename, "wb") as f:
        f.write(tflite_model)
    print(f"Saved {filename}")


if __name__ == "__main__":
    model_names = ["mobilenet_v2", "inception_v3", "resnet50"]
    
    # Download models
    for name in model_names:
        convert_and_save(name)
    
    # Download labels
    filename = os.path.join(MODEL_DIR, "imagenet_labels.json")
    if not os.path.exists(filename):
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded ImageNet labels to {filename}")
