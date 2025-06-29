from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import base64
from io import BytesIO
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Carrega o modelo pré-treinado MobileNetV2 (sem a camada final de classificação)
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

class ImagePair(BaseModel):
    image1_base64: str
    image2_base64: str

def decode_base64_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    image = image.resize((224, 224))
    return np.array(image)

def get_embedding(image_array):
    image_array = preprocess_input(image_array.astype(np.float32))
    image_array = np.expand_dims(image_array, axis=0)
    embedding = model.predict(image_array, verbose=0)
    return embedding[0]

@app.post("/compare")
def compare_signatures(images: ImagePair):
    img1 = decode_base64_image(images.image1_base64)
    img2 = decode_base64_image(images.image2_base64)

    emb1 = get_embedding(img1)
    emb2 = get_embedding(img2)

    similarity = cosine_similarity([emb1], [emb2])[0][0]
    threshold = 0.79  # Pode ser calibrado com testes reais

    return {
        "similarity": float(similarity),
        "same_signature": bool(similarity > threshold),  # ✅ Corrigido aqui
        "threshold": threshold,
        "model": "MobileNetV2 (ImageNet, deep features)"
    }
