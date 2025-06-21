Opção 2: Usar MobileNetV2 (pré-treinado) como extrator de embeddings
Sem precisar treinar do zero

Extraímos embeddings de cada assinatura

Comparamos usando cosine similarity

Ainda é IA, pois usa um modelo de deep learning (CNN) pré-treinado

👨‍🏫 O que dizer ao professor (exemplo de explicação)
“Minha API compara duas assinaturas usando inteligência artificial com um modelo de rede neural convolucional (MobileNetV2) para extrair características profundas das imagens. As assinaturas são convertidas em vetores e comparadas com a métrica de similaridade cosseno. Utilizei embeddings pré-treinados para garantir qualidade e evitar necessidade de um dataset grande. A IA entra no processo como extratora de padrões complexos das imagens.”

❓Você quer que eu monte:
Uma API usando MobileNetV2 (pré-treinado) como IA?

Ou quer que eu mostre como treinar uma rede siamesa simples com dataset?

Posso te entregar os dois prontos. Qual prefere?

You said:
Uma API usando MobileNetV2 (pré-treinado) como IA?

ChatGPT said:
Perfeito! Vamos montar uma API em Python que usa o MobileNetV2 pré-treinado como modelo de inteligência artificial para comparar assinaturas em imagens base64.

✅ O que essa API faz:
Recebe duas imagens de assinaturas em base64

Converte para imagem e pré-processa

Usa MobileNetV2 (pré-treinado no ImageNet) como extrator de características

Compara os vetores de características usando similaridade cosseno

Retorna a similaridade (de 0 a 1) e se são da mesma pessoa com base em um limite (threshold)

📦 Requisitos
Instale os pacotes:

bash
Copy
Edit
pip install fastapi uvicorn pillow numpy tensorflow
🧠 Código da API com IA (MobileNetV2)
python
Copy
Edit

# api_mobilenet.py

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
    threshold = 0.85  # Pode ser calibrado com testes reais

    return {
        "similarity": float(similarity),
        "same_signature": similarity > threshold,
        "threshold": threshold,
        "model": "MobileNetV2 (ImageNet, deep features)"
    }

▶️ Como executar
bash
Copy
Edit
uvicorn api_mobilenet:app --reload
Acesse: http://127.0.0.1:8000/docs para testar pela interface Swagger.

🧪 Exemplo de como testar
Envie um POST para /compare com um corpo como:

json
Copy
Edit
{
"image1_base64": "<assinatura_base64_1>",
"image2_base64": "<assinatura_base64_2>"
}
🎓 Como explicar para o professor
“A API usa MobileNetV2, uma rede neural convolucional treinada no ImageNet, como extratora de características profundas das imagens. As assinaturas são comparadas por similaridade cosseno dos embeddings (vetores gerados pela rede). Como se trata de um modelo de deep learning, ele representa padrões visuais complexos da imagem, sendo uma aplicação legítima de inteligência artificial.”
