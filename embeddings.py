# подключение и работа с ROSBERTa
import os
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

missing_vars = []
if not HF_TOKEN:
    missing_vars.append("HF_TOKEN")
if not EMBEDDING_MODEL:
    missing_vars.append("EMBEDDING_MODEL")

if missing_vars:
    raise ValueError(
        f"Не найдены обязательные переменные окружения: {', '.join(missing_vars)}.\n"
        f"Проверь .env файл и перезапусти скрипт."
    )

client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN,
)

def get_embeddings(texts, model_name=EMBEDDING_MODEL):
    """
    Преобразует текст(ы) в эмбеддинги через Hugging Face Inference API.
    Возвращает список numpy-векторов.
    """
    if isinstance(texts, str):
        texts = [texts]

    vectors = []
    for text in texts:
        try:
            result = client.feature_extraction(text, model=model_name)
            if isinstance(result, dict):
                result = result.get("embeddings") or result.get("features") or result
            arr = np.array(result, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr.mean(axis=0)
            vectors.append(arr)
        except Exception as e:
            print(f"Ошибка при обработке '{text[:30]}...': {e}")
            vectors.append(np.zeros(1024, dtype=np.float32))
    return vectors
