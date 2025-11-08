# загрузка базы знаний в Qdrant

import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance

from embeddings import get_embeddings  # модуль ROSBERTa

# === Загружаем env ===
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
DATA_PATH = os.getenv("DATA_PATH")


# --- Разбиение текста на чанки ---
def chunk_text(question: str, answer: str) -> list[str]:
    """
    Для небольшой базы знаний (до ~1000 строк) используем простую схему:
    один FAQ = один чанк.
    Объединяем вопрос и ответ в общий контекст.
    """
    context = f"Вопрос: {question}\nОтвет: {answer}"
    return [context.strip()]


# --- Основная функция загрузки базы знаний ---
def upload_context_from_csv(
    csv_path: str = DATA_PATH,
    collection_name: str = COLLECTION_NAME,
    qdrant_url: str = QDRANT_URL
):
    """
    Загружает CSV с вопросами/ответами в Qdrant.
    Каждый вопрос+ответ -> один чанк.
    """
    # Проверяем наличие файла
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Файл не найден: {csv_path}")

    # Загружаем CSV
    df = pd.read_csv(csv_path)
    required_cols = {"question", "answer"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"В CSV должны быть колонки: {required_cols}. "
            f"Найдены: {set(df.columns)}"
        )

    print(f"📚 Загружено {len(df)} записей из {csv_path}")

    # Подключаемся к Qdrant
    client = QdrantClient(url=qdrant_url)
    VECTOR_SIZE = 1024

    # Пересоздаём коллекцию
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        print(f"Удалена старая коллекция '{collection_name}'")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )
    print(f"Коллекция '{collection_name}' создана")

    # --- Обработка и загрузка данных ---
    all_points = []
    idx_counter = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="🔹 Обработка записей"):
        question = str(row["question"])
        answer = str(row["answer"])
        category = row.get("category", "unknown")

        # один FAQ = один чанк
        chunks = chunk_text(question, answer)

        embeddings = get_embeddings(chunks)

        for i, emb in enumerate(embeddings):
            all_points.append(
                models.PointStruct(
                    id=idx_counter,
                    vector=emb,
                    payload={
                        "question": question,
                        "answer": answer,
                        "category": category,
                        "chunk_index": i
                    }
                )
            )
            idx_counter += 1

    # --- Загрузка в Qdrant ---
    client.upsert(collection_name=collection_name, points=all_points)
    print(f"Загрузка завершена: {len(all_points)} векторов добавлено в коллекцию '{collection_name}'")

    return client


if __name__ == "__main__":
    upload_context_from_csv()