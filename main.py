# main.py — RAG на ROSBERTa + Mistral + Qdrant

import os
import re
import textwrap
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from utils.text_processing import clean_text

from embeddings import get_embeddings

# =====================================================
# 0) Окружение и клиенты
# =====================================================
load_dotenv()

HF_TOKEN        = os.getenv("HF_TOKEN")
BASE_URL        = os.getenv("BASE_URL")
MAIN_MODEL      = os.getenv("MAIN_MODEL")
QDRANT_URL      = os.getenv("QDRANT_URL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# опциональные (с дефолтами)
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.4))
TOP_K                 = int(os.getenv("TOP_K", 3))

missing = [k for k, v in {
    "HF_TOKEN": HF_TOKEN,
    "BASE_URL": BASE_URL,
    "MAIN_MODEL": MAIN_MODEL,
    "QDRANT_URL": QDRANT_URL,
    "COLLECTION_NAME": COLLECTION_NAME,
}.items() if not v]
if missing:
    raise ValueError(f"Не заданы переменные окружения: {', '.join(missing)}")

# LLM (OpenAI-совместимый эндпоинт) и Qdrant
llm = OpenAI(base_url=BASE_URL, api_key=HF_TOKEN)
qdrant = QdrantClient(url=QDRANT_URL)

print("✅ Инициализация: подключено к Mistral и Qdrant")


# =====================================================
# 1) Поиск релевантных контекстов в Qdrant
#    (Distance.COSINE задан при создании коллекции -> сравнение по косинусному сходству)
# =====================================================
def get_top_contexts(collection: str, query_vector, top_k: int = 3) -> List[dict]:
    """
    Возвращает список payload'ов топ-k совпадений.
    Пытаемся через search(); если версия клиента иная — используем fallback.
    """
    # 1) Нормальный путь — search()
    try:
        hits = qdrant.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )
        # score — косинусная близость [0..1] (чем ближе к 1, тем релевантнее)
        hits = [h for h in hits if getattr(h, "score", 0.0) >= SIMILARITY_THRESHOLD]
        return [getattr(h, "payload", {}) for h in hits]
    except Exception as e:
        print(f"⚠️ search() недоступен, пробуем fallback: {e}")

    # 2) Fallback — query_points (разные версии возвращают разные структуры)
    try:
        qr = qdrant.query_points(
            collection_name=collection,
            query=query_vector.tolist() if hasattr(query_vector, "tolist") else query_vector,
            limit=top_k,
            with_payload=True
        )
        if hasattr(qr, "points"):
            raw = qr.points
        elif isinstance(qr, tuple):
            raw = qr[0]
        else:
            raw = qr

        payloads = []
        for item in raw:
            if hasattr(item, "payload") and isinstance(item.payload, dict):
                payloads.append(item.payload)
            elif isinstance(item, dict) and "payload" in item:
                payloads.append(item["payload"])
        return payloads[:top_k]
    except Exception as e:
        print(f"⚠️ Ошибка при fallback-поиске: {e}")
        return []


# =====================================================
# 2) Построение промпта
# =====================================================
def build_prompt(context: str, question: str) -> str:
    return f"""
Ты — виртуальный помощник Альфа-Банка. Отвечай ТОЛЬКО на основе предоставленного контекста.
Если информации недостаточно — вежливо сообщи об этом.

Инструкции:
1) Используй только факты из контекста.
2) Отвечай кратко (2–3 предложения), понятным языком.
3) Если данных нет — скажи: "Извините, информации по вашему вопросу нет в базе знаний."

Контекст:
{context}

Вопрос:
{question}
""".strip()


# =====================================================
# 3) Генерация ответа через LLM
# =====================================================
def generate_answer(question: str, context: str) -> str:
    system_prompt = build_prompt(context, question)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    completion = llm.chat.completions.create(
        model=MAIN_MODEL,
        temperature=0.2,
        max_tokens=512,
        messages=messages,
    )
    raw = completion.choices[0].message.content
    return clean_text(raw)


# =====================================================
# 4) Основной обработчик запроса
# =====================================================
def ask_question(user_input: str) -> str:
    # эмбеддинг запроса через ROSBERTa (embeddings.py)
    query_vec = get_embeddings(user_input)[0]

    # поиск релевантных записей (косинусная близость в Qdrant)
    payloads = get_top_contexts(COLLECTION_NAME, query_vec, top_k=TOP_K)
    if not payloads:
        print("⚠️ Не найдено релевантных записей в базе знаний.")
        return "Извините, информации по вашему вопросу нет в базе знаний."

    # формируем контекст из ответов
    context = "\n\n".join([f"• {p.get('answer', '')}" for p in payloads if p.get("answer")])

    # генерация ответа
    reply = generate_answer(user_input, context)
    print("\n🤖 Ответ модели:\n")
    print(textwrap.fill(reply, width=90))
    return reply


# =====================================================
# 5) CLI для ручного тестирования
# =====================================================
if __name__ == "__main__":
    print("\n💬 Система готова. Введите вопрос (или 'exit' для выхода).")
    while True:
        q = input("\nВаш вопрос: ").strip()
        if q.lower() in {"exit", "quit"}:
            print("👋 Выход.")
            break
        ask_question(q)