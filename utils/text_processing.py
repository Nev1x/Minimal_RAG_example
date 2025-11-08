# токенизация, очистка текста
import re

def clean_text(text: str) -> str:
    """Очистка текста модели от мусора"""
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    text = re.sub(r"[^а-яА-ЯёЁa-zA-Z0-9\s.,!?;:()«»\"'–—\-]", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()