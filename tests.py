# файлик для тестов фичей разработки

from dotenv import load_dotenv
import os

load_dotenv()
print(os.getenv("HF_TOKEN"))
print(os.getenv("EMBEDDING_MODEL"))