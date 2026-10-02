"""
Практика 10, упражнение 7: первый вызов OpenAI API.

Скрипт читает отзывы из practice/data/reviews.txt и для каждого получает
структурированный ответ модели: тема, тональность, краткая суть.

Запуск:
    pip install openai pydantic
    export OPENAI_API_KEY="sk-..."          # Windows: set OPENAI_API_KEY=sk-...
    export OPENAI_MODEL="gpt-5-mini"        # необязательно; актуальные модели —
                                            # https://platform.openai.com/docs/models
    python practice/examples/api_example.py
"""

import os
import re
import sys
from pathlib import Path
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "reviews.txt"
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")


class ReviewAnalysis(BaseModel):
    """Схема ответа: модель обязана вернуть именно эти поля."""

    topic: Literal["оплата", "стабильность", "доставка", "качество товаров",
                   "поддержка", "интерфейс", "цены", "другое"]
    sentiment: Literal["позитив", "нейтрально", "негатив"]
    summary: str  # суть отзыва в 5–10 словах


INSTRUCTIONS = (
    "Ты — продуктовый аналитик. Определи главную тему отзыва, тональность "
    "и сформулируй суть в 5–10 словах. Не додумывай то, чего нет в тексте."
)


def load_reviews(path: Path) -> list[str]:
    if not path.exists():
        sys.exit(f"Файл не найден: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    # отзывы начинаются с номера: «1. Текст...»
    return [re.sub(r"^\d+\.\s*", "", ln) for ln in lines if re.match(r"^\d+\.", ln)]


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Задайте переменную окружения OPENAI_API_KEY (не храните ключ в коде).")

    client = OpenAI()
    reviews = load_reviews(DATA_FILE)
    print(f"Модель: {MODEL}. Отзывов: {len(reviews)}\n")

    for i, review in enumerate(reviews, start=1):
        response = client.responses.parse(
            model=MODEL,
            instructions=INSTRUCTIONS,
            input=review,
            text_format=ReviewAnalysis,
        )
        result = response.output_parsed
        print(f"{i:>2}. [{result.sentiment:^10}] {result.topic:<16} — {result.summary}")

    # Задание: попросите ChatGPT доработать скрипт —
    # 1) сохранять результаты в reviews_analysis.csv;
    # 2) печатать сводку: сколько отзывов по каждой теме и доля негатива;
    # 3) отправлять все отзывы одним запросом, чтобы сэкономить токены.


if __name__ == "__main__":
    main()
