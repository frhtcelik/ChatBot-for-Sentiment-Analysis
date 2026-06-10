"""Chatbot response helpers."""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Iterable, Mapping

from app.sentiment import SentimentResult


def build_reply(result: SentimentResult) -> str:
    if result.label == "positive":
        return "Mesajiniz pozitif gorunuyor. Bu iyi giden noktayi biraz daha detaylandirabilirsiniz."
    if result.label == "negative":
        return "Mesajiniz negatif bir duygu tasiyor. Isterseniz sizi en cok etkileyen kismi acabilirsiniz."
    return "Mesajiniz notr gorunuyor. Daha net bir analiz icin dusuncenizi biraz daha somut ifade edebilirsiniz."


def summarize_conversation(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    items = list(rows)
    if not items:
        return {
            "total": 0,
            "average_score": 0.0,
            "dominant_label": "neutral",
            "counts": {"positive": 0, "neutral": 0, "negative": 0},
        }

    counts = Counter(str(item["label"]) for item in items)
    for label in ("positive", "neutral", "negative"):
        counts.setdefault(label, 0)

    dominant_label = counts.most_common(1)[0][0]
    scores = [float(item["score"]) for item in items]

    return {
        "total": len(items),
        "average_score": round(mean(scores), 4),
        "dominant_label": dominant_label,
        "counts": dict(counts),
    }


def label_to_tr(label: str) -> str:
    return {
        "positive": "Pozitif",
        "neutral": "Notr",
        "negative": "Negatif",
    }.get(label, label)

