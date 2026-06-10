"""Command-line entry point for the sentiment chatbot."""

from __future__ import annotations

from app.chatbot import build_reply
from app.database import save_message
from app.sentiment import analyze_sentiment


def main() -> None:
    print("Sentiment Analysis Chatbot")
    print("Cikmak icin 'q' veya 'exit' yazin.")

    while True:
        user_message = input("\nMesajiniz: ").strip()
        if user_message.lower() in {"q", "quit", "exit", "cikis", "çıkış"}:
            print("Gorusmek uzere.")
            break

        result = analyze_sentiment(user_message)
        reply = build_reply(result)
        save_message(user_message, reply, result)

        print(f"Duygu: {result.label_tr}")
        print(f"Skor: {result.score:.2f} | Guven: {result.confidence:.2f}")
        print(f"Bot: {reply}")


if __name__ == "__main__":
    main()

