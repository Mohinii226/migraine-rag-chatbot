"""Chat with your papers in the terminal.

    python chat_cli.py

Type a question and press Enter. Type `quit` to leave.
"""
from rag.pipeline import answer_question, load_store


def main():
    print("Loading index and embedding model (a few seconds)...")
    store = load_store()
    print(f"Ready. {len(store.chunks)} chunks indexed (search backend: {store.backend}).")
    print("This is a research prototype, not medical advice. Type 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        result = answer_question(store, question)
        print(f"\nAssistant ({result['mode']}): {result['answer']}\n")
        print("Sources used:")
        for number, p in enumerate(result["passages"], start=1):
            print(f"  [{number}] {p['source']}, page {p['page']} (similarity {p['score']:.2f})")
        print()


if __name__ == "__main__":
    main()
