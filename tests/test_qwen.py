from backend.ai.model_manager import ModelManager


def main():
    manager = ModelManager()

    response = manager.generate(
        "Explain what a qubit is in 3 simple sentences.",
        max_tokens=200
    )

    print("\n===== QWEN RESPONSE =====\n")
    print(response)


if __name__ == "__main__":
    main()