from pathlib import Path
import json
from llama_cpp import Llama


class ModelManager:
    def __init__(self, config_path: str = "config/models.json"):
        self.config_path = Path(config_path)
        self.model = None
        self.config = self._load_config()

    def _load_config(self):
        with self.config_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def load_chat_model(self):
        config = self.config["chat"]
        model_path = Path(config["path"])

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        print(f"Loading {config['name']}...")

        self.model = Llama(
            model_path=str(model_path),
            n_ctx=config["context_size"],
            verbose=False,
        )

        print("Model loaded.")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float | None = None,
        system_prompt: str = "You are a helpful AI presentation assistant.",
    ) -> str:

        if self.model is None:
            self.load_chat_model()

        config = self.config["chat"]

        if temperature is None:
            temperature = config["temperature"]

        response = self.model.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return response["choices"][0]["message"]["content"].strip()