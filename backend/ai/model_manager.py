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

        # GPU configuration
        n_gpu_layers = config.get("n_gpu_layers", -1)
        flash_attention = config.get("flash_attention", True)

        print(f"GPU layers: {n_gpu_layers}")
        print(f"Flash Attention: {flash_attention}")

        self.model = Llama(
            model_path=str(model_path),

            # Context window
            n_ctx=config["context_size"],

            # CUDA GPU offloading
            # -1 = offload all possible layers
            n_gpu_layers=n_gpu_layers,

            # Keep KV cache operations on GPU when possible
            offload_kqv=True,

            # Use Flash Attention
            flash_attn=flash_attention,

            # Batch size
            n_batch=config.get("n_batch", 512),
            n_ubatch=config.get("n_ubatch", 512),

            # Let llama.cpp use available CPU threads
            n_threads=config.get("n_threads"),
            n_threads_batch=config.get("n_threads_batch"),

            # Reduce unnecessary llama.cpp console output
            verbose=config.get("verbose", False),
        )

        print("Model loaded.")
        print("CUDA GPU offloading enabled.")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float | None = None,
        system_prompt: str = (
            "You are a helpful AI presentation assistant."
        ),
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

        content = response["choices"][0]["message"]["content"]

        return self._clean_response(content)

    @staticmethod
    def _clean_response(response: str) -> str:
        """
        Remove Qwen's internal <think>...</think> section
        from the returned response.
        """

        response = response.strip()

        # Remove complete thinking block
        if "<think>" in response and "</think>" in response:
            response = response.split("</think>", 1)[1].strip()

        # Handle an incomplete thinking block safely
        elif "<think>" in response:
            response = response.split("<think>", 1)[0].strip()

        return response