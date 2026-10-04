from pathlib import Path
import gc
import json

from llama_cpp import Llama


class ModelManager:
    """
    Manages the local Qwen model lifecycle.

    Model lifecycle:

        load_chat_model()
              ↓
        generate(...)
              ↓
        unload()
              ↓
        GPU / CPU resources released
    """

    # =========================================================
    # DEFAULT SETTINGS
    # =========================================================

    # Qwen3-4B supports a much larger context, but 8192 is a
    # safer runtime target for the current 4 GB VRAM system.
    DEFAULT_CONTEXT_SIZE = 8192

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        config_path: str = "config/models.json",
    ):
        self.config_path = Path(config_path)

        self.model = None

        self.config = self._load_config()

    # =========================================================
    # CONFIGURATION
    # =========================================================

    def _load_config(self):
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Model configuration not found: "
                f"{self.config_path}"
            )

        with self.config_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    # =========================================================
    # LOAD MODEL
    # =========================================================

    def load_chat_model(self):
        """
        Load Qwen into memory/GPU.

        If the model is already loaded, do nothing.
        """

        # Prevent loading the same model twice.
        if self.model is not None:
            print("Qwen model already loaded.")
            return self.model

        config = self.config["chat"]

        model_path = Path(
            config["path"]
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        print()
        print("================================")
        print("LOADING CHAT MODEL")
        print("================================")

        print(
            f"Model: {config['name']}"
        )

        print(
            f"Path: {model_path}"
        )

        # -----------------------------------------------------
        # Context configuration
        # -----------------------------------------------------

        context_size = config.get(
            "context_size",
            self.DEFAULT_CONTEXT_SIZE,
        )

        # Make sure the configured value is valid.
        if context_size < 4096:
            print(
                "Warning: configured context size is "
                f"{context_size}. "
                "Using minimum safe context size of 4096."
            )

            context_size = 4096

        print(
            f"Context size: {context_size}"
        )

        # -----------------------------------------------------
        # GPU configuration
        # -----------------------------------------------------

        n_gpu_layers = config.get(
            "n_gpu_layers",
            -1,
        )

        flash_attention = config.get(
            "flash_attention",
            True,
        )

        print(
            f"GPU layers: {n_gpu_layers}"
        )

        print(
            f"Flash Attention: {flash_attention}"
        )

        # -----------------------------------------------------
        # Load llama.cpp model
        # -----------------------------------------------------

        self.model = Llama(
            model_path=str(model_path),

            # -------------------------------------------------
            # Context window
            # -------------------------------------------------
            #
            # 8192 is used instead of the model's full
            # 40960-token training context because this system
            # currently has only 4 GB VRAM.
            #
            n_ctx=context_size,

            # -------------------------------------------------
            # CUDA GPU offloading
            # -------------------------------------------------
            #
            # -1 = offload all possible layers
            #
            n_gpu_layers=n_gpu_layers,

            # -------------------------------------------------
            # KV cache
            # -------------------------------------------------

            offload_kqv=True,

            # -------------------------------------------------
            # Flash Attention
            # -------------------------------------------------

            flash_attn=flash_attention,

            # -------------------------------------------------
            # Batch sizes
            # -------------------------------------------------

            n_batch=config.get(
                "n_batch",
                512,
            ),

            n_ubatch=config.get(
                "n_ubatch",
                512,
            ),

            # -------------------------------------------------
            # CPU threads
            # -------------------------------------------------

            n_threads=config.get(
                "n_threads"
            ),

            n_threads_batch=config.get(
                "n_threads_batch"
            ),

            # -------------------------------------------------
            # llama.cpp logging
            # -------------------------------------------------

            verbose=config.get(
                "verbose",
                False,
            ),
        )

        print()
        print("Chat model loaded.")
        print("CUDA GPU offloading enabled.")
        print(
            f"Runtime context: {context_size} tokens"
        )
        print("================================")

        return self.model

    # =========================================================
    # GENERATE
    # =========================================================

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float | None = None,
        system_prompt: str = (
            "You are a helpful AI presentation assistant."
        ),
    ) -> str:
        """
        Generate text using Qwen.

        The model is automatically loaded if necessary.

        IMPORTANT:
        This method does NOT unload the model automatically.
        The caller should call unload() when the current
        generation phase is finished.
        """

        if not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty."
            )

        # Load automatically if needed.
        if self.model is None:
            self.load_chat_model()

        config = self.config["chat"]

        if temperature is None:
            temperature = config["temperature"]

        print()
        print("Generating with Qwen...")

        # -----------------------------------------------------
        # Generate
        # -----------------------------------------------------

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

        content = (
            response["choices"][0]["message"]["content"]
        )

        return self._clean_response(
            content
        )

    # =========================================================
    # UNLOAD MODEL
    # =========================================================

    def unload(self):
        """
        Completely unload the Qwen model.

        This is important on low-VRAM systems because
        llama.cpp may keep GPU/CPU resources alive while
        the Python object still exists.
        """

        if self.model is None:
            print("Chat model is already unloaded.")
            return

        print()
        print("================================")
        print("UNLOADING CHAT MODEL")
        print("================================")

        # Keep a temporary reference so we can explicitly
        # destroy the llama.cpp object before garbage collection.
        model = self.model

        # Remove our main reference first.
        self.model = None

        # Explicitly delete the llama.cpp object.
        del model

        # Run Python garbage collection.
        gc.collect()

        print("Chat model unloaded.")
        print("GPU/CPU resources released.")
        print("================================")

    # =========================================================
    # STATUS
    # =========================================================

    def is_loaded(self) -> bool:
        """
        Return True when Qwen is currently loaded.
        """

        return self.model is not None

    # =========================================================
    # RESPONSE CLEANING
    # =========================================================

    @staticmethod
    def _clean_response(
        response: str,
    ) -> str:
        """
        Remove Qwen's internal
        <think>...</think> section.
        """

        response = response.strip()

        # Complete thinking block
        if (
            "<think>" in response
            and "</think>" in response
        ):
            response = response.split(
                "</think>",
                1,
            )[1].strip()

        # Incomplete thinking block
        elif "<think>" in response:
            response = response.split(
                "<think>",
                1,
            )[0].strip()

        return response