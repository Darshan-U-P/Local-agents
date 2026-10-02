from pathlib import Path

from backend.assets.image_model_manager import ImageModelManager
from backend.assets.generators.flux_generator import FluxGenerator


MODEL_PATH = (
    r"D:\all project\Projects\AI projects\ai assentent"
    r"\Local_codex\models\image\flux1-schnell-Q2_K.gguf"
)


def main():
    manager = ImageModelManager(
        model_path=MODEL_PATH,
        offload_path="generated/assets/flux_offload_cache",
    )

    generator = FluxGenerator(
        model_manager=manager,
        output_dir="generated/assets",
    )

    output = generator.generate(
        prompt=(
            "A futuristic artificial intelligence presentation "
            "environment, clean modern technology laboratory, "
            "professional presentation visual, cinematic lighting"
        ),
        output_name="flux_production_test.png",
        width=1024,
        height=1024,
        steps=4,
        seed=42,
    )

    print()
    print("================================")
    print("FLUX PRODUCTION TEST COMPLETE")
    print("================================")
    print(f"Output: {output}")
    print(f"Exists: {Path(output).exists()}")


if __name__ == "__main__":
    main()