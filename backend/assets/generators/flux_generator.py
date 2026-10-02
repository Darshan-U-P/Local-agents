from pathlib import Path

import torch

from backend.assets.image_model_manager import ImageModelManager


class FluxGenerator:
    """
    Generates images using the locally loaded FLUX model.

    The FLUX model and its disk offload cache are kept alive
    across multiple image generations.

    Lifecycle is managed by ImageModelManager:

        load FLUX
            ↓
        generate image 1
            ↓
        generate image 2
            ↓
        generate image 3
            ↓
        unload FLUX
            ↓
        delete offload cache
    """

    def __init__(
        self,
        model_manager: ImageModelManager,
        output_dir: str = "generated/assets",
    ):
        self.model_manager = model_manager
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def generate(
        self,
        prompt: str,
        output_name: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = 4,
        seed: int = 42,
    ) -> str:
        """
        Generate one image using the already-managed FLUX pipeline.

        Important:
        This method DOES NOT unload FLUX and DOES NOT delete the
        disk offload cache.

        The cache must remain available when multiple images are
        generated in the same FLUX session.
        """

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        output_path = self.output_dir / output_name

        # Load FLUX only if it is not already loaded.
        # If the AssetGenerationManager is processing multiple
        # images, the same pipeline will be reused.
        pipe = self.model_manager.get_pipeline()

        # Keep the random generator on CPU.
        generator = torch.Generator(
            device="cpu"
        ).manual_seed(seed)

        print()
        print("Generating image...")
        print(f"Prompt: {prompt}")
        print(f"Resolution: {width}x{height}")
        print(f"Steps: {steps}")
        print(f"Seed: {seed}")

        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=0.0,
                generator=generator,
            )

        image = result.images[0]

        image.save(output_path)

        print(f"Image saved: {output_path}")

        return str(output_path)