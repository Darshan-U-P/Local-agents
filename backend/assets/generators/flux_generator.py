from pathlib import Path
import traceback

import torch
from diffusers import (
    FluxPipeline,
    FluxTransformer2DModel,
    GGUFQuantizationConfig,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path(
    r"D:\all project\Projects\AI projects\ai assentent"
    r"\Local_codex\models\image\flux1-schnell-Q2_K.gguf"
)

BASE_MODEL = "black-forest-labs/FLUX.1-schnell"

OUTPUT_DIR = Path("generated/assets")

OUTPUT_FILE = OUTPUT_DIR / "flux_test.png"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FLUX IMAGE GENERATOR")
    print("=" * 70)

    # --------------------------------------------------------
    # Hardware check
    # --------------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = (
        torch.cuda.get_device_properties(0).total_memory
        / 1024**3
    )

    print(f"GPU: {gpu_name}")
    print(f"VRAM: {gpu_memory:.2f} GB")
    print()

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"FLUX GGUF model not found:\n{MODEL_PATH}"
        )

    print(f"Model: {MODEL_PATH}")
    print()

    # --------------------------------------------------------
    # Load local FLUX Q2_K transformer
    # --------------------------------------------------------

    print("=" * 70)
    print("LOADING LOCAL FLUX Q2_K TRANSFORMER")
    print("=" * 70)

    print("Loading transformer...")
    print()

    transformer = FluxTransformer2DModel.from_single_file(
        str(MODEL_PATH),
        quantization_config=GGUFQuantizationConfig(
            compute_dtype=torch.float16
        ),
        torch_dtype=torch.float16,
    )

    print()
    print("Transformer loaded successfully.")
    print()

    # --------------------------------------------------------
    # Build FLUX pipeline
    # --------------------------------------------------------

    print("=" * 70)
    print("LOADING FLUX PIPELINE")
    print("=" * 70)

    print(f"Base model: {BASE_MODEL}")
    print()

    pipe = FluxPipeline.from_pretrained(
        BASE_MODEL,
        transformer=transformer,
        torch_dtype=torch.float16,
    )

    print()
    print("Pipeline loaded successfully.")
    print()

    # --------------------------------------------------------
    # Enable low-VRAM CPU offloading
    # --------------------------------------------------------

    print("=" * 70)
    print("ENABLING LOW-VRAM MODE")
    print("=" * 70)

    pipe.enable_model_cpu_offload()

    print("CPU offload enabled.")
    print()

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = (
        "A futuristic artificial intelligence laboratory, "
        "clean modern technology, cinematic lighting, "
        "professional presentation illustration"
    )

    print("=" * 70)
    print("STARTING IMAGE GENERATION")
    print("=" * 70)

    print(f"Prompt: {prompt}")
    print("Resolution: 512x512")
    print("Steps: 4")
    print("Guidance scale: 0.0")
    print()

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    try:

        print("Creating random generator...")

        generator = torch.Generator(
            device="cpu"
        ).manual_seed(42)

        print("Starting FLUX inference...")
        print()

        result = pipe(
            prompt=prompt,
            guidance_scale=0.0,
            num_inference_steps=4,
            width=512,
            height=512,
            max_sequence_length=256,
            generator=generator,
        )

        print()
        print("FLUX inference completed.")

        # ----------------------------------------------------
        # Get generated image
        # ----------------------------------------------------

        if not result.images:
            raise RuntimeError(
                "FLUX returned no images."
            )

        image = result.images[0]

        print("Image received from pipeline.")

        # ----------------------------------------------------
        # Save image
        # ----------------------------------------------------

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        image.save(OUTPUT_FILE)

        print()
        print("=" * 70)
        print("IMAGE GENERATED SUCCESSFULLY")
        print("=" * 70)
        print(f"Output: {OUTPUT_FILE}")
        print("=" * 70)

    except Exception as e:

        print()
        print("=" * 70)
        print("FLUX INFERENCE FAILED")
        print("=" * 70)

        print(f"Error type: {type(e).__name__}")
        print(f"Error: {e}")

        print()
        print("FULL TRACEBACK")
        print("-" * 70)

        traceback.print_exc()

        print("=" * 70)

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()