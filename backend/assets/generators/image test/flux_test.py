import torch

from diffusers import (
    FluxTransformer2DModel,
    GGUFQuantizationConfig,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL_PATH = r"D:\all project\Projects\AI projects\ai assentent\Local_codex\models\image\flux1-schnell-Q2_K.gguf"


# ==========================================================
# STARTUP INFORMATION
# ==========================================================

print()
print("=" * 70)
print("FLUX GGUF TEST")
print("=" * 70)

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")

    gpu_properties = torch.cuda.get_device_properties(0)

    print(
        f"VRAM    : "
        f"{gpu_properties.total_memory / 1024**3:.2f} GB"
    )

print(f"Model   : {MODEL_PATH}")

print("=" * 70)


# ==========================================================
# CHECK MODEL
# ==========================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. "
        "FLUX generation requires CUDA for this setup."
    )


import os

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        "\nFLUX GGUF model was not found:\n"
        f"{MODEL_PATH}\n\n"
        "Check that the .gguf file exists at exactly this path."
    )


# ==========================================================
# LOAD FLUX GGUF
# ==========================================================

print()
print("=" * 70)
print("LOADING FLUX Q2_K GGUF")
print("=" * 70)

print("Loading transformer...")
print()
print(MODEL_PATH)
print()

try:

    transformer = FluxTransformer2DModel.from_single_file(
        MODEL_PATH,

        quantization_config=GGUFQuantizationConfig(
            compute_dtype=torch.float16
        ),

        torch_dtype=torch.float16,
    )

except Exception as e:

    print()
    print("=" * 70)
    print("FLUX LOAD FAILED")
    print("=" * 70)

    print(type(e).__name__)
    print(e)

    raise


# ==========================================================
# SUCCESS
# ==========================================================

print()
print("=" * 70)
print("FLUX TRANSFORMER LOADED SUCCESSFULLY")
print("=" * 70)

print()
print("Model : FLUX.1-schnell Q2_K")
print("Format: GGUF")
print("Compute dtype: float16")
print("CUDA: True")
print()

print("Next step:")
print("Load VAE + CLIP-L + T5-XXL")
print("and construct the complete FluxPipeline.")

print("=" * 70)