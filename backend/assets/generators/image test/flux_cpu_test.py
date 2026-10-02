from pathlib import Path
import gc
import os
import time

# IMPORTANT:
# Disable optional optimized GGUF CUDA kernels for this diagnostic.
# This lets us test the standard PyTorch/Diffusers CUDA path.
os.environ["DIFFUSERS_GGUF_CUDA_KERNELS"] = "false"

import torch

from diffusers import (
    FluxPipeline,
    FluxTransformer2DModel,
    GGUFQuantizationConfig,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = Path(
    r"D:\all project\Projects\AI projects\ai assentent\Local_codex\models\image\flux1-schnell-Q2_K.gguf"
)

OUTPUT_PATH = Path(
    r"D:\all project\Projects\AI projects\ai assentent\Local-agents\generated\flux_gpu_offload_test.png"
)

PROMPT = (
    "A futuristic artificial intelligence laboratory, "
    "clean modern technology, cinematic lighting, "
    "professional presentation illustration"
)

WIDTH = 512
HEIGHT = 512
STEPS = 4
SEED = 42


# ============================================================
# GPU MEMORY
# ============================================================

def print_gpu_memory(label):

    if not torch.cuda.is_available():
        print(f"[GPU MEMORY] {label}: CUDA unavailable")
        return

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)

    free, total = torch.cuda.mem_get_info()

    free_gb = free / (1024 ** 3)
    total_gb = total / (1024 ** 3)

    print(
        f"[GPU MEMORY] {label}: "
        f"allocated={allocated:.2f} GB | "
        f"reserved={reserved:.2f} GB | "
        f"free={free_gb:.2f} GB / {total_gb:.2f} GB"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FLUX Q2_K — GPU + RAM OFFLOAD DIAGNOSTIC")
    print("=" * 70)

    print(f"Model : {MODEL_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"FLUX model not found:\n{MODEL_PATH}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("GPU:")
    print(torch.cuda.get_device_name(0))
    print()

    capability = torch.cuda.get_device_capability(0)

    print("CUDA capability:", capability)
    print("PyTorch version :", torch.__version__)
    print()

    print_gpu_memory("START")

    # --------------------------------------------------------
    # STEP 1
    # Load quantized FLUX transformer in CPU RAM
    # --------------------------------------------------------

    print()
    print("[1/5] Loading FLUX Q2_K transformer into RAM...")
    print()

    transformer = FluxTransformer2DModel.from_single_file(
        str(MODEL_PATH),

        quantization_config=GGUFQuantizationConfig(
            compute_dtype=torch.float16
        ),

        dtype=torch.float16,
    )

    print()
    print("Transformer loaded successfully.")
    print_gpu_memory("AFTER TRANSFORMER LOAD")

    # --------------------------------------------------------
    # STEP 2
    # Load pipeline
    # --------------------------------------------------------

    print()
    print("[2/5] Loading FLUX pipeline...")
    print()

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        transformer=transformer,
        dtype=torch.float16,
    )

    print()
    print("Pipeline loaded successfully.")

    print_gpu_memory("AFTER PIPELINE LOAD")

    # --------------------------------------------------------
    # STEP 3
    # Enable sequential CPU offload
    # --------------------------------------------------------

    print()
    print("[3/5] Enabling SEQUENTIAL CPU OFFLOAD...")
    print()

    # IMPORTANT:
    # Do NOT call pipe.to("cuda") before this.
    # Diffusers will move individual submodules to GPU
    # only when they are needed.
    pipe.enable_sequential_cpu_offload(
        device="cuda"
    )

    # VAE memory optimizations
    try:
        pipe.vae.enable_slicing()
        print("VAE slicing: enabled")
    except Exception as e:
        print("VAE slicing unavailable:", e)

    try:
        pipe.vae.enable_tiling()
        print("VAE tiling: enabled")
    except Exception as e:
        print("VAE tiling unavailable:", e)

    print()
    print("Sequential CPU offload enabled.")
    print("Model weights remain primarily in system RAM.")
    print("Required computation is moved to GPU dynamically.")

    print_gpu_memory("AFTER OFFLOAD SETUP")

    # --------------------------------------------------------
    # STEP 4
    # Generate
    # --------------------------------------------------------

    print()
    print("[4/5] STARTING GPU GENERATION")
    print("=" * 70)

    print("Prompt:")
    print(PROMPT)
    print()

    print(f"Resolution : {WIDTH}x{HEIGHT}")
    print(f"Steps      : {STEPS}")
    print(f"Seed       : {SEED}")
    print()

    generator = torch.Generator(
        device="cpu"
    ).manual_seed(SEED)

    print_gpu_memory("BEFORE INFERENCE")

    start = time.time()

    try:

        with torch.inference_mode():

            result = pipe(
                prompt=PROMPT,

                width=WIDTH,
                height=HEIGHT,

                num_inference_steps=STEPS,

                guidance_scale=0.0,

                generator=generator,
            )

        elapsed = time.time() - start

        image = result.images[0]

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        image.save(OUTPUT_PATH)

        print()
        print("=" * 70)
        print("GPU OFFLOAD TEST PASSED")
        print("=" * 70)

        print(f"Image saved : {OUTPUT_PATH}")
        print(f"Time        : {elapsed:.2f} seconds")

        print_gpu_memory("AFTER INFERENCE")

    except Exception as e:

        print()
        print("=" * 70)
        print("PYTHON EXCEPTION DURING GPU INFERENCE")
        print("=" * 70)

        print(type(e).__name__)
        print(str(e))

        raise

    finally:

        print()
        print("[5/5] Cleaning up...")

        try:
            pipe.to("cpu")
        except Exception:
            pass

        del pipe
        del transformer

        gc.collect()

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        print_gpu_memory("AFTER CLEANUP")

        print()
        print("=" * 70)
        print("TEST FINISHED")
        print("=" * 70)


if __name__ == "__main__":
    main()