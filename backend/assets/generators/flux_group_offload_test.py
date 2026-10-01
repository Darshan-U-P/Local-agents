from pathlib import Path
import gc
import time

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
    r"D:\all project\Projects\AI projects\ai assentent\Local-agents\generated\flux_group_offload_test.png"
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

def gpu_memory(label):

    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)

    free, total = torch.cuda.mem_get_info()

    print(
        f"[GPU] {label}: "
        f"allocated={allocated:.2f} GB | "
        f"reserved={reserved:.2f} GB | "
        f"free={free / (1024 ** 3):.2f} GB / "
        f"{total / (1024 ** 3):.2f} GB"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FLUX Q2_K — GPU + RAM GROUP OFFLOAD TEST")
    print("=" * 70)

    print()
    print("GPU:")
    print(torch.cuda.get_device_name(0))

    print("PyTorch:", torch.__version__)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    gpu_memory("START")

    # --------------------------------------------------------
    # STEP 1
    # Load FLUX GGUF transformer
    # --------------------------------------------------------

    print()
    print("[1/5] Loading FLUX Q2_K transformer...")
    print()

    transformer = FluxTransformer2DModel.from_single_file(
        str(MODEL_PATH),

        quantization_config=GGUFQuantizationConfig(
            compute_dtype=torch.float16
        ),

        torch_dtype=torch.float16,
    )

    print("Transformer loaded.")
    gpu_memory("AFTER TRANSFORMER")

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
        torch_dtype=torch.float16,
    )

    print()
    print("Pipeline loaded.")

    gpu_memory("AFTER PIPELINE")

    # --------------------------------------------------------
    # STEP 3
    # Move pipeline to CPU first
    # --------------------------------------------------------

    print()
    print("[3/5] Preparing CPU/GPU group offloading...")
    print()

    pipe = pipe.to("cpu")

    # --------------------------------------------------------
    # Group offload
    # --------------------------------------------------------

    print("Enabling group offload...")

    pipe.enable_group_offload(
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        offload_type="leaf_level",
        use_stream=False,
    )

    print("Group offload enabled.")

    # VAE optimizations
    try:
        pipe.vae.enable_slicing()
        print("VAE slicing enabled.")
    except Exception as e:
        print("VAE slicing unavailable:", e)

    try:
        pipe.vae.enable_tiling()
        print("VAE tiling enabled.")
    except Exception as e:
        print("VAE tiling unavailable:", e)

    gpu_memory("AFTER OFFLOAD SETUP")

    # --------------------------------------------------------
    # STEP 4
    # Generate
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("[4/5] STARTING GPU GENERATION")
    print("=" * 70)

    print()
    print("Prompt:")
    print(PROMPT)

    print()
    print(f"Resolution : {WIDTH}x{HEIGHT}")
    print(f"Steps      : {STEPS}")
    print(f"Seed       : {SEED}")

    generator = torch.Generator(
        device="cpu"
    ).manual_seed(SEED)

    gpu_memory("BEFORE INFERENCE")

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
        print("GPU + RAM GROUP OFFLOAD TEST PASSED")
        print("=" * 70)

        print()
        print("Image:", OUTPUT_PATH)
        print(f"Time : {elapsed:.2f} seconds")

        gpu_memory("AFTER INFERENCE")

    except Exception as e:

        print()
        print("=" * 70)
        print("GPU INFERENCE FAILED")
        print("=" * 70)

        print()
        print("Exception:", type(e).__name__)
        print("Message  :", str(e))

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

        gpu_memory("AFTER CLEANUP")

        print()
        print("=" * 70)
        print("TEST FINISHED")
        print("=" * 70)


if __name__ == "__main__":
    main()