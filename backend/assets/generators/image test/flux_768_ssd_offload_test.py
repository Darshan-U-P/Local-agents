from pathlib import Path
import gc
import shutil
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
    r"D:\all project\Projects\AI projects\ai assentent\Local-agents\generated\flux_1024_ssd_test.png"
)

# Temporary SSD offload cache.
# This will be DELETED automatically after generation.
SSD_OFFLOAD_PATH = Path(
    r"D:\all project\Projects\AI projects\ai assentent\Local-agents\generated\flux_offload_cache"
)

PROMPT = (
    "A futuristic artificial intelligence laboratory, "
    "clean modern technology, cinematic lighting, "
    "professional presentation illustration"
)


# ============================================================
# IMAGE SETTINGS
# ============================================================

WIDTH = 1024
HEIGHT = 1024

STEPS = 4

SEED = 42


# ============================================================
# GPU MEMORY MONITOR
# ============================================================

def gpu_memory(label: str):

    if not torch.cuda.is_available():
        print(
            f"[GPU] {label}: CUDA unavailable"
        )
        return

    allocated = (
        torch.cuda.memory_allocated()
        / (1024 ** 3)
    )

    reserved = (
        torch.cuda.memory_reserved()
        / (1024 ** 3)
    )

    free, total = torch.cuda.mem_get_info()

    free_gb = free / (1024 ** 3)
    total_gb = total / (1024 ** 3)

    print(
        f"[GPU] {label}: "
        f"allocated={allocated:.2f} GB | "
        f"reserved={reserved:.2f} GB | "
        f"free={free_gb:.2f} GB / "
        f"{total_gb:.2f} GB"
    )


# ============================================================
# SSD INFORMATION
# ============================================================

def print_ssd_info():

    SSD_OFFLOAD_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("SSD OFFLOAD DIRECTORY:")
    print(SSD_OFFLOAD_PATH)
    print()


# ============================================================
# DELETE SSD CACHE
# ============================================================

def cleanup_ssd_cache():

    print()
    print("Removing temporary SSD offload cache...")

    if not SSD_OFFLOAD_PATH.exists():

        print(
            "SSD offload cache does not exist."
        )

        return

    try:

        shutil.rmtree(
            SSD_OFFLOAD_PATH
        )

        print(
            "SSD offload cache deleted successfully."
        )

    except Exception as e:

        print()
        print(
            "WARNING: Could not completely delete "
            "SSD offload cache."
        )

        print(
            f"Reason: {e}"
        )

        print()
        print(
            "The cache can be manually deleted later:"
        )

        print(
            SSD_OFFLOAD_PATH
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "FLUX Q2_K — 1024x1024 GPU + RAM + SSD OFFLOAD TEST"
    )
    print("=" * 75)

    # ========================================================
    # BASIC CHECKS
    # ========================================================

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"FLUX model not found:\n{MODEL_PATH}"
        )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )

    print()
    print("GPU:")
    print(
        torch.cuda.get_device_name(0)
    )

    print()
    print("PyTorch:")
    print(
        torch.__version__
    )

    print()
    print("CUDA capability:")
    print(
        torch.cuda.get_device_capability(0)
    )

    print()
    print("Resolution:")
    print(
        f"{WIDTH} x {HEIGHT}"
    )

    print()
    print("Steps:")
    print(
        STEPS
    )

    print()
    print("SSD offloading:")
    print("ENABLED")

    print()
    print("SSD cache:")
    print(
        SSD_OFFLOAD_PATH
    )

    print_ssd_info()

    gpu_memory(
        "START"
    )

    transformer = None
    pipe = None

    generation_success = False

    try:

        # ====================================================
        # STEP 1
        # Load FLUX GGUF transformer
        # ====================================================

        print()
        print(
            "[1/6] Loading FLUX Q2_K transformer..."
        )
        print()

        transformer = (
            FluxTransformer2DModel.from_single_file(

                str(MODEL_PATH),

                quantization_config=(
                    GGUFQuantizationConfig(
                        compute_dtype=torch.float16
                    )
                ),

                torch_dtype=torch.float16,
            )
        )

        print()
        print(
            "Transformer loaded successfully."
        )

        gpu_memory(
            "AFTER TRANSFORMER"
        )

        # ====================================================
        # STEP 2
        # Load pipeline
        # ====================================================

        print()
        print(
            "[2/6] Loading FLUX pipeline..."
        )
        print()

        pipe = FluxPipeline.from_pretrained(

            "black-forest-labs/FLUX.1-schnell",

            transformer=transformer,

            torch_dtype=torch.float16,
        )

        print()
        print(
            "Pipeline loaded successfully."
        )

        gpu_memory(
            "AFTER PIPELINE"
        )

        # ====================================================
        # STEP 3
        # Prepare CPU/RAM offloading
        # ====================================================

        print()
        print(
            "[3/6] Preparing GPU + RAM + SSD offloading..."
        )
        print()

        # The model stays primarily in system RAM.
        #
        # Individual groups are moved to the GPU only
        # when required.
        #
        # We intentionally do NOT use:
        #
        #     pipe.to("cuda")
        #
        # because that would defeat the offloading strategy.

        pipe = pipe.to("cpu")

        print(
            "Pipeline placed in CPU/RAM."
        )

        # ====================================================
        # STEP 4
        # Enable group offloading + SSD
        # ====================================================

        print()
        print(
            "[4/6] Enabling group offloading with SSD..."
        )
        print()

        pipe.enable_group_offload(

            onload_device=torch.device(
                "cuda"
            ),

            offload_device=torch.device(
                "cpu"
            ),

            offload_type="leaf_level",

            use_stream=False,

            record_stream=False,

            low_cpu_mem_usage=True,

            offload_to_disk_path=str(
                SSD_OFFLOAD_PATH
            ),
        )

        print()
        print(
            "GPU + RAM + SSD offloading ENABLED."
        )

        print()
        print(
            "Memory architecture:"
        )

        print()
        print(
            "       SSD"
        )

        print(
            "        ↕"
        )

        print(
            "       RAM"
        )

        print(
            "        ↕"
        )

        print(
            "       VRAM"
        )

        print(
            "        ↓"
        )

        print(
            " RTX 3050 Ti"
        )

        # ====================================================
        # VAE MEMORY OPTIMIZATION
        # ====================================================

        try:

            pipe.vae.enable_slicing()

            print()
            print(
                "VAE slicing: enabled"
            )

        except Exception as e:

            print()
            print(
                "VAE slicing unavailable:"
            )

            print(e)

        try:

            pipe.vae.enable_tiling()

            print(
                "VAE tiling: enabled"
            )

        except Exception as e:

            print(
                "VAE tiling unavailable:"
            )

            print(e)

        gpu_memory(
            "AFTER OFFLOAD SETUP"
        )

        # ====================================================
        # STEP 5
        # GENERATION
        # ====================================================

        print()
        print("=" * 75)
        print(
            "[5/6] STARTING 1024x1024 GPU GENERATION"
        )
        print("=" * 75)

        print()
        print(
            "Prompt:"
        )

        print(
            PROMPT
        )

        print()
        print(
            f"Resolution : {WIDTH}x{HEIGHT}"
        )

        print(
            f"Steps      : {STEPS}"
        )

        print(
            f"Seed       : {SEED}"
        )

        print()
        print(
            "GPU + RAM + SSD offloading active."
        )

        print()

        # CPU generator is intentional.
        generator = torch.Generator(
            device="cpu"
        ).manual_seed(
            SEED
        )

        gpu_memory(
            "BEFORE INFERENCE"
        )

        start_time = time.time()

        # ----------------------------------------------------
        # Run inference
        # ----------------------------------------------------

        with torch.inference_mode():

            result = pipe(

                prompt=PROMPT,

                width=WIDTH,

                height=HEIGHT,

                num_inference_steps=STEPS,

                guidance_scale=0.0,

                generator=generator,
            )

        elapsed = (
            time.time()
            - start_time
        )

        # ----------------------------------------------------
        # Save image
        # ----------------------------------------------------

        image = result.images[0]

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        image.save(
            OUTPUT_PATH
        )

        generation_success = True

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        print()
        print("=" * 75)
        print(
            "GPU + RAM + SSD OFFLOAD TEST PASSED"
        )
        print("=" * 75)

        print()
        print(
            "Image:"
        )

        print(
            OUTPUT_PATH
        )

        print()
        print(
            f"Generation time: {elapsed:.2f} seconds"
        )

        print()

        gpu_memory(
            "AFTER INFERENCE"
        )

    except Exception as e:

        print()
        print("=" * 75)
        print(
            "GPU GENERATION FAILED"
        )
        print("=" * 75)

        print()
        print(
            "Exception type:"
        )

        print(
            type(e).__name__
        )

        print()
        print(
            "Message:"
        )

        print(
            str(e)
        )

        print()

        raise

    finally:

        # ====================================================
        # STEP 6
        # COMPLETE CLEANUP
        # ====================================================

        print()
        print(
            "[6/6] Cleaning up GPU/RAM/SSD resources..."
        )

        # ----------------------------------------------------
        # Release pipeline
        #
        # IMPORTANT:
        # Do NOT call pipe.to("cpu") here.
        #
        # Group-offloaded modules have hooks attached and
        # calling .to() generates the warnings you saw.
        # ----------------------------------------------------

        if pipe is not None:

            del pipe

            pipe = None

        if transformer is not None:

            del transformer

            transformer = None

        # ----------------------------------------------------
        # Python garbage collection
        # ----------------------------------------------------

        gc.collect()

        # ----------------------------------------------------
        # CUDA cleanup
        # ----------------------------------------------------

        torch.cuda.empty_cache()

        try:

            torch.cuda.ipc_collect()

        except Exception:

            pass

        gpu_memory(
            "AFTER GPU CLEANUP"
        )

        # ----------------------------------------------------
        # Give Windows a moment to release file handles
        # ----------------------------------------------------

        time.sleep(
            1
        )

        # ----------------------------------------------------
        # Delete temporary SSD cache
        # ----------------------------------------------------

        cleanup_ssd_cache()

        # ----------------------------------------------------
        # Final CUDA cleanup
        # ----------------------------------------------------

        gc.collect()

        torch.cuda.empty_cache()

        try:

            torch.cuda.ipc_collect()

        except Exception:

            pass

        print()

        gpu_memory(
            "FINAL"
        )

        # ----------------------------------------------------
        # Final status
        # ----------------------------------------------------

        print()
        print("=" * 75)

        if generation_success:

            print(
                "GENERATION + CLEANUP COMPLETE"
            )

        else:

            print(
                "GENERATION FAILED — CLEANUP COMPLETE"
            )

        print("=" * 75)

        print()

        if generation_success:

            print(
                "Generated image:"
            )

            print(
                OUTPUT_PATH
            )

            print()

            print(
                "Temporary SSD cache:"
            )

            print(
                "DELETED"
            )

        print()
        print("=" * 75)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()