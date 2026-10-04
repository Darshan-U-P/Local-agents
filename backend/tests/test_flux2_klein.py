import subprocess
import time
from pathlib import Path


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# We will update this after sd-cli.exe is downloaded.
SD_CLI = (
    PROJECT_ROOT
    / "tools"
    / "stable-diffusion.cpp"
    / "bin"
    / "Release"
    / "sd-cli.exe"
)

MODEL_DIR = Path(
    r"D:\all project\Projects\AI projects\ai assentent"
    r"\Local_codex\models\image\flux2-klein-4b"
)

DIFFUSION_MODEL = (
    MODEL_DIR / "flux-2-klein-4b-Q4_0.gguf"
)

QWEN_MODEL = (
    MODEL_DIR / "Qwen3-4B-Q4_K_M.gguf"
)

# Use this if you downloaded the small-decoder file.
VAE_MODEL = (
    MODEL_DIR / "full_encoder_small_decoder.safetensors"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "generated"
    / "assets"
    / "flux2_klein_test.png"
)


# ============================================================
# Generation settings
# ============================================================

WIDTH = 512
HEIGHT = 512

STEPS = 4

CFG_SCALE = 1.0

SEED = 42


# ============================================================
# Prompt
# ============================================================

PROMPT = (
    "A highly detailed scientific visualization of a "
    "superconducting quantum processor inside a dilution "
    "refrigerator. Show a realistic quantum processor at "
    "the center, surrounded by cryogenic components and "
    "control wiring. Professional scientific visualization, "
    "realistic engineering hardware, metallic surfaces, "
    "dark laboratory environment, blue and white lighting, "
    "accurate physical proportions, no text, no labels, "
    "no logos, no watermark."
)


# ============================================================
# Helpers
# ============================================================

def check_file(path: Path, name: str):

    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

    size_gb = path.stat().st_size / (1024 ** 3)

    print(
        f"{name}: {path.name} "
        f"({size_gb:.2f} GB)"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("FLUX.2 Klein 4B - Q4_0 GGUF Test")
    print("=" * 70)

    print()
    print("Project:")
    print(PROJECT_ROOT)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CHECKING REQUIRED FILES")
    print("=" * 70)

    check_file(
        SD_CLI,
        "stable-diffusion.cpp"
    )

    check_file(
        DIFFUSION_MODEL,
        "FLUX.2 Klein Q4_0"
    )

    check_file(
        QWEN_MODEL,
        "Qwen3-4B"
    )

    check_file(
        VAE_MODEL,
        "FLUX.2 VAE"
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)

    print(f"Runtime      : {SD_CLI}")
    print(f"Diffusion    : {DIFFUSION_MODEL}")
    print(f"Qwen         : {QWEN_MODEL}")
    print(f"VAE          : {VAE_MODEL}")

    print()
    print(f"Resolution   : {WIDTH}x{HEIGHT}")
    print(f"Steps        : {STEPS}")
    print(f"CFG scale    : {CFG_SCALE}")
    print(f"Seed         : {SEED}")

    print()
    print("Backend      : CUDA")
    print("Offload      : CPU")
    print("Flash Attn   : enabled")

    print()
    print("Prompt:")
    print(PROMPT)

    # --------------------------------------------------------
    # Build command
    # --------------------------------------------------------

    command = [

        str(SD_CLI),

        # FLUX.2 Klein Q4_0
        "--diffusion-model",
        str(DIFFUSION_MODEL),

        # Qwen3 4B
        "--llm",
        str(QWEN_MODEL),

        # VAE
        "--vae",
        str(VAE_MODEL),

        # CUDA
        "--backend",
        "cuda",

        # RAM offload
        "--offload-to-cpu",

        # Flash attention
        "--diffusion-fa",

        # FLUX.2 Klein settings
        "--cfg-scale",
        str(CFG_SCALE),

        "--sampling-method",
        "euler",

        "--steps",
        str(STEPS),

        # Image size
        "-W",
        str(WIDTH),

        "-H",
        str(HEIGHT),

        # Reproducibility
        "--seed",
        str(SEED),

        # Prompt
        "-p",
        PROMPT,

        # Output
        "-o",
        str(OUTPUT_PATH),

        # Verbose
        "-v",
    ]

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STARTING FLUX.2 KLEIN")
    print("=" * 70)

    print()
    print("Running stable-diffusion.cpp...")
    print()

    start_time = time.time()

    result = subprocess.run(
        command,
        cwd=SD_CLI.parent,
    )

    elapsed = time.time() - start_time

    # --------------------------------------------------------
    # Check result
    # --------------------------------------------------------

    print()
    print("=" * 70)

    if result.returncode != 0:

        print("FLUX.2 KLEIN FAILED")

        print("=" * 70)

        print()
        print(
            f"Exit code: {result.returncode}"
        )

        raise RuntimeError(
            "stable-diffusion.cpp failed."
        )

    # --------------------------------------------------------
    # Verify output
    # --------------------------------------------------------

    if not OUTPUT_PATH.exists():

        print("FLUX.2 KLEIN FAILED")

        print("=" * 70)

        raise RuntimeError(
            "sd-cli completed without creating "
            f"the expected image:\n{OUTPUT_PATH}"
        )

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    file_size_mb = (
        OUTPUT_PATH.stat().st_size
        / (1024 ** 2)
    )

    print("FLUX.2 KLEIN SUCCESS")

    print("=" * 70)

    print()
    print(f"Output       : {OUTPUT_PATH}")
    print(f"Resolution   : {WIDTH}x{HEIGHT}")
    print(f"Steps        : {STEPS}")
    print(f"CFG          : {CFG_SCALE}")
    print(f"Seed         : {SEED}")
    print(f"Time         : {elapsed:.2f}s")
    print(f"File size    : {file_size_mb:.2f} MB")

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()