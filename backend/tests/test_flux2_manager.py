from pathlib import Path

from backend.generation.asset_generation_manager import Flux2KleinManager


ROOT = Path(
    r"D:\all project\Projects\AI projects\ai assentent\Local-agents"
)

OUTPUT = (
    ROOT
    / "generated"
    / "assets"
    / "python_flux2_test.png"
)


def main():
    manager = Flux2KleinManager(
        output_dir=OUTPUT.parent
    )

    manager.load()

    prompt = (
        "A highly detailed scientific visualization of a "
        "superconducting quantum processor inside a dilution "
        "refrigerator. Show a realistic quantum processor "
        "at the center, surrounded by cryogenic components "
        "and control wiring. Professional scientific "
        "visualization, realistic engineering hardware, "
        "metallic surfaces, dark laboratory environment, "
        "blue and white lighting, accurate physical "
        "proportions, no text, no labels, no logos, "
        "no watermark."
    )

    result = manager.generate(
        prompt=prompt,
        output_path=OUTPUT,
        seed=42,
        width=512,
        height=512,
        steps=4,
    )

    print()
    print("=" * 60)
    print("FLUX.2 PYTHON INTEGRATION TEST PASSED")
    print("=" * 60)
    print(f"Output: {result}")

    manager.unload()


if __name__ == "__main__":
    main()