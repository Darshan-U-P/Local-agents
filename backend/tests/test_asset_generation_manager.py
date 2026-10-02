from pathlib import Path

from backend.generation.presentation_generator import (
    PresentationGenerator,
)

from backend.generation.asset_generation_manager import (
    AssetGenerationManager,
)


def main():
    # ---------------------------------------------------------
    # CREATE PRESENTATION SESSION
    # ---------------------------------------------------------

    generator = PresentationGenerator()

    session = generator.create(
        topic="Quantum Computing",
        slide_count=6,
    )

    # ---------------------------------------------------------
    # CREATE ASSET GENERATION MANAGER
    # ---------------------------------------------------------

    manager = AssetGenerationManager(
        session=session,
    )

    # ---------------------------------------------------------
    # GENERATE ASSETS
    # ---------------------------------------------------------

    manager.generate_all()

    # ---------------------------------------------------------
    # LOAD FINAL MANIFEST
    # ---------------------------------------------------------

    manifest = manager.manifest.load_assets()

    assets = manifest.get(
        "assets",
        [],
    )

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------

    print()
    print("================================")
    print("ASSET GENERATION TEST")
    print("================================")

    print(
        f"Total assets: {len(assets)}"
    )

    print()

    for asset in assets:

        print(
            f"{asset['id']} | "
            f"Slide {asset['slide_number']} | "
            f"{asset['type']} | "
            f"{asset['status']}"
        )

        if asset.get("path"):
            print(
                f"    Path: {asset['path']}"
            )

    # ---------------------------------------------------------
    # VERIFY IMAGE ASSETS
    # ---------------------------------------------------------

    image_assets = [
        asset
        for asset in assets
        if asset["type"] == "image"
    ]

    if len(image_assets) != 3:
        raise RuntimeError(
            "Expected 3 image assets."
        )

    for asset in image_assets:

        if asset["status"] != "completed":
            raise RuntimeError(
                f"{asset['id']} "
                f"was not completed."
            )

        if not asset.get("path"):
            raise RuntimeError(
                f"{asset['id']} "
                f"has no output path."
            )

        output_path = Path(
            asset["path"]
        )

        if not output_path.exists():
            raise RuntimeError(
                f"Generated file does not exist: "
                f"{output_path}"
            )

    # ---------------------------------------------------------
    # VERIFY NON-IMAGE ASSETS
    # ---------------------------------------------------------

    pending_types = {
        "icon",
        "diagram",
        "chart",
    }

    for asset in assets:

        if asset["type"] in pending_types:

            if asset["status"] != "pending":
                raise RuntimeError(
                    f"{asset['id']} should "
                    f"still be pending."
                )

    # ---------------------------------------------------------
    # SUCCESS
    # ---------------------------------------------------------

    print()
    print("================================")
    print("ASSET GENERATION TEST PASSED")
    print("================================")

    print()
    print("Generated image assets:")

    for asset in image_assets:
        print(
            f"  ✓ {asset['id']} "
            f"→ {asset['path']}"
        )

    print()
    print("Pending assets:")

    for asset in assets:

        if asset["type"] in pending_types:
            print(
                f"  ○ {asset['id']} "
                f"→ {asset['type']}"
            )


if __name__ == "__main__":
    main()