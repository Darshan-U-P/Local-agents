from pathlib import Path

from backend.generation.presentation_generator import (
    PresentationGenerator,
)

from backend.generation.asset_generation_manager import (
    AssetGenerationManager,
)


def main():
    # =========================================================
    # HEADER
    # =========================================================

    print()
    print("========================================")
    print("ASSET GENERATION MANAGER TEST")
    print("========================================")
    print()

    # =========================================================
    # CREATE PRESENTATION SESSION
    # =========================================================

    print("Creating Quantum Computing presentation...")

    generator = PresentationGenerator()

    session = generator.create(
        topic="Quantum Computing",
        slide_count=6,
    )

    print("Presentation session created.")
    print()

    # =========================================================
    # CREATE ASSET GENERATION MANAGER
    # =========================================================

    print("Creating AssetGenerationManager...")

    manager = AssetGenerationManager(
        session=session,
    )

    print("AssetGenerationManager created.")
    print()

    # =========================================================
    # GENERATE ASSETS
    # =========================================================

    print("Generating assets...")
    print()

    manager.generate_all()

    # =========================================================
    # LOAD FINAL MANIFEST
    # =========================================================

    manifest = manager.manifest.load_assets()

    assets = manifest.get(
        "assets",
        [],
    )

    # =========================================================
    # BASIC VALIDATION
    # =========================================================

    if not assets:
        raise RuntimeError(
            "Asset manifest contains no assets."
        )

    # =========================================================
    # RESULTS
    # =========================================================

    print()
    print("========================================")
    print("ASSET GENERATION RESULTS")
    print("========================================")
    print()

    print(
        f"Total assets: {len(assets)}"
    )

    print()

    # =========================================================
    # PRINT ASSET RESULTS
    # =========================================================

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

        if asset.get("source"):
            print(
                f"    Source: {asset['source']}"
            )

        if asset.get("model"):
            print(
                f"    Model: {asset['model']}"
            )

        print()

    # =========================================================
    # IMAGE ASSET VALIDATION
    # =========================================================

    image_types = {
        "image",
        "illustration",
        "photo",
    }

    image_assets = [
        asset
        for asset in assets
        if asset.get("type", "").strip().lower()
        in image_types
    ]

    # At least one image-like asset must exist.
    if not image_assets:
        raise RuntimeError(
            "Expected at least one image-like asset."
        )

    print(
        f"Image-like assets: {len(image_assets)}"
    )

    print()

    # =========================================================
    # VERIFY IMAGE ASSETS
    # =========================================================

    for asset in image_assets:

        asset_id = asset["id"]

        print(
            f"Validating {asset_id}..."
        )

        # -----------------------------------------------------
        # STATUS
        # -----------------------------------------------------

        if asset.get("status") != "completed":
            raise RuntimeError(
                f"{asset_id} was not completed."
            )

        # -----------------------------------------------------
        # OUTPUT PATH
        # -----------------------------------------------------

        if not asset.get("path"):
            raise RuntimeError(
                f"{asset_id} has no output path."
            )

        output_path = Path(
            asset["path"]
        )

        # -----------------------------------------------------
        # FILE EXISTS
        # -----------------------------------------------------

        if not output_path.exists():
            raise RuntimeError(
                f"Generated file does not exist: "
                f"{output_path}"
            )

        # -----------------------------------------------------
        # FILE NOT EMPTY
        # -----------------------------------------------------

        if output_path.stat().st_size == 0:
            raise RuntimeError(
                f"Generated file is empty: "
                f"{output_path}"
            )

        # -----------------------------------------------------
        # SOURCE
        # -----------------------------------------------------

        source = asset.get("source")

        if source != "flux2-klein":
            raise RuntimeError(
                f"{asset_id} has unexpected source: "
                f"{source}"
            )

        # -----------------------------------------------------
        # MODEL
        # -----------------------------------------------------

        model = asset.get("model")

        if not model:
            raise RuntimeError(
                f"{asset_id} has no model metadata."
            )

        if "flux-2-klein" not in model.lower():
            raise RuntimeError(
                f"{asset_id} has unexpected model: "
                f"{model}"
            )

        # -----------------------------------------------------
        # RUNTIME
        # -----------------------------------------------------

        runtime = asset.get("runtime")

        if runtime != "stable-diffusion.cpp":
            raise RuntimeError(
                f"{asset_id} has unexpected runtime: "
                f"{runtime}"
            )

        # -----------------------------------------------------
        # PROMPT
        # -----------------------------------------------------

        if not asset.get("prompt"):
            raise RuntimeError(
                f"{asset_id} has no generation prompt."
            )

        print(
            f"    ✓ Status: completed"
        )

        print(
            f"    ✓ File exists: {output_path}"
        )

        print(
            f"    ✓ File size: "
            f"{output_path.stat().st_size:,} bytes"
        )

        print(
            f"    ✓ Source: {source}"
        )

        print(
            f"    ✓ Model: {model}"
        )

        print(
            f"    ✓ Runtime: {runtime}"
        )

        print()

    # =========================================================
    # VERIFY NONE ASSETS
    # =========================================================

    none_assets = [
        asset
        for asset in assets
        if asset.get("type", "").strip().lower()
        == "none"
    ]

    print(
        f"None assets: {len(none_assets)}"
    )

    for asset in none_assets:

        if asset.get("status") != "completed":
            raise RuntimeError(
                f"{asset['id']} none asset "
                f"was not completed."
            )

        if asset.get("source") != "none":
            raise RuntimeError(
                f"{asset['id']} none asset has "
                f"unexpected source: "
                f"{asset.get('source')}"
            )

        if asset.get("path") is not None:
            raise RuntimeError(
                f"{asset['id']} none asset should "
                f"not have an output path."
            )

    # =========================================================
    # VERIFY DUPLICATE ASSET IDS
    # =========================================================

    asset_ids = [
        asset.get("id")
        for asset in assets
    ]

    if len(asset_ids) != len(set(asset_ids)):
        raise RuntimeError(
            "Duplicate asset IDs detected."
        )

    # =========================================================
    # SUMMARY
    # =========================================================

    print()
    print("========================================")
    print("ASSET GENERATION TEST PASSED")
    print("========================================")
    print()

    print(
        f"Total assets      : {len(assets)}"
    )

    print(
        f"Image-like assets : {len(image_assets)}"
    )

    print(
        f"None assets       : {len(none_assets)}"
    )

    print()
    print("FLUX.2 Klein asset generation is working.")
    print()


if __name__ == "__main__":
    main()