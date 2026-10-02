from backend.ir.presentation_ir import (
    PresentationIR,
    ThemeIR,
    SlideIR,
    AssetIR,
)

from backend.assets.asset_router import AssetRouter


def main():
    # ---------------------------------------------------------
    # CREATE TEST ASSET
    # ---------------------------------------------------------

    asset = AssetIR(
        id="asset-router-test",
        asset_type="image",
        description=(
            "A futuristic AI presentation scene, "
            "modern computer laboratory, "
            "artificial intelligence visualization, "
            "professional corporate presentation style"
        ),
    )

    # ---------------------------------------------------------
    # CREATE TEST PRESENTATION
    # ---------------------------------------------------------

    presentation = PresentationIR(
        title="Asset Router Test",
        subtitle="Testing local FLUX integration",
        slide_count=1,
        theme=ThemeIR(),
        slides=[
            SlideIR(
                id="slide-001",
                slide_number=1,
                title="Asset Router Test",
                purpose="Test FLUX image generation",
                layout="full_image",
                asset_ids=["asset-router-test"],
            )
        ],
        assets=[asset],
    )

    # ---------------------------------------------------------
    # CREATE ROUTER
    # ---------------------------------------------------------

    router = AssetRouter()

    # ---------------------------------------------------------
    # ROUTE ASSETS
    # ---------------------------------------------------------

    presentation = router.route(
        presentation
    )

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------

    print()
    print("================================")
    print("ASSET ROUTER TEST COMPLETE")
    print("================================")

    for asset in presentation.assets:
        print(f"ID:     {asset.id}")
        print(f"Type:   {asset.asset_type}")
        print(f"Source: {asset.source}")
        print(f"Path:   {asset.path}")

    # ---------------------------------------------------------
    # VALIDATE
    # ---------------------------------------------------------

    presentation.validate()

    print()
    print("Presentation IR validation: PASS")


if __name__ == "__main__":
    main()