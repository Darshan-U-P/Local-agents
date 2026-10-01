from backend.planner.presentation_planner import (
    PresentationPlanner,
)

from backend.ir.builder import (
    PresentationIRBuilder,
)

from backend.assets.asset_router import (
    AssetRouter,
)


def main():
    print()
    print("===== ASSET ROUTER TEST =====")
    print()

    # -----------------------------------------------------
    # 1. Planner
    # -----------------------------------------------------

    planner = PresentationPlanner()

    plan = planner.create_plan(
        topic="Quantum Computing",
        slide_count=6,
    )

    print("Planner: OK")

    # -----------------------------------------------------
    # 2. Presentation IR
    # -----------------------------------------------------

    builder = PresentationIRBuilder()

    presentation = builder.build(plan)

    print("Presentation IR: OK")
    print(
        f"Assets before routing: "
        f"{len(presentation.assets)}"
    )

    # -----------------------------------------------------
    # 3. Asset Router
    # -----------------------------------------------------

    router = AssetRouter()

    router.route(presentation)

    print("Asset Router: OK")

    # -----------------------------------------------------
    # 4. Inspect assets
    # -----------------------------------------------------

    print()
    print("===== ROUTED ASSETS =====")

    for asset in presentation.assets:
        print(
            f"{asset.id}: "
            f"type={asset.asset_type}, "
            f"source={asset.source}, "
            f"path={asset.path}"
        )

    # -----------------------------------------------------
    # 5. Validation
    # -----------------------------------------------------

    presentation.validate()

    print()
    print("===== ASSET VALIDATION =====")
    print("ALL ASSETS ROUTED SUCCESSFULLY")
    print()


if __name__ == "__main__":
    main()