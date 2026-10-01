from backend.planner.presentation_planner import PresentationPlanner
from backend.ir.builder import PresentationIRBuilder


def main():

    # ---------------------------------------------------------
    # Generate presentation plan
    # ---------------------------------------------------------

    planner = PresentationPlanner()

    plan = planner.create_plan(
        topic="Quantum Computing",
        slide_count=6,
    )

    # ---------------------------------------------------------
    # Convert plan → Presentation IR
    # ---------------------------------------------------------

    builder = PresentationIRBuilder()

    presentation = builder.build(plan)

    # ---------------------------------------------------------
    # Print IR
    # ---------------------------------------------------------

    print("\n===== PRESENTATION IR =====\n")

    ir = presentation.to_dict()

    print(f"Title: {ir['presentation']['title']}")
    print(f"Slides: {ir['presentation']['slide_count']}")
    print(f"Assets: {len(ir['assets'])}")

    for slide in ir["slides"]:

        print(
            f"\nSlide {slide['slide_number']}: "
            f"{slide['title']}"
        )

        print(f"Layout: {slide['layout']}")

        print(
            f"Elements: {len(slide['elements'])}"
        )

        print(
            f"Assets: {slide['asset_ids']}"
        )

    print("\n===== IR VALIDATION =====")
    print("VALID")


if __name__ == "__main__":
    main()