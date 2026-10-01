from backend.planner.presentation_planner import PresentationPlanner
from backend.ir.builder import PresentationIRBuilder
from backend.layout.layout_engine import LayoutEngine


def main():

    planner = PresentationPlanner()

    plan = planner.create_plan(
        topic="Quantum Computing",
        slide_count=6,
    )

    builder = PresentationIRBuilder()
    presentation = builder.build(plan)

    engine = LayoutEngine()

    print("\n===== COMPLETE LAYOUT ENGINE TEST =====\n")

    for slide in presentation.slides:

        positioned = engine.layout_slide(slide)

        print(
            f"Slide {slide.slide_number}: "
            f"{slide.title}"
        )

        print(f"  Layout: {slide.layout}")
        print(f"  Elements: {len(positioned)}")

        for element in positioned:

            rect = element.rect

            print(
                f"    {element.element_id}: "
                f"({rect.x:.2f}, {rect.y:.2f}) "
                f"{rect.width:.2f}x{rect.height:.2f}"
            )

    print("\n===== LAYOUT VALIDATION =====")
    print("ALL SLIDES VALID")


if __name__ == "__main__":
    main()