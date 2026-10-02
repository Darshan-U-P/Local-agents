from pathlib import Path

from backend.generation.presentation_generator import (
    PresentationGenerator,
)


def main():
    # ---------------------------------------------------------
    # CREATE GENERATOR
    # ---------------------------------------------------------

    generator = PresentationGenerator()

    # ---------------------------------------------------------
    # CREATE PRESENTATION
    # ---------------------------------------------------------

    session = generator.create(
        topic="Quantum Computing",
        slide_count=6,
    )

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------

    session_dir = Path(
        session.output_dir
    )

    print()
    print("================================")
    print("PRESENTATION GENERATOR TEST")
    print("================================")

    print(f"Session ID: {session.session_id}")
    print(f"Topic:      {session.topic}")
    print(f"Status:     {session.status}")
    print(f"Directory:  {session_dir}")

    print()
    print("Files:")

    expected_files = [
        "session.json",
        "presentation_plan.json",
        "asset_manifest.json",
    ]

    all_exist = True

    for filename in expected_files:
        path = session_dir / filename

        exists = path.exists()

        print(
            f"  {'✓' if exists else '✗'} {filename}"
        )

        if not exists:
            all_exist = False

    # ---------------------------------------------------------
    # SLIDE STATUS
    # ---------------------------------------------------------

    print()
    print("Slides:")

    for slide in session.slides:
        print(
            f"  Slide {slide.slide_number}: "
            f"{slide.status}"
        )

    # ---------------------------------------------------------
    # FINAL VALIDATION
    # ---------------------------------------------------------

    if not all_exist:
        raise RuntimeError(
            "One or more generation files were not created."
        )

    print()
    print("================================")
    print("TEST PASSED")
    print("================================")


if __name__ == "__main__":
    main()