from backend.planner.presentation_planner import PresentationPlanner


def main():

    planner = PresentationPlanner()

    plan = planner.create_plan(
        topic="Quantum Computing",
        slide_count=6
    )

    print("\n===== PRESENTATION PLAN =====\n")

    for key, value in plan.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()