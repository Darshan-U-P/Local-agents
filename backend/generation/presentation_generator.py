from pathlib import Path
from uuid import uuid4

from backend.planner.presentation_planner import PresentationPlanner
from backend.generation.generation_session import GenerationSession
from backend.generation.manifest import GenerationManifest


class PresentationGenerator:
    """
    Controls the presentation-generation workflow.

    Current workflow:

        User topic
            ↓
        Presentation Planner
            ↓
        presentation_plan.json
            ↓
        Asset extraction
            ↓
        asset_manifest.json

    Model lifecycle:

        Load Qwen
            ↓
        Generate presentation plan
            ↓
        UNLOAD Qwen
            ↓
        Continue with non-LLM processing

    Asset generation and slide rendering will be
    connected in later stages.

    Important:

    The planner now generates detailed visual specifications
    for assets.

    Those specifications are preserved in asset_manifest.json
    so downstream generators can use them.

    Example:

        visual_spec:
            purpose
            subject
            composition
            style
            color_palette
            must_show
            must_avoid
            text_policy
    """

    def __init__(
        self,
        output_root: str = "generated/presentations",
    ):
        self.output_root = Path(output_root)

        self.output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.planner = PresentationPlanner()

    # =========================================================
    # CREATE PRESENTATION
    # =========================================================

    def create(
        self,
        topic: str,
        slide_count: int = 6,
    ) -> GenerationSession:

        if not topic or not topic.strip():
            raise ValueError(
                "Presentation topic cannot be empty."
            )

        # -----------------------------------------------------
        # CREATE SESSION
        # -----------------------------------------------------

        session_id = (
            f"{self._safe_name(topic)}-"
            f"{uuid4().hex[:8]}"
        )

        session_dir = (
            self.output_root / session_id
        )

        session = GenerationSession(
            session_id=session_id,
            topic=topic,
            output_dir=str(session_dir),
        )

        session.set_status(
            "planning"
        )

        # -----------------------------------------------------
        # PRESENTATION PLANNING
        # -----------------------------------------------------

        print()
        print("================================")
        print("PRESENTATION PLANNING")
        print("================================")
        print(f"Topic: {topic}")
        print(f"Slides: {slide_count}")
        print()

        # -----------------------------------------------------
        # QWEN MODEL LIFECYCLE
        #
        # Qwen is loaded automatically by ModelManager when
        # PresentationPlanner calls create_plan().
        #
        # The finally block guarantees that Qwen is unloaded
        # regardless of whether planning succeeds or fails.
        # -----------------------------------------------------

        try:

            plan = self.planner.create_plan(
                topic=topic,
                slide_count=slide_count,
            )

        finally:

            print()
            print("Presentation planning finished.")
            print("Releasing Qwen model...")

            self.planner.unload()

        # -----------------------------------------------------
        # SAVE PRESENTATION PLAN
        # -----------------------------------------------------

        manifest = GenerationManifest(
            session_dir=str(session_dir)
        )

        manifest.save_presentation_plan(
            plan
        )

        # -----------------------------------------------------
        # INITIALIZE SLIDE STATES
        # -----------------------------------------------------

        actual_slide_count = len(
            plan.get(
                "slides",
                [],
            )
        )

        if actual_slide_count == 0:
            raise ValueError(
                "Presentation planner returned no slides."
            )

        session.initialize_slides(
            actual_slide_count
        )

        # -----------------------------------------------------
        # BUILD ASSET MANIFEST
        # -----------------------------------------------------

        assets = self._extract_assets(
            plan
        )

        manifest.initialize_assets(
            assets
        )

        # -----------------------------------------------------
        # UPDATE SESSION
        # -----------------------------------------------------

        session.set_status(
            "planned"
        )

        # -----------------------------------------------------
        # REPORT
        # -----------------------------------------------------

        print()
        print("================================")
        print("PRESENTATION PLAN CREATED")
        print("================================")

        print(
            f"Session: {session.session_id}"
        )

        print(
            f"Slides: {actual_slide_count}"
        )

        print(
            f"Assets: {len(assets)}"
        )

        print(
            f"Directory: {session_dir}"
        )

        print()
        print("Qwen model has been unloaded.")
        print(
            "GPU memory is now available for asset generation."
        )

        return session

    # =========================================================
    # ASSET EXTRACTION
    # =========================================================

    @staticmethod
    def _extract_assets(
        plan: dict,
    ) -> list[dict]:
        """
        Extract asset requirements from the presentation plan.

        Qwen provides:

            type
            description
            visual_spec

        The backend assigns the asset ID.

        Example planner output:

            {
                "type": "image",
                "description": "Quantum computer",
                "visual_spec": {
                    "purpose": "...",
                    "subject": "...",
                    "composition": "...",
                    "style": "...",
                    "color_palette": [...],
                    "must_show": [...],
                    "must_avoid": [...],
                    "text_policy": "..."
                }
            }

        Backend manifest entry:

            {
                "id": "asset-001",
                "type": "image",
                "description": "Quantum computer",
                "visual_spec": {...}
            }

        The visual specification is preserved exactly so that
        downstream asset generators can use it to create
        better prompts and more accurate visual assets.

        Asset identity remains controlled by the backend.
        """

        assets = []

        asset_counter = 1

        # -----------------------------------------------------
        # LOOP THROUGH SLIDES
        # -----------------------------------------------------

        for slide in plan.get(
            "slides",
            [],
        ):

            slide_number = slide.get(
                "slide_number"
            )

            # -------------------------------------------------
            # LOOP THROUGH ASSETS
            # -------------------------------------------------

            for asset in slide.get(
                "assets",
                [],
            ):

                asset_type = (
                    asset.get(
                        "type",
                        "image",
                    )
                    .strip()
                    .lower()
                )

                description = asset.get(
                    "description",
                    "",
                ).strip()

                # ---------------------------------------------
                # VALIDATE ASSET DESCRIPTION
                # ---------------------------------------------

                if not description:
                    continue

                # ---------------------------------------------
                # EXTRACT VISUAL SPECIFICATION
                # ---------------------------------------------

                visual_spec = asset.get(
                    "visual_spec"
                )

                # ---------------------------------------------
                # NONE ASSET
                #
                # A "none" asset does not need a visual spec.
                # ---------------------------------------------

                if asset_type == "none":
                    visual_spec = None

                # ---------------------------------------------
                # GENERATE BACKEND ASSET ID
                # ---------------------------------------------

                asset_id = (
                    f"asset-{asset_counter:03d}"
                )

                # ---------------------------------------------
                # CREATE MANIFEST ENTRY
                # ---------------------------------------------

                assets.append(
                    {
                        # Backend identity
                        "id": asset_id,

                        # Asset classification
                        "type": asset_type,

                        # Slide relationship
                        "slide_number": slide_number,

                        # Human-readable description
                        "description": description,

                        # Structured visual requirements
                        "visual_spec": visual_spec,

                        # Final generation prompt
                        #
                        # This will be created later by the
                        # Asset Prompt Builder.
                        "prompt": None,

                        # Generation source
                        #
                        # Example:
                        # flux
                        # svg
                        # native_chart
                        # ppt
                        "source": None,

                        # Model information
                        "model": None,
                        "runtime": None,

                        # Generation parameters
                        "seed": None,
                        "width": None,
                        "height": None,
                        "steps": None,

                        # Generation state
                        "status": "pending",

                        # Generated file
                        "path": None,

                        # Error information
                        "error": None,
                    }
                )

                asset_counter += 1

        return assets

    # =========================================================
    # SAFE SESSION NAME
    # =========================================================

    @staticmethod
    def _safe_name(
        topic: str,
    ) -> str:

        allowed = (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789-_"
        )

        name = "".join(
            char
            if char in allowed
            else "-"
            for char in topic.strip()
        )

        while "--" in name:
            name = name.replace(
                "--",
                "-",
            )

        name = name.strip("-")

        if not name:
            name = "presentation"

        return name.lower()[:80]