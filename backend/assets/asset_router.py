from pathlib import Path
import json

from backend.ir.presentation_ir import (
    PresentationIR,
    AssetIR,
)

from backend.assets.image_model_manager import (
    ImageModelManager,
)

from backend.assets.generators.flux_generator import (
    FluxGenerator,
)

from backend.assets.generators.placeholder_generator import (
    PlaceholderGenerator,
)

from backend.assets.asset_prompt_builder import (
    AssetPromptBuilder,
)


class AssetRouter:
    """
    Determines how each presentation asset should be generated.

    Routing:

        image       -> FLUX image generator
        illustration -> FLUX image generator
        photo       -> FLUX image generator
        icon        -> icon generator
        diagram     -> diagram generator
        chart       -> chart generator

    The router is responsible for selecting the appropriate
    generator and preparing the required input.

    The actual generation logic remains inside the individual
    generators.

    Image pipeline:

        Asset description
                +
        visual_spec
                ↓
        AssetPromptBuilder
                ↓
        Detailed FLUX prompt
                ↓
        FluxGenerator
                ↓
        Generated image
    """

    SUPPORTED_TYPES = {
        "image",
        "icon",
        "diagram",
        "chart",
        "illustration",
        "photo",
    }

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        output_dir: str = "generated/assets",
        config_path: str = "config/models.json",
    ):
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.config_path = Path(
            config_path
        )

        self.config = self._load_config()

        # -----------------------------------------------------
        # PLACEHOLDER GENERATOR
        # -----------------------------------------------------

        self.placeholder_generator = (
            PlaceholderGenerator()
        )

        # -----------------------------------------------------
        # ASSET PROMPT BUILDER
        # -----------------------------------------------------

        self.prompt_builder = (
            AssetPromptBuilder()
        )

        # -----------------------------------------------------
        # IMAGE MODEL
        # -----------------------------------------------------

        image_config = self.config["image"]

        self.image_model_manager = (
            ImageModelManager(
                model_path=image_config["path"],
                offload_path=str(
                    self.output_dir
                    / "flux_offload_cache"
                ),
            )
        )

        # -----------------------------------------------------
        # FLUX GENERATOR
        # -----------------------------------------------------

        self.image_generator = (
            FluxGenerator(
                model_manager=self.image_model_manager,
                output_dir=str(
                    self.output_dir
                ),
            )
        )

    # =========================================================
    # CONFIG
    # =========================================================

    def _load_config(self):
        """
        Load model configuration from config/models.json.
        """

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Model configuration not found: "
                f"{self.config_path}"
            )

        with self.config_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    # =========================================================
    # PUBLIC API
    # =========================================================

    def route(
        self,
        presentation: PresentationIR,
    ) -> PresentationIR:
        """
        Route all assets in a PresentationIR.

        The PresentationIR is updated with the resolved
        asset source/path.
        """

        presentation.validate()

        for asset in presentation.assets:
            self._route_asset(asset)

        presentation.validate()

        return presentation

    # =========================================================
    # GENERATE SINGLE ASSET
    # =========================================================

    def generate_asset(
        self,
        asset_type: str,
        description: str,
        asset_id: str,
        visual_spec: dict | None = None,
    ) -> str | None:
        """
        Generate a single asset.

        Used by AssetGenerationManager.

        Parameters
        ----------
        asset_type:
            Type of asset.

        description:
            Human-readable asset description.

        asset_id:
            Backend-controlled asset identifier.

        visual_spec:
            Structured visual specification produced by
            PresentationPlanner.

        Example
        -------
        router.generate_asset(
            asset_type="image",
            description="Quantum computer",
            asset_id="asset-001",
            visual_spec={
                "purpose": "...",
                "subject": "...",
                "composition": "...",
                "style": "...",
                "color_palette": ["blue", "cyan"],
                "must_show": ["quantum processor"],
                "must_avoid": ["text"],
                "text_policy": "no text",
            },
        )
        """

        asset_type = (
            asset_type
            .strip()
            .lower()
        )

        if asset_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported asset type: "
                f"{asset_type}"
            )

        if not description.strip():
            raise ValueError(
                "Asset description cannot be empty."
            )

        # -----------------------------------------------------
        # IMAGE
        # -----------------------------------------------------

        if asset_type in {
            "image",
            "illustration",
            "photo",
        }:
            return self._route_image_generation(
                description=description,
                asset_id=asset_id,
                visual_spec=visual_spec,
                asset_type=asset_type,
            )

        # -----------------------------------------------------
        # ICON
        # -----------------------------------------------------

        if asset_type == "icon":
            return self._generate_icon(
                description=description,
                asset_id=asset_id,
                visual_spec=visual_spec,
            )

        # -----------------------------------------------------
        # DIAGRAM
        # -----------------------------------------------------

        if asset_type == "diagram":
            return self._generate_diagram(
                description=description,
                asset_id=asset_id,
                visual_spec=visual_spec,
            )

        # -----------------------------------------------------
        # CHART
        # -----------------------------------------------------

        if asset_type == "chart":
            return self._generate_chart(
                description=description,
                asset_id=asset_id,
                visual_spec=visual_spec,
            )

        return None

    # =========================================================
    # ASSET ROUTING — PRESENTATION IR
    # =========================================================

    def _route_asset(
        self,
        asset: AssetIR,
    ):
        """
        Route an AssetIR object to its generator.
        """

        asset_type = (
            asset.asset_type
            .strip()
            .lower()
        )

        if asset_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported asset type: "
                f"{asset.asset_type}"
            )

        if asset_type in {
            "image",
            "illustration",
            "photo",
        }:
            self._route_image(asset)

        elif asset_type == "icon":
            self._route_icon(asset)

        elif asset_type == "diagram":
            self._route_diagram(asset)

        elif asset_type == "chart":
            self._route_chart(asset)

    # =========================================================
    # BUILD IMAGE PROMPT
    # =========================================================

    def _build_image_prompt(
        self,
        description: str,
        visual_spec: dict | None,
        asset_type: str,
    ) -> str:
        """
        Convert the structured asset specification into the
        final generator prompt.

        If visual_spec is available, AssetPromptBuilder is used.

        A fallback asset structure is created when older
        PresentationIR objects do not contain visual_spec.
        """

        if visual_spec is None:
            visual_spec = {
                "purpose": (
                    "Create a presentation visual "
                    "supporting the slide content."
                ),
                "subject": description,
                "composition": (
                    "Create a clean, balanced composition "
                    "with the main subject clearly visible."
                ),
                "style": (
                    "Professional presentation "
                    "visualization."
                ),
                "color_palette": [
                    "blue",
                    "cyan",
                    "white",
                ],
                "must_show": [
                    description,
                ],
                "must_avoid": [
                    "text",
                    "letters",
                    "numbers",
                    "logos",
                    "watermarks",
                ],
                "text_policy": "no text",
            }

        asset = {
            "type": asset_type,
            "description": description,
            "visual_spec": visual_spec,
        }

        return self.prompt_builder.build(
            asset
        )

    # =========================================================
    # IMAGE — PRESENTATION IR
    # =========================================================

    def _route_image(
        self,
        asset: AssetIR,
    ):
        """
        Generate an image for an AssetIR object
        using local FLUX.
        """

        output_name = (
            f"{asset.id}.png"
        )

        prompt = self._build_image_prompt(
            description=asset.description,
            visual_spec=getattr(
                asset,
                "visual_spec",
                None,
            ),
            asset_type="image",
        )

        output_path = (
            self.image_generator.generate(
                prompt=prompt,
                output_name=output_name,
                width=self.config["image"].get(
                    "width",
                    1024,
                ),
                height=self.config["image"].get(
                    "height",
                    1024,
                ),
                steps=self.config["image"].get(
                    "steps",
                    4,
                ),
                seed=42,
            )
        )

        asset.source = "flux"
        asset.path = output_path

    # =========================================================
    # IMAGE — GENERATION MANAGER
    # =========================================================

    def _route_image_generation(
        self,
        description: str,
        asset_id: str,
        visual_spec: dict | None = None,
        asset_type: str = "image",
    ) -> str:
        """
        Generate a single image using FLUX.

        This method is used by AssetGenerationManager.

        The important part is that the short asset description
        is no longer sent directly to FLUX.

        Instead:

            description
                +
            visual_spec
                ↓
            AssetPromptBuilder
                ↓
            detailed FLUX prompt
        """

        output_name = (
            f"{asset_id}.png"
        )

        # -----------------------------------------------------
        # BUILD DETAILED PROMPT
        # -----------------------------------------------------

        prompt = self._build_image_prompt(
            description=description,
            visual_spec=visual_spec,
            asset_type=asset_type,
        )

        print()
        print("================================")
        print("FLUX PROMPT")
        print("================================")
        print(prompt)
        print("================================")
        print()

        # -----------------------------------------------------
        # GENERATE IMAGE
        # -----------------------------------------------------

        output_path = (
            self.image_generator.generate(
                prompt=prompt,
                output_name=output_name,
                width=self.config["image"].get(
                    "width",
                    1024,
                ),
                height=self.config["image"].get(
                    "height",
                    1024,
                ),
                steps=self.config["image"].get(
                    "steps",
                    4,
                ),
                seed=42,
            )
        )

        return output_path

    # =========================================================
    # ICON
    # =========================================================

    def _route_icon(
        self,
        asset: AssetIR,
    ):
        """
        Icon routing for PresentationIR.

        Real vector icon generation will be implemented
        in a later stage.
        """

        asset.source = "pending"
        asset.path = None

    def _generate_icon(
        self,
        description: str,
        asset_id: str,
        visual_spec: dict | None = None,
    ) -> str | None:
        """
        Placeholder for the future icon generator.
        """

        return None

    # =========================================================
    # DIAGRAM
    # =========================================================

    def _route_diagram(
        self,
        asset: AssetIR,
    ):
        """
        Diagram routing for PresentationIR.

        A deterministic SVG/vector diagram renderer will be
        implemented later.
        """

        asset.source = "pending"
        asset.path = None

    def _generate_diagram(
        self,
        description: str,
        asset_id: str,
        visual_spec: dict | None = None,
    ) -> str | None:
        """
        Placeholder for the future diagram generator.
        """

        return None

    # =========================================================
    # CHART
    # =========================================================

    def _route_chart(
        self,
        asset: AssetIR,
    ):
        """
        Chart routing for PresentationIR.

        A deterministic chart generator will be implemented
        later.
        """

        asset.source = "pending"
        asset.path = None

    def _generate_chart(
        self,
        description: str,
        asset_id: str,
        visual_spec: dict | None = None,
    ) -> str | None:
        """
        Placeholder for the future chart generator.
        """

        return None