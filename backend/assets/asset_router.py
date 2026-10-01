from pathlib import Path

from backend.ir.presentation_ir import (
    PresentationIR,
    AssetIR,
)

from backend.assets.generators.placeholder_generator import (
    PlaceholderGenerator,
)


class AssetRouter:
    """
    Determines how each presentation asset should be generated.

    The router does NOT generate the actual asset itself.

    It decides:

        image  -> image generator
        icon   -> icon generator
        diagram -> diagram generator
        chart  -> chart generator

    This keeps the architecture modular.
    """

    SUPPORTED_TYPES = {
        "image",
        "icon",
        "diagram",
        "chart",
    }

    def __init__(
        self,
        output_dir: str = "generated/assets",
    ):
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Temporary generator.
        #
        # Later this can become:
        #
        # self.image_generator = ...
        # self.icon_generator = ...
        # self.diagram_generator = ...
        # self.chart_generator = ...

        self.placeholder_generator = (
            PlaceholderGenerator()
        )

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------

    def route(
        self,
        presentation: PresentationIR,
    ) -> PresentationIR:
        """
        Route all assets in a presentation.

        The PresentationIR itself is updated with
        the resolved asset source/path information.
        """

        presentation.validate()

        for asset in presentation.assets:
            self._route_asset(asset)

        presentation.validate()

        return presentation

    # ---------------------------------------------------------
    # ASSET ROUTING
    # ---------------------------------------------------------

    def _route_asset(
        self,
        asset: AssetIR,
    ):
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

        if asset_type == "image":
            self._route_image(asset)

        elif asset_type == "icon":
            self._route_icon(asset)

        elif asset_type == "diagram":
            self._route_diagram(asset)

        elif asset_type == "chart":
            self._route_chart(asset)

    # ---------------------------------------------------------
    # IMAGE
    # ---------------------------------------------------------

    def _route_image(
        self,
        asset: AssetIR,
    ):
        """
        Route image generation.

        Phase 6 currently uses a placeholder
        generator.

        Phase 7 will connect the actual
        local image model.
        """

        output_path = (
            self.output_dir
            / f"{asset.id}.png"
        )

        self.placeholder_generator.generate(
            description=asset.description,
            output_path=str(output_path),
        )

        asset.source = "placeholder"
        asset.path = None

    # ---------------------------------------------------------
    # ICON
    # ---------------------------------------------------------

    def _route_icon(
        self,
        asset: AssetIR,
    ):
        """
        Icon routing.

        Real icon generation will be added later.
        """

        asset.source = "pending"
        asset.path = None

    # ---------------------------------------------------------
    # DIAGRAM
    # ---------------------------------------------------------

    def _route_diagram(
        self,
        asset: AssetIR,
    ):
        """
        Diagram routing.

        Later this will connect to a deterministic
        diagram renderer.
        """

        asset.source = "pending"
        asset.path = None

    # ---------------------------------------------------------
    # CHART
    # ---------------------------------------------------------

    def _route_chart(
        self,
        asset: AssetIR,
    ):
        """
        Chart routing.

        Later this will connect to a deterministic
        chart generator.
        """

        asset.source = "pending"
        asset.path = None