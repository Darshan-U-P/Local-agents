from pathlib import Path

from backend.assets.asset_router import AssetRouter
from backend.generation.manifest import GenerationManifest
from backend.generation.generation_session import GenerationSession


class AssetGenerationManager:
    """
    Controls asset generation for a presentation.

    Model lifecycle:

        Qwen
          ↓
        UNLOAD
          ↓
        AssetGenerationManager
          ↓
        Load FLUX only when needed
          ↓
        Generate image assets
          ↓
        UNLOAD FLUX
          ↓
        Continue with other asset generators

    Image-like assets:
        image
        illustration
        photo

    Current generators:

        image / illustration / photo
            ↓
          FLUX

        icon
            ↓
        future vector generator

        diagram
            ↓
        future diagram generator

        chart
            ↓
        future deterministic chart generator

    Generation is resumable.

    Already completed assets are skipped.
    """

    # ---------------------------------------------------------
    # Asset type groups
    # ---------------------------------------------------------

    IMAGE_ASSET_TYPES = {
        "image",
        "illustration",
        "photo",
    }

    FUTURE_ASSET_TYPES = {
        "icon",
        "diagram",
        "chart",
    }

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        session: GenerationSession,
    ):
        self.session = session

        self.session_dir = Path(
            session.output_dir
        )

        self.assets_dir = (
            self.session_dir / "assets"
        )

        self.assets_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.manifest = GenerationManifest(
            session_dir=str(
                self.session_dir
            )
        )

        self.router = AssetRouter(
            output_dir=str(
                self.assets_dir
            )
        )

    # =========================================================
    # GENERATE ALL ASSETS
    # =========================================================

    def generate_all(self):
        """
        Generate every pending asset in the manifest.

        Image-like assets are processed as one batch:

            Load FLUX
                ↓
            image 1
                ↓
            image 2
                ↓
            illustration 1
                ↓
            photo 1
                ↓
            Unload FLUX

        This avoids repeatedly loading the FLUX model.

        FLUX is also guaranteed to be unloaded if
        generation fails.
        """

        manifest = self.manifest.load_assets()

        assets = manifest.get(
            "assets",
            [],
        )

        if not assets:
            print(
                "No assets found in asset manifest."
            )
            return

        print()
        print("================================")
        print("ASSET GENERATION")
        print("================================")

        print(
            f"Total assets: {len(assets)}"
        )

        self.session.set_status(
            "generating_assets"
        )

        try:

            # =================================================
            # IMAGE-LIKE ASSETS
            # =================================================

            self._generate_image_assets(
                assets
            )

            # =================================================
            # FUTURE ASSET TYPES
            # =================================================

            self._process_non_image_assets(
                assets
            )

            # =================================================
            # FINAL STATUS
            # =================================================

            final_manifest = (
                self.manifest.load_assets()
            )

            final_assets = (
                final_manifest.get(
                    "assets",
                    [],
                )
            )

            if self._all_assets_completed(
                final_assets
            ):

                self.session.set_status(
                    "assets_ready"
                )

                print()
                print("================================")
                print("ASSET GENERATION COMPLETE")
                print("================================")

            else:

                self.session.set_status(
                    "assets_partial"
                )

                print()
                print("================================")
                print(
                    "ASSET GENERATION PARTIALLY COMPLETE"
                )
                print("================================")

                self._print_remaining_assets(
                    final_assets
                )

        except Exception as exc:

            # -------------------------------------------------
            # Safety:
            #
            # FLUX must never remain loaded if something fails.
            # -------------------------------------------------

            self._unload_flux()

            self.session.set_status(
                "failed"
            )

            print()
            print("================================")
            print("ASSET GENERATION FAILED")
            print("================================")

            print(
                f"Error: {exc}"
            )

            raise

    # =========================================================
    # IMAGE GENERATION
    # =========================================================

    def _generate_image_assets(
        self,
        assets: list[dict],
    ):
        """
        Generate all pending image-like assets.

        Image-like types:

            image
            illustration
            photo

        FLUX is loaded once for the complete batch and
        unloaded after the batch finishes.
        """

        image_assets = [
            asset
            for asset in assets
            if asset.get("type", "").strip().lower()
            in self.IMAGE_ASSET_TYPES
            and asset.get("status") != "completed"
        ]

        # -----------------------------------------------------
        # Nothing to generate
        # -----------------------------------------------------

        if not image_assets:

            print()
            print(
                "No pending image assets."
            )

            return

        print()
        print("================================")
        print("IMAGE ASSET GENERATION")
        print("================================")

        print(
            f"Pending image assets: "
            f"{len(image_assets)}"
        )

        # -----------------------------------------------------
        # Load FLUX
        # -----------------------------------------------------

        print()
        print(
            "Loading FLUX for image generation..."
        )

        self.router.image_model_manager.load()

        try:

            # -------------------------------------------------
            # Generate images sequentially
            # -------------------------------------------------

            for asset in image_assets:

                self.generate_asset(
                    asset
                )

        finally:

            # -------------------------------------------------
            # IMPORTANT:
            #
            # Always unload FLUX after the image batch.
            # -------------------------------------------------

            print()
            print(
                "Image generation batch finished."
            )

            self._unload_flux()

            print(
                "FLUX has been unloaded."
            )

    # =========================================================
    # NON-IMAGE ASSETS
    # =========================================================

    def _process_non_image_assets(
        self,
        assets: list[dict],
    ):
        """
        Process asset types that do not currently require FLUX.

        Currently:

            icon
                -> pending

            diagram
                -> pending

            chart
                -> pending

        Their specialized generators will be connected later.

        Image-like assets are skipped because they were already
        processed by _generate_image_assets().
        """

        for asset in assets:

            asset_type = (
                asset.get("type", "")
                .strip()
                .lower()
            )

            # -------------------------------------------------
            # Image-like assets were already processed.
            # -------------------------------------------------

            if asset_type in self.IMAGE_ASSET_TYPES:
                continue

            # -------------------------------------------------
            # Already completed.
            # -------------------------------------------------

            if asset.get(
                "status",
                "pending",
            ) == "completed":

                continue

            # -------------------------------------------------
            # Process icon / diagram / chart.
            # -------------------------------------------------

            self.generate_asset(
                asset
            )

    # =========================================================
    # GENERATE SINGLE ASSET
    # =========================================================

    def generate_asset(
        self,
        asset: dict,
    ):
        """
        Generate one asset.

        If generation fails, the asset is marked as failed
        and the exception is re-raised.

        Image-like assets should normally be generated while
        FLUX is already loaded by _generate_image_assets().
        """

        asset_id = asset["id"]

        asset_type = (
            asset["type"]
            .strip()
            .lower()
        )

        description = asset[
            "description"
        ]

        slide_number = asset[
            "slide_number"
        ]

        visual_spec = asset.get(
            "visual_spec"
        )

        print()
        print("--------------------------------")
        print(
            f"Asset: {asset_id}"
        )
        print(
            f"Slide: {slide_number}"
        )
        print(
            f"Type: {asset_type}"
        )
        print(
            f"Description: {description}"
        )

        if visual_spec:
            print(
                "Visual spec: available"
            )
        else:
            print(
                "Visual spec: not provided"
            )

        print("--------------------------------")

        # -----------------------------------------------------
        # MARK AS GENERATING
        # -----------------------------------------------------

        self.manifest.update_asset(
            asset_id,
            status="generating",
            error=None,
        )

        try:

            # =================================================
            # IMAGE / ILLUSTRATION / PHOTO
            # =================================================

            if asset_type in self.IMAGE_ASSET_TYPES:

                return self._generate_image(
                    asset
                )

            # =================================================
            # ICON
            # =================================================

            if asset_type == "icon":

                return self._generate_icon(
                    asset
                )

            # =================================================
            # DIAGRAM
            # =================================================

            if asset_type == "diagram":

                return self._generate_diagram(
                    asset
                )

            # =================================================
            # CHART
            # =================================================

            if asset_type == "chart":

                return self._generate_chart(
                    asset
                )

            # =================================================
            # UNKNOWN TYPE
            # =================================================

            raise ValueError(
                f"Unsupported asset type: "
                f"{asset_type}"
            )

        except Exception as exc:

            # -------------------------------------------------
            # Record failure
            # -------------------------------------------------

            self.manifest.update_asset(
                asset_id,
                status="failed",
                error=str(exc),
            )

            print()
            print(
                f"[FAILED] {asset_id}"
            )

            print(
                f"Error: {exc}"
            )

            raise

    # =========================================================
    # IMAGE GENERATOR
    # =========================================================

    def _generate_image(
        self,
        asset: dict,
    ):
        """
        Generate one image-like asset using FLUX.

        The structured visual_spec is passed to AssetRouter.

        AssetRouter / AssetPromptBuilder then converts:

            description
            +
            visual_spec

        into the detailed FLUX prompt.
        """

        asset_id = asset["id"]

        asset_type = (
            asset["type"]
            .strip()
            .lower()
        )

        description = asset[
            "description"
        ]

        visual_spec = asset.get(
            "visual_spec"
        )

        # -----------------------------------------------------
        # Make sure FLUX is actually available.
        #
        # This protects against direct calls to generate_asset()
        # outside generate_all().
        # -----------------------------------------------------

        if not self.router.image_model_manager.is_loaded():

            print(
                "FLUX is not loaded. "
                "Loading it now..."
            )

            self.router.image_model_manager.load()

        # -----------------------------------------------------
        # Build the exact prompt that AssetRouter will use.
        #
        # This lets us preserve the actual prompt inside
        # asset_manifest.json for reproducibility/debugging.
        # -----------------------------------------------------

        prompt = self._build_asset_prompt(
            asset_type=asset_type,
            description=description,
            visual_spec=visual_spec,
        )

        # -----------------------------------------------------
        # Generate image
        # -----------------------------------------------------

        output_path = (
            self.router.generate_asset(
                asset_type=asset_type,
                description=description,
                asset_id=asset_id,
                visual_spec=visual_spec,
            )
        )

        if not output_path:

            raise RuntimeError(
                f"FLUX did not return an output path "
                f"for {asset_id}"
            )

        # -----------------------------------------------------
        # Read generation configuration
        # -----------------------------------------------------

        image_config = (
            self.router.config["image"]
        )

        # -----------------------------------------------------
        # Store generation information
        # -----------------------------------------------------

        self.manifest.update_asset(
            asset_id,

            status="completed",

            source="flux",

            model=image_config.get(
                "name",
                "FLUX.1-schnell",
            ),

            runtime=image_config.get(
                "runtime",
                "diffusers",
            ),

            # Store the actual detailed prompt.
            prompt=prompt,

            # Preserve the structured visual specification.
            visual_spec=visual_spec,

            seed=42,

            width=image_config.get(
                "width",
                1024,
            ),

            height=image_config.get(
                "height",
                1024,
            ),

            steps=image_config.get(
                "steps",
                4,
            ),

            path=output_path,

            error=None,
        )

        print()
        print(
            f"[COMPLETED] {asset_id}"
        )

        print(
            f"Output: {output_path}"
        )

        return output_path

    # =========================================================
    # BUILD ASSET PROMPT
    # =========================================================

    def _build_asset_prompt(
        self,
        asset_type: str,
        description: str,
        visual_spec: dict | None,
    ) -> str:
        """
        Build the detailed generator prompt using the same
        AssetPromptBuilder used by AssetRouter.

        This is used only so the exact prompt can be stored
        in asset_manifest.json.

        The actual generation is still performed by AssetRouter.
        """

        # -----------------------------------------------------
        # Use the router's prompt builder when available.
        # -----------------------------------------------------

        prompt_builder = getattr(
            self.router,
            "prompt_builder",
            None,
        )

        if prompt_builder is None:

            # -------------------------------------------------
            # Fallback for compatibility with an older router.
            # -------------------------------------------------

            return description

        # -----------------------------------------------------
        # Build the same structured asset object expected by
        # AssetPromptBuilder.
        # -----------------------------------------------------

        asset_data = {
            "type": asset_type,
            "description": description,
            "visual_spec": visual_spec,
        }

        return prompt_builder.build(
            asset_data
        )

    # =========================================================
    # ICON GENERATOR
    # =========================================================

    def _generate_icon(
        self,
        asset: dict,
    ):
        """
        Icon generation is not implemented yet.

        Future implementation:

            structured icon spec
                    ↓
                SVG/vector
                    ↓
              editable PPT asset
        """

        asset_id = asset["id"]

        self.manifest.update_asset(
            asset_id,

            status="pending",

            source="pending",

            visual_spec=asset.get(
                "visual_spec"
            ),

            error=None,
        )

        print()
        print(
            f"[PENDING] Icon generator "
            f"not implemented yet: {asset_id}"
        )

        return None

    # =========================================================
    # DIAGRAM GENERATOR
    # =========================================================

    def _generate_diagram(
        self,
        asset: dict,
    ):
        """
        Diagram generation is not implemented yet.

        Future implementation:

            structured diagram spec
                    ↓
                SVG/vector
                    ↓
              editable PPT elements
        """

        asset_id = asset["id"]

        self.manifest.update_asset(
            asset_id,

            status="pending",

            source="pending",

            visual_spec=asset.get(
                "visual_spec"
            ),

            error=None,
        )

        print()
        print(
            f"[PENDING] Diagram generator "
            f"not implemented yet: {asset_id}"
        )

        return None

    # =========================================================
    # CHART GENERATOR
    # =========================================================

    def _generate_chart(
        self,
        asset: dict,
    ):
        """
        Chart generation is not implemented yet.

        Future implementation:

            structured chart specification
                    ↓
              deterministic chart
                    ↓
              editable PPT chart
        """

        asset_id = asset["id"]

        self.manifest.update_asset(
            asset_id,

            status="pending",

            source="pending",

            visual_spec=asset.get(
                "visual_spec"
            ),

            error=None,
        )

        print()
        print(
            f"[PENDING] Chart generator "
            f"not implemented yet: {asset_id}"
        )

        return None

    # =========================================================
    # UNLOAD FLUX
    # =========================================================

    def _unload_flux(self):
        """
        Safely unload FLUX.

        This method is intentionally centralized so every
        failure path uses the same model cleanup operation.
        """

        try:

            if (
                self.router.image_model_manager.is_loaded()
            ):

                print()
                print(
                    "Releasing FLUX model..."
                )

                self.router.image_model_manager.unload()

            else:

                print(
                    "FLUX is already unloaded."
                )

        except Exception as exc:

            print()
            print(
                "Warning: FLUX unload failed:"
            )

            print(
                f"  {exc}"
            )

    # =========================================================
    # CHECK FINAL STATUS
    # =========================================================

    @staticmethod
    def _all_assets_completed(
        assets: list[dict],
    ) -> bool:
        """
        Return True only when every asset has been generated.
        """

        if not assets:
            return False

        return all(
            asset.get("status")
            == "completed"
            for asset in assets
        )

    # =========================================================
    # PRINT REMAINING ASSETS
    # =========================================================

    @staticmethod
    def _print_remaining_assets(
        assets: list[dict],
    ):
        """
        Print assets that still need processing.
        """

        print()
        print("Remaining assets:")

        for asset in assets:

            if asset.get(
                "status"
            ) != "completed":

                print(
                    f"  - {asset['id']} | "
                    f"{asset['type']} | "
                    f"{asset.get('status', 'pending')}"
                )

    # =========================================================
    # RESUME
    # =========================================================

    def resume(self):
        """
        Resume asset generation.

        Only assets that are not completed will be processed.
        """

        print()
        print("================================")
        print("RESUMING ASSET GENERATION")
        print("================================")

        self.generate_all()