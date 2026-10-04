from pathlib import Path
import os
import subprocess
import time

from backend.assets.asset_router import AssetRouter
from backend.generation.manifest import GenerationManifest
from backend.generation.generation_session import GenerationSession


class Flux2KleinManager:
    """
    Local FLUX.2 Klein 4B image generator.

    Uses the already validated stable-diffusion.cpp CUDA executable
    instead of loading FLUX through Diffusers/Python.

    Pipeline:

        Python
           â†“
        sd-cli.exe
           â†“
        FLUX.2 Klein 4B Q4_0
           +
        Qwen3-4B
           +
        FLUX.2 VAE
           â†“
        PNG
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)

        # ---------------------------------------------------------
        # Project paths
        # ---------------------------------------------------------

        self.project_root = Path(
            r"D:\all project\Projects\AI projects\ai assentent\Local-agents"
        )

        self.sd_cli = (
            self.project_root
            / "sd-master-3f8527a-bin-win-cuda12-x64"
            / "sd-cli.exe"
        )

        self.image_model_dir = Path(
            r"D:\all project\Projects\AI projects\ai assentent\Local_codex\models\image"
        )

        self.chat_model_dir = Path(
            r"D:\all project\Projects\AI projects\ai assentent\Local_codex\models\chat\qwen3-4b"
        )

        # ---------------------------------------------------------
        # Model paths
        # ---------------------------------------------------------

        self.diffusion_model = (
            self.image_model_dir
            / "flux-2-klein-4b-Q4_0.gguf"
        )

        self.qwen_model = (
            self.chat_model_dir
            / "Qwen3-4B-Q4_K_M.gguf"
        )

        self.vae_model = (
            self.image_model_dir
            / "full_encoder_small_decoder.safetensors"
        )

        # ---------------------------------------------------------
        # Generation configuration
        # ---------------------------------------------------------

        self.width = int(os.getenv("FLUX2_WIDTH", "512"))
        self.height = int(os.getenv("FLUX2_HEIGHT", "512"))

        self.steps = int(os.getenv("FLUX2_STEPS", "4"))
        self.cfg_scale = float(os.getenv("FLUX2_CFG", "1.0"))
        self.seed = int(os.getenv("FLUX2_SEED", "42"))

        self.sampling_method = "euler"

        self.loaded = False

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # =============================================================
    # VALIDATION
    # =============================================================

    def _validate_files(self):
        """Validate all required FLUX.2 runtime files."""

        required = {
            "sd-cli.exe": self.sd_cli,
            "FLUX.2 Klein Q4_0": self.diffusion_model,
            "Qwen3-4B": self.qwen_model,
            "FLUX.2 VAE": self.vae_model,
        }

        missing = []

        for name, path in required.items():
            if not path.exists():
                missing.append(
                    f"{name}: {path}"
                )

        if missing:
            raise FileNotFoundError(
                "Missing FLUX.2 runtime files:\n"
                + "\n".join(missing)
            )

    # =============================================================
    # MODEL LIFECYCLE
    # =============================================================

    def load(self):
        """
        Validate the FLUX.2 runtime.

        Unlike Diffusers, stable-diffusion.cpp does not require us
        to explicitly load the model into Python memory.
        """

        if self.loaded:
            return

        print()
        print("================================")
        print("FLUX.2 KLEIN RUNTIME")
        print("================================")

        self._validate_files()

        print()
        print("sd-cli:")
        print(f"  {self.sd_cli}")

        print()
        print("FLUX.2:")
        print(f"  {self.diffusion_model}")

        print()
        print("Qwen3-4B:")
        print(f"  {self.qwen_model}")

        print()
        print("VAE:")
        print(f"  {self.vae_model}")

        self.loaded = True

        print()
        print("FLUX.2 Klein runtime ready.")

    def unload(self):
        """
        Release the logical runtime state.

        sd-cli is launched per generation, so there is no persistent
        Python-side model object to delete.
        """

        self.loaded = False

        print(
            "FLUX.2 Klein runtime released."
        )

    def is_loaded(self):
        return self.loaded

    # =============================================================
    # GENERATION
    # =============================================================

    def generate(
        self,
        prompt: str,
        output_path: Path,
        seed: int | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
    ):
        """
        Generate one image using FLUX.2 Klein through sd-cli.
        """

        if not self.loaded:
            self.load()

        output_path = Path(output_path).resolve()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        actual_seed = (
            self.seed
            if seed is None
            else seed
        )

        actual_width = (
            self.width
            if width is None
            else width
        )

        actual_height = (
            self.height
            if height is None
            else height
        )

        actual_steps = (
            self.steps
            if steps is None
            else steps
        )

        command = [
            str(self.sd_cli),

            "--diffusion-model",
            str(self.diffusion_model),

            "--vae",
            str(self.vae_model),

            "--llm",
            str(self.qwen_model),

            "--backend",
            "cuda",

            "--offload-to-cpu",

            "--diffusion-fa",

            "--cfg-scale",
            str(self.cfg_scale),

            "--sampling-method",
            self.sampling_method,

            "--steps",
            str(actual_steps),

            "-W",
            str(actual_width),

            "-H",
            str(actual_height),

            "--seed",
            str(actual_seed),

            "-p",
            prompt,

            "-o",
            str(output_path),

            "-v",
        ]

        print()
        print("--------------------------------")
        print("FLUX.2 GENERATION")
        print("--------------------------------")

        print(
            f"Resolution: "
            f"{actual_width}x{actual_height}"
        )

        print(
            f"Steps: {actual_steps}"
        )

        print(
            f"Seed: {actual_seed}"
        )

        print(
            f"Output: {output_path}"
        )

        print()
        print("Prompt:")
        print(prompt)

        print()
        print("Starting sd-cli...")
        print("--------------------------------")

        start_time = time.perf_counter()

        # ---------------------------------------------------------
        # Start stable-diffusion.cpp
        # ---------------------------------------------------------

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(self.sd_cli.parent),
        )

        # ---------------------------------------------------------
        # Stream output to terminal
        # ---------------------------------------------------------

        if process.stdout is not None:

            for line in process.stdout:
                print(
                    line.rstrip()
                )

        return_code = process.wait()

        elapsed = (
            time.perf_counter()
            - start_time
        )

        print()
        print(
            f"FLUX.2 process finished "
            f"in {elapsed:.2f}s"
        )

        # ---------------------------------------------------------
        # Validate process result
        # ---------------------------------------------------------

        if return_code != 0:

            raise RuntimeError(
                "FLUX.2 sd-cli failed "
                f"with exit code {return_code}"
            )

        # ---------------------------------------------------------
        # Validate output
        # ---------------------------------------------------------

        if not output_path.exists():

            raise RuntimeError(
                "FLUX.2 completed without "
                f"creating the expected output:\n"
                f"{output_path}"
            )

        if output_path.stat().st_size == 0:

            raise RuntimeError(
                "FLUX.2 created an empty output file:\n"
                f"{output_path}"
            )

        print()
        print(
            f"FLUX.2 image generated successfully:"
        )
        print(output_path)

        return str(output_path)


class AssetGenerationManager:
    """
    Controls asset generation for a presentation.

    Current image pipeline:

        AssetGenerationManager
                â†“
        AssetPromptBuilder
                â†“
        Flux2KleinManager
                â†“
        stable-diffusion.cpp
                â†“
        FLUX.2 Klein 4B
                â†“
        PNG

    Other asset types remain separate:

        icon
            â†“
        future vector generator

        diagram
            â†“
        future diagram generator

        chart
            â†“
        future deterministic chart generator

        none
            â†“
        completed immediately
    """

    # =========================================================
    # ASSET TYPE GROUPS
    # =========================================================

    IMAGE_ASSET_TYPES = {
        "image",
        "illustration",
        "photo",
    }

    NONE_ASSET_TYPES = {
        "none",
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
            self.session_dir
            / "assets"
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

        # Keep AssetRouter because it owns the
        # AssetPromptBuilder used by the existing
        # presentation asset system.
        self.router = AssetRouter(
            output_dir=str(
                self.assets_dir
            )
        )

        # New FLUX.2 backend.
        self.flux2 = Flux2KleinManager(
            output_dir=self.assets_dir
        )

    # =========================================================
    # GENERATE ALL ASSETS
    # =========================================================

    def generate_all(self):
        """
        Generate every pending asset in the manifest.

        Image assets are processed as a batch.

        FLUX.2 runtime validation:
            â†“
        image 1
            â†“
        image 2
            â†“
        illustration
            â†“
        photo
            â†“
        release runtime

        No Python-side 4B model is kept in VRAM between assets.
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

            # -------------------------------------------------
            # IMAGE ASSETS
            # -------------------------------------------------

            self._generate_image_assets(
                assets
            )

            # -------------------------------------------------
            # NON-IMAGE ASSETS
            # -------------------------------------------------

            self._process_non_image_assets(
                assets
            )

            # -------------------------------------------------
            # FINAL STATUS
            # -------------------------------------------------

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
                    "ASSET GENERATION "
                    "PARTIALLY COMPLETE"
                )
                print("================================")

                self._print_remaining_assets(
                    final_assets
                )

        except Exception as exc:

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
        """

        image_assets = [
            asset
            for asset in assets
            if asset.get(
                "type",
                "",
            ).strip().lower()
            in self.IMAGE_ASSET_TYPES
            and asset.get(
                "status"
            ) != "completed"
        ]

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
        # Validate FLUX.2 runtime once
        # -----------------------------------------------------

        print()
        print(
            "Preparing FLUX.2 Klein..."
        )

        self.flux2.load()

        try:

            # -------------------------------------------------
            # Generate sequentially
            # -------------------------------------------------

            for asset in image_assets:

                self.generate_asset(
                    asset
                )

        finally:

            print()
            print(
                "Image generation batch finished."
            )

            self._unload_flux()

            print(
                "FLUX.2 Klein runtime released."
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

        none
            -> completed

        icon
            -> pending

        diagram
            -> pending

        chart
            -> pending
        """

        for asset in assets:

            asset_type = (
                asset.get(
                    "type",
                    "",
                )
                .strip()
                .lower()
            )

            # -------------------------------------------------
            # Image assets already processed
            # -------------------------------------------------

            if asset_type in self.IMAGE_ASSET_TYPES:
                continue

            # -------------------------------------------------
            # NONE
            # -------------------------------------------------

            if asset_type in self.NONE_ASSET_TYPES:

                if asset.get(
                    "status",
                    "pending",
                ) != "completed":

                    self.manifest.update_asset(
                        asset["id"],
                        status="completed",
                        source="none",
                        path=None,
                        visual_spec=asset.get(
                            "visual_spec"
                        ),
                        error=None,
                    )

                    print()
                    print(
                        "[SKIPPED] No visual "
                        f"asset required: "
                        f"{asset['id']}"
                    )

                continue

            # -------------------------------------------------
            # Already completed
            # -------------------------------------------------

            if asset.get(
                "status",
                "pending",
            ) == "completed":

                continue

            # -------------------------------------------------
            # Future generators
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
        # NONE
        # -----------------------------------------------------

        if asset_type in self.NONE_ASSET_TYPES:

            self.manifest.update_asset(
                asset_id,
                status="completed",
                source="none",
                path=None,
                visual_spec=visual_spec,
                error=None,
            )

            print(
                f"[SKIPPED] No visual "
                f"asset required: {asset_id}"
            )

            return None

        # -----------------------------------------------------
        # MARK GENERATING
        # -----------------------------------------------------

        self.manifest.update_asset(
            asset_id,
            status="generating",
            error=None,
        )

        try:

            # -------------------------------------------------
            # IMAGE / ILLUSTRATION / PHOTO
            # -------------------------------------------------

            if asset_type in self.IMAGE_ASSET_TYPES:

                return self._generate_image(
                    asset
                )

            # -------------------------------------------------
            # ICON
            # -------------------------------------------------

            if asset_type == "icon":

                return self._generate_icon(
                    asset
                )

            # -------------------------------------------------
            # DIAGRAM
            # -------------------------------------------------

            if asset_type == "diagram":

                return self._generate_diagram(
                    asset
                )

            # -------------------------------------------------
            # CHART
            # -------------------------------------------------

            if asset_type == "chart":

                return self._generate_chart(
                    asset
                )

            raise ValueError(
                f"Unsupported asset type: "
                f"{asset_type}"
            )

        except Exception as exc:

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
        Generate image-like asset using FLUX.2 Klein.

        Description + visual_spec
            â†“
        AssetPromptBuilder
            â†“
        FLUX.2 prompt
            â†“
        stable-diffusion.cpp
            â†“
        PNG
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
        # Build prompt using existing AssetPromptBuilder
        # -----------------------------------------------------

        prompt = self._build_asset_prompt(
            asset_type=asset_type,
            description=description,
            visual_spec=visual_spec,
        )

        # -----------------------------------------------------
        # Output filename
        # -----------------------------------------------------

        output_path = (
            self.assets_dir
            / f"{asset_id}.png"
        )

        # -----------------------------------------------------
        # Generate using FLUX.2
        # -----------------------------------------------------

        output_path = self.flux2.generate(
            prompt=prompt,
            output_path=output_path,
            seed=42,
        )

        # -----------------------------------------------------
        # Store generation information
        # -----------------------------------------------------

        self.manifest.update_asset(
            asset_id,

            status="completed",

            source="flux2-klein",

            model=(
                "flux-2-klein-4b-Q4_0.gguf"
            ),

            runtime="stable-diffusion.cpp",

            prompt=prompt,

            visual_spec=visual_spec,

            seed=42,

            width=self.flux2.width,

            height=self.flux2.height,

            steps=self.flux2.steps,

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
        Build the exact prompt using the existing
        AssetPromptBuilder from AssetRouter.
        """

        prompt_builder = getattr(
            self.router,
            "prompt_builder",
            None,
        )

        if prompt_builder is None:

            return description

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
            f"not implemented yet: "
            f"{asset_id}"
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
            f"not implemented yet: "
            f"{asset_id}"
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
            f"not implemented yet: "
            f"{asset_id}"
        )

        return None

    # =========================================================
    # UNLOAD FLUX.2
    # =========================================================

    def _unload_flux(self):
        """
        Release the FLUX.2 runtime.
        """

        try:

            if self.flux2.is_loaded():

                print()
                print(
                    "Releasing FLUX.2 Klein..."
                )

                self.flux2.unload()

            else:

                print(
                    "FLUX.2 Klein is already "
                    "released."
                )

        except Exception as exc:

            print()
            print(
                "Warning: FLUX.2 release failed:"
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
        Return True only when every asset is completed.
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
        print(
            "Remaining assets:"
        )

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

        Only incomplete assets are processed.
        """

        print()
        print("================================")
        print("RESUMING ASSET GENERATION")
        print("================================")

        self.generate_all()
