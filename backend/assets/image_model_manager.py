from pathlib import Path
import gc
import shutil

import torch

from diffusers import (
    FluxPipeline,
    FluxTransformer2DModel,
    GGUFQuantizationConfig,
)


class ImageModelManager:
    """
    Manages the FLUX image model lifecycle.

    Lifecycle:

        load()
          ↓
        get_pipeline()
          ↓
        generate images
          ↓
        unload()
          ↓
        GPU / CPU resources released

    FLUX is intentionally loaded only when image generation
    is required. This prevents it from competing with Qwen
    for the laptop's limited VRAM.
    """

    def __init__(
        self,
        model_path: str,
        offload_path: str = "generated/assets/flux_offload_cache",
    ):
        self.model_path = Path(model_path)
        self.offload_path = Path(offload_path)

        self.pipe = None
        self.loaded = False

    # =========================================================
    # LOAD MODEL
    # =========================================================

    def load(self):
        """
        Load the FLUX pipeline.

        If FLUX is already loaded, return the existing pipeline.
        """

        # -----------------------------------------------------
        # Prevent duplicate loading
        # -----------------------------------------------------

        if self.loaded and self.pipe is not None:
            print("FLUX image model already loaded.")
            return self.pipe

        # -----------------------------------------------------
        # Validate model
        # -----------------------------------------------------

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"FLUX model not found: {self.model_path}"
            )

        print()
        print("================================")
        print("LOADING FLUX IMAGE MODEL")
        print("================================")

        print(
            f"Model: {self.model_path}"
        )

        print(
            f"Offload path: {self.offload_path}"
        )

        # -----------------------------------------------------
        # Prepare disk offload directory
        # -----------------------------------------------------

        self.offload_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:

            # -------------------------------------------------
            # Load quantized FLUX transformer
            # -------------------------------------------------

            print()
            print("Loading FLUX transformer...")

            transformer = (
                FluxTransformer2DModel.from_single_file(
                    str(self.model_path),
                    quantization_config=(
                        GGUFQuantizationConfig(
                            compute_dtype=torch.float16
                        )
                    ),
                    torch_dtype=torch.float16,
                )
            )

            # -------------------------------------------------
            # Build FLUX pipeline
            # -------------------------------------------------

            print(
                "Loading FLUX pipeline..."
            )

            self.pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-schnell",
                transformer=transformer,
                torch_dtype=torch.float16,
            )

            # -------------------------------------------------
            # Group offloading
            #
            # Important for 4 GB VRAM systems.
            # -------------------------------------------------

            print(
                "Configuring GPU/RAM/SSD offloading..."
            )

            self.pipe.enable_group_offload(
                onload_device=torch.device(
                    "cuda"
                ),
                offload_device=torch.device(
                    "cpu"
                ),
                offload_type="leaf_level",
                use_stream=False,
                record_stream=False,
                low_cpu_mem_usage=True,
                offload_to_disk_path=str(
                    self.offload_path
                ),
            )

            # -------------------------------------------------
            # VAE memory optimizations
            # -------------------------------------------------

            self.pipe.vae.enable_slicing()
            self.pipe.vae.enable_tiling()

            # -------------------------------------------------
            # Mark model as loaded
            # -------------------------------------------------

            self.loaded = True

            print()
            print("FLUX image model loaded.")
            print("GPU/RAM/SSD offloading enabled.")
            print("================================")

            return self.pipe

        except Exception:

            # -------------------------------------------------
            # If loading fails, clean everything immediately.
            # -------------------------------------------------

            print()
            print(
                "FLUX loading failed."
            )

            self._force_cleanup()

            raise

    # =========================================================
    # GET PIPELINE
    # =========================================================

    def get_pipeline(self):
        """
        Return the loaded FLUX pipeline.

        Automatically loads FLUX if necessary.
        """

        if (
            not self.loaded
            or self.pipe is None
        ):
            return self.load()

        return self.pipe

    # =========================================================
    # UNLOAD MODEL
    # =========================================================

    def unload(self):
        """
        Completely unload FLUX.

        This should be called when the image-generation
        phase is finished.

        The pipeline is deleted, Python garbage collection
        runs, CUDA caches are released, and the temporary
        SSD offload cache is removed.
        """

        print()
        print("================================")
        print("UNLOADING FLUX IMAGE MODEL")
        print("================================")

        # -----------------------------------------------------
        # Check whether anything is loaded
        # -----------------------------------------------------

        if (
            self.pipe is None
            and not self.loaded
        ):
            print(
                "FLUX image model is already unloaded."
            )

            self._cleanup_offload_cache()

            print("================================")

            return

        # -----------------------------------------------------
        # Remove pipeline reference
        # -----------------------------------------------------

        pipe = self.pipe

        self.pipe = None
        self.loaded = False

        # -----------------------------------------------------
        # Explicitly destroy pipeline
        # -----------------------------------------------------

        if pipe is not None:
            del pipe

        # -----------------------------------------------------
        # Python garbage collection
        # -----------------------------------------------------

        gc.collect()

        # -----------------------------------------------------
        # Release CUDA cached memory
        # -----------------------------------------------------

        self._clear_cuda_memory()

        # -----------------------------------------------------
        # Remove temporary SSD offload files
        # -----------------------------------------------------

        self._cleanup_offload_cache()

        # -----------------------------------------------------
        # Run cleanup again after removing cache
        # -----------------------------------------------------

        gc.collect()

        self._clear_cuda_memory()

        print()
        print("FLUX image model unloaded.")
        print("GPU/CPU resources released.")
        print("================================")

    # =========================================================
    # FORCE CLEANUP
    # =========================================================

    def _force_cleanup(self):
        """
        Emergency cleanup used when loading FLUX fails.
        """

        pipe = self.pipe

        self.pipe = None
        self.loaded = False

        if pipe is not None:
            del pipe

        gc.collect()

        self._clear_cuda_memory()

        self._cleanup_offload_cache()

        gc.collect()

        self._clear_cuda_memory()

    # =========================================================
    # CUDA MEMORY CLEANUP
    # =========================================================

    @staticmethod
    def _clear_cuda_memory():
        """
        Release cached CUDA memory.
        """

        if not torch.cuda.is_available():
            return

        try:
            torch.cuda.empty_cache()
        except Exception as exc:
            print(
                f"Warning: torch.cuda.empty_cache() "
                f"failed: {exc}"
            )

        try:
            torch.cuda.ipc_collect()
        except Exception as exc:
            print(
                f"Warning: torch.cuda.ipc_collect() "
                f"failed: {exc}"
            )

    # =========================================================
    # OFFLOAD CACHE CLEANUP
    # =========================================================

    def _cleanup_offload_cache(self):
        """
        Remove the temporary FLUX SSD offload cache.
        """

        if not self.offload_path.exists():
            return

        try:

            shutil.rmtree(
                self.offload_path
            )

            print(
                f"Removed FLUX offload cache: "
                f"{self.offload_path}"
            )

        except Exception as exc:

            print(
                "Warning: could not remove "
                f"FLUX offload cache: {exc}"
            )

    # =========================================================
    # STATUS
    # =========================================================

    def is_loaded(self) -> bool:
        """
        Return True if FLUX is currently loaded.
        """

        return (
            self.loaded
            and self.pipe is not None
        )