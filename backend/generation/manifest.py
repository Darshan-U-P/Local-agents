from pathlib import Path
import json
from datetime import datetime


class GenerationManifest:
    """
    Manages persistent JSON manifests for a presentation
    generation session.
    """

    def __init__(self, session_dir: str):
        self.session_dir = Path(session_dir)

        self.session_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.presentation_plan_path = (
            self.session_dir / "presentation_plan.json"
        )

        self.asset_manifest_path = (
            self.session_dir / "asset_manifest.json"
        )

    # ---------------------------------------------------------
    # PRESENTATION PLAN
    # ---------------------------------------------------------

    def save_presentation_plan(
        self,
        plan: dict,
    ):
        """
        Save the Qwen-generated presentation plan.
        """

        self._save_json(
            self.presentation_plan_path,
            plan,
        )

    def load_presentation_plan(self) -> dict:
        """
        Load the saved presentation plan.
        """

        return self._load_json(
            self.presentation_plan_path
        )

    # ---------------------------------------------------------
    # ASSET MANIFEST
    # ---------------------------------------------------------

    def initialize_assets(
        self,
        assets: list[dict],
    ):
        """
        Create the initial asset manifest.
        """

        manifest = {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "assets": assets,
        }

        self._save_json(
            self.asset_manifest_path,
            manifest,
        )

    def load_assets(self) -> dict:
        """
        Load the asset manifest.
        """

        return self._load_json(
            self.asset_manifest_path
        )

    # ---------------------------------------------------------
    # UPDATE ASSET
    # ---------------------------------------------------------

    def update_asset(
        self,
        asset_id: str,
        **updates,
    ):
        """
        Update information about a generated asset.

        Example:

            update_asset(
                "asset-001",
                status="completed",
                path="assets/asset-001.png",
                source="flux"
            )
        """

        manifest = self.load_assets()

        for asset in manifest["assets"]:

            if asset["id"] == asset_id:

                asset.update(updates)

                manifest["updated_at"] = (
                    datetime.now().isoformat()
                )

                self._save_json(
                    self.asset_manifest_path,
                    manifest,
                )

                return

        raise ValueError(
            f"Asset not found: {asset_id}"
        )

    # ---------------------------------------------------------
    # ADD ASSET
    # ---------------------------------------------------------

    def add_asset(
        self,
        asset: dict,
    ):
        """
        Add a new asset to the manifest.
        """

        if self.asset_manifest_path.exists():
            manifest = self.load_assets()
        else:
            manifest = {
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "assets": [],
            }

        manifest["assets"].append(asset)

        manifest["updated_at"] = (
            datetime.now().isoformat()
        )

        self._save_json(
            self.asset_manifest_path,
            manifest,
        )

    # ---------------------------------------------------------
    # JSON HELPERS
    # ---------------------------------------------------------

    @staticmethod
    def _save_json(
        path: Path,
        data: dict,
    ):
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
            )

    @staticmethod
    def _load_json(
        path: Path,
    ) -> dict:

        if not path.exists():
            raise FileNotFoundError(
                f"Manifest file not found: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)