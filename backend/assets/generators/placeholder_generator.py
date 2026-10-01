from pathlib import Path


class PlaceholderGenerator:
    """
    Temporary asset generator.

    This will later be replaced or supplemented by:
    - local image generation
    - icon generation
    - diagram generation
    - chart generation
    """

    def generate(
        self,
        description: str,
        output_path: str,
    ) -> str:
        output = Path(output_path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # We intentionally don't create an image yet.
        # The Asset Router only needs to establish
        # the generator interface at this stage.

        return str(output)