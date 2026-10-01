from backend.ir.presentation_ir import (
    PresentationIR,
    ThemeIR,
    AssetIR,
    ElementIR,
    SlideIR,
)


class PresentationIRBuilder:

    def build(self, plan: dict) -> PresentationIR:

        assets = []
        slides = []

        # ---------------------------------------------------------
        # Build global assets
        # ---------------------------------------------------------

        asset_counter = 1

        for slide in plan["slides"]:

            for asset in slide["assets"]:

                asset_id = f"asset-{asset_counter:03d}"

                assets.append(
                    AssetIR(
                        id=asset_id,
                        asset_type=asset["type"],
                        description=asset["description"],
                    )
                )

                asset_counter += 1

        # ---------------------------------------------------------
        # Build slides
        # ---------------------------------------------------------

        asset_index = 0

        for slide_data in plan["slides"]:

            slide_asset_ids = []

            elements = [
                ElementIR(
                    type="text",
                    role="title",
                    content=slide_data["title"],
                )
            ]

            # Add key points
            for point in slide_data["key_points"]:

                elements.append(
                    ElementIR(
                        type="text",
                        role="bullet",
                        content=point,
                    )
                )

            # Connect slide assets
            asset_count = len(slide_data["assets"])

            for _ in range(asset_count):

                asset = assets[asset_index]

                slide_asset_ids.append(asset.id)

                elements.append(
                    ElementIR(
                        type=asset.asset_type,
                        role="visual",
                        asset_id=asset.id,
                    )
                )

                asset_index += 1

            slides.append(
                SlideIR(
                    id=f"slide-{slide_data['slide_number']:03d}",
                    slide_number=slide_data["slide_number"],
                    title=slide_data["title"],
                    purpose=slide_data["purpose"],
                    layout=slide_data["layout"],
                    elements=elements,
                    asset_ids=slide_asset_ids,
                )
            )

        # ---------------------------------------------------------
        # Create presentation
        # ---------------------------------------------------------

        presentation = PresentationIR(
            title=plan["title"],
            subtitle=plan["subtitle"],
            slide_count=plan["slide_count"],

            theme=ThemeIR(),

            slides=slides,

            assets=assets,
        )

        # ---------------------------------------------------------
        # Validate before returning
        # ---------------------------------------------------------

        presentation.validate()

        return presentation