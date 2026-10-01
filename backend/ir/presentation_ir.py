from dataclasses import dataclass, field
from typing import Any


@dataclass
class ThemeIR:
    name: str = "default"
    font_family: str = "Aptos"
    primary_color: str = "#2563EB"
    secondary_color: str = "#64748B"
    background_color: str = "#FFFFFF"
    text_color: str = "#111827"


@dataclass
class AssetIR:
    id: str
    asset_type: str
    description: str
    source: str = "pending"
    path: str | None = None


@dataclass
class ElementIR:
    type: str
    role: str
    content: Any = None
    asset_id: str | None = None
    x: float | None = None
    y: float | None = None
    width: float | None = None
    height: float | None = None


@dataclass
class SlideIR:
    id: str
    slide_number: int
    title: str
    purpose: str
    layout: str
    elements: list[ElementIR] = field(default_factory=list)
    asset_ids: list[str] = field(default_factory=list)


@dataclass
class PresentationIR:
    title: str
    subtitle: str
    slide_count: int
    theme: ThemeIR
    slides: list[SlideIR]
    assets: list[AssetIR] = field(default_factory=list)

    def validate(self):
        if not self.title.strip():
            raise ValueError("Presentation title cannot be empty.")

        if self.slide_count != len(self.slides):
            raise ValueError(
                "slide_count does not match number of slides."
            )

        slide_numbers = [
            slide.slide_number
            for slide in self.slides
        ]

        expected_numbers = list(
            range(1, self.slide_count + 1)
        )

        if slide_numbers != expected_numbers:
            raise ValueError(
                f"Invalid slide numbering: {slide_numbers}"
            )

        asset_ids = {
            asset.id
            for asset in self.assets
        }

        for slide in self.slides:

            for asset_id in slide.asset_ids:

                if asset_id not in asset_ids:
                    raise ValueError(
                        f"Slide {slide.slide_number} references "
                        f"unknown asset: {asset_id}"
                    )

            for element in slide.elements:

                if element.asset_id is not None:
                    if element.asset_id not in asset_ids:
                        raise ValueError(
                            f"Element references unknown asset: "
                            f"{element.asset_id}"
                        )

        return True

    def to_dict(self) -> dict:

        return {
            "presentation": {
                "title": self.title,
                "subtitle": self.subtitle,
                "slide_count": self.slide_count,
            },

            "theme": {
                "name": self.theme.name,
                "font_family": self.theme.font_family,
                "primary_color": self.theme.primary_color,
                "secondary_color": self.theme.secondary_color,
                "background_color": self.theme.background_color,
                "text_color": self.theme.text_color,
            },

            "assets": [
                {
                    "id": asset.id,
                    "type": asset.asset_type,
                    "description": asset.description,
                    "source": asset.source,
                    "path": asset.path,
                }
                for asset in self.assets
            ],

            "slides": [
                {
                    "id": slide.id,
                    "slide_number": slide.slide_number,
                    "title": slide.title,
                    "purpose": slide.purpose,
                    "layout": slide.layout,

                    "asset_ids": slide.asset_ids,

                    "elements": [
                        {
                            "type": element.type,
                            "role": element.role,
                            "content": element.content,
                            "asset_id": element.asset_id,
                            "x": element.x,
                            "y": element.y,
                            "width": element.width,
                            "height": element.height,
                        }
                        for element in slide.elements
                    ],
                }
                for slide in self.slides
            ],
        }