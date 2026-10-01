from dataclasses import dataclass


# Standard 16:9 presentation dimensions.
# Units are abstract layout units.
SLIDE_WIDTH = 13.333
SLIDE_HEIGHT = 7.5


@dataclass
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height


@dataclass
class TextStyle:
    font_size: float = 24
    bold: bool = False
    alignment: str = "left"


@dataclass
class PositionedElement:
    element_id: str
    element_type: str
    role: str
    content: object
    rect: Rect
    style: TextStyle | None = None
    asset_id: str | None = None