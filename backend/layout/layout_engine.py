from backend.ir.presentation_ir import SlideIR

from backend.layout.geometry import (
    SLIDE_WIDTH,
    SLIDE_HEIGHT,
    Rect,
    TextStyle,
    PositionedElement,
)


class LayoutEngine:
    """
    Converts Presentation IR into positioned slide elements.

    Responsibilities:
    - Map logical layouts to physical coordinates.
    - Position titles, bullets, visuals, charts, etc.
    - Keep elements inside slide boundaries.
    - Detect accidental element collisions.
    """

    MARGIN = 0.55
    GAP = 0.25

    def layout_slide(self, slide: SlideIR) -> list[PositionedElement]:
        handlers = {
            "hero": self._hero,
            "title_content": self._title_content,
            "two_column": self._two_column,
            "three_column": self._three_column,
            "image_left": self._image_left,
            "image_right": self._image_right,
            "big_stat": self._big_stat,
            "timeline": self._timeline,
            "process": self._process,
            "comparison": self._comparison,
            "quote": self._quote,
            "full_image": self._full_image,
            "diagram": self._diagram,
            "chart": self._chart,
            "cards": self._cards,
            "section_divider": self._section_divider,
        }

        handler = handlers.get(slide.layout)

        if handler is None:
            raise ValueError(
                f"Unsupported layout: {slide.layout}"
            )

        elements = handler(slide)

        # Validate geometry.
        self._validate_bounds(elements)

        # Validate collisions.
        self._validate_collisions(elements)

        return elements

    # ---------------------------------------------------------
    # HERO
    # ---------------------------------------------------------

    def _hero(self, slide):
        elements = []

        title = self._find_title(slide)

        if title:
            elements.append(
                self._text(
                    slide,
                    "title",
                    title.content,
                    Rect(
                        self.MARGIN,
                        0.9,
                        SLIDE_WIDTH - 2 * self.MARGIN,
                        1.0,
                    ),
                    34,
                    True,
                    "center",
                )
            )

        visual = self._find_visual(slide)

        if visual:
            elements.append(
                PositionedElement(
                    element_id=f"{slide.id}-visual",
                    element_type=visual.type,
                    role="visual",
                    content=visual.content,
                    rect=Rect(
                        3.0,
                        2.2,
                        7.333,
                        4.5,
                    ),
                    asset_id=visual.asset_id,
                )
            )

        return elements

    # ---------------------------------------------------------
    # TITLE + CONTENT
    # ---------------------------------------------------------

    def _title_content(self, slide):
        elements = []

        title = self._find_title(slide)

        if title:
            elements.append(
                self._text(
                    slide,
                    "title",
                    title.content,
                    Rect(
                        self.MARGIN,
                        0.45,
                        SLIDE_WIDTH - 2 * self.MARGIN,
                        0.8,
                    ),
                    30,
                    True,
                )
            )

        bullets = self._bullets(slide)

        y = 1.6

        for index, bullet in enumerate(bullets):
            elements.append(
                self._text(
                    slide,
                    f"bullet-{index + 1}",
                    bullet.content,
                    Rect(
                        self.MARGIN,
                        y,
                        SLIDE_WIDTH - 2 * self.MARGIN,
                        0.65,
                    ),
                    22,
                )
            )

            y += 0.75

        return elements

    # ---------------------------------------------------------
    # TWO COLUMN
    # ---------------------------------------------------------

    def _two_column(self, slide):
        elements = self._title(slide)

        bullets = self._bullets(slide)

        available_width = (
            SLIDE_WIDTH
            - 2 * self.MARGIN
            - self.GAP
        )

        column_width = available_width / 2

        for index, bullet in enumerate(bullets):
            column = index % 2
            row = index // 2

            x = (
                self.MARGIN
                + column * (column_width + self.GAP)
            )

            y = 1.65 + row * 1.35

            elements.append(
                self._text(
                    slide,
                    f"bullet-{index + 1}",
                    bullet.content,
                    Rect(
                        x,
                        y,
                        column_width,
                        1.0,
                    ),
                    20,
                )
            )

        return elements

    # ---------------------------------------------------------
    # THREE COLUMN
    # ---------------------------------------------------------

    def _three_column(self, slide):
        elements = self._title(slide)

        bullets = self._bullets(slide)

        available_width = (
            SLIDE_WIDTH
            - 2 * self.MARGIN
            - 2 * self.GAP
        )

        column_width = available_width / 3

        for index, bullet in enumerate(bullets[:3]):
            x = (
                self.MARGIN
                + index * (column_width + self.GAP)
            )

            elements.append(
                self._text(
                    slide,
                    f"bullet-{index + 1}",
                    bullet.content,
                    Rect(
                        x,
                        1.8,
                        column_width,
                        2.2,
                    ),
                    18,
                )
            )

        return elements

    # ---------------------------------------------------------
    # IMAGE LEFT
    # ---------------------------------------------------------

    def _image_left(self, slide):
        elements = self._title(slide)

        visual = self._find_visual(slide)

        if visual:
            elements.append(
                self._visual(
                    slide,
                    visual,
                    Rect(
                        self.MARGIN,
                        1.65,
                        5.5,
                        5.2,
                    ),
                )
            )

        self._add_bullets(
            elements,
            slide,
            x=6.3,
            y=1.8,
            width=6.4,
        )

        return elements

    # ---------------------------------------------------------
    # IMAGE RIGHT
    # ---------------------------------------------------------

    def _image_right(self, slide):
        elements = self._title(slide)

        self._add_bullets(
            elements,
            slide,
            x=self.MARGIN,
            y=1.8,
            width=6.0,
        )

        visual = self._find_visual(slide)

        if visual:
            elements.append(
                self._visual(
                    slide,
                    visual,
                    Rect(
                        7.0,
                        1.65,
                        5.78,
                        5.2,
                    ),
                )
            )

        return elements

    # ---------------------------------------------------------
    # BIG STAT
    # ---------------------------------------------------------

    def _big_stat(self, slide):
        elements = self._title(slide)

        bullets = self._bullets(slide)

        if bullets:
            elements.append(
                self._text(
                    slide,
                    "big-stat",
                    bullets[0].content,
                    Rect(
                        1.0,
                        2.0,
                        11.33,
                        1.6,
                    ),
                    42,
                    True,
                    "center",
                )
            )

            for index, bullet in enumerate(bullets[1:]):
                elements.append(
                    self._text(
                        slide,
                        f"bullet-{index + 2}",
                        bullet.content,
                        Rect(
                            1.0 + index * 6.0,
                            4.2,
                            5.5,
                            1.0,
                        ),
                        18,
                        False,
                        "center",
                    )
                )

        return elements

    # ---------------------------------------------------------
    # TIMELINE
    # ---------------------------------------------------------

    def _timeline(self, slide):
        elements = self._title(slide)

        bullets = self._bullets(slide)

        count = max(len(bullets), 1)

        width = (
            SLIDE_WIDTH
            - 2 * self.MARGIN
            - (count - 1) * self.GAP
        ) / count

        for index, bullet in enumerate(bullets):
            x = (
                self.MARGIN
                + index * (width + self.GAP)
            )

            elements.append(
                self._text(
                    slide,
                    f"timeline-{index + 1}",
                    bullet.content,
                    Rect(
                        x,
                        2.4,
                        width,
                        2.0,
                    ),
                    17,
                    False,
                    "center",
                )
            )

        return elements

    # ---------------------------------------------------------
    # PROCESS
    # ---------------------------------------------------------

    def _process(self, slide):
        elements = self._title(slide)

        bullets = self._bullets(slide)

        count = max(len(bullets), 1)

        width = (
            SLIDE_WIDTH
            - 2 * self.MARGIN
            - (count - 1) * self.GAP
        ) / count

        for index, bullet in enumerate(bullets):
            x = (
                self.MARGIN
                + index * (width + self.GAP)
            )

            elements.append(
                self._text(
                    slide,
                    f"process-{index + 1}",
                    bullet.content,
                    Rect(
                        x,
                        2.0,
                        width,
                        2.5,
                    ),
                    17,
                    False,
                    "center",
                )
            )

        return elements

    # ---------------------------------------------------------
    # COMPARISON
    # ---------------------------------------------------------

    def _comparison(self, slide):
        elements = self._title(slide)

        bullets = self._bullets(slide)

        half = max((len(bullets) + 1) // 2, 1)

        column_width = (
            SLIDE_WIDTH
            - 2 * self.MARGIN
            - self.GAP
        ) / 2

        for index, bullet in enumerate(bullets):
            column = 0 if index < half else 1

            row = (
                index
                if column == 0
                else index - half
            )

            x = (
                self.MARGIN
                + column * (column_width + self.GAP)
            )

            y = 1.7 + row * 1.0

            elements.append(
                self._text(
                    slide,
                    f"comparison-{index + 1}",
                    bullet.content,
                    Rect(
                        x,
                        y,
                        column_width,
                        0.8,
                    ),
                    18,
                )
            )

        return elements

    # ---------------------------------------------------------
    # QUOTE
    # ---------------------------------------------------------

    def _quote(self, slide):
        elements = self._title(slide)

        bullets = self._bullets(slide)

        if bullets:
            elements.append(
                self._text(
                    slide,
                    "quote",
                    bullets[0].content,
                    Rect(
                        1.2,
                        2.2,
                        10.9,
                        2.0,
                    ),
                    32,
                    False,
                    "center",
                )
            )

        return elements

    # ---------------------------------------------------------
    # FULL IMAGE
    # ---------------------------------------------------------

    def _full_image(self, slide):
        elements = []

        visual = self._find_visual(slide)

        if visual:
            elements.append(
                self._visual(
                    slide,
                    visual,
                    Rect(
                        0,
                        0,
                        SLIDE_WIDTH,
                        SLIDE_HEIGHT,
                    ),
                )
            )

        title = self._find_title(slide)

        if title:
            elements.append(
                self._text(
                    slide,
                    "title",
                    title.content,
                    Rect(
                        0.8,
                        5.8,
                        11.7,
                        1.0,
                    ),
                    34,
                    True,
                    "center",
                )
            )

        return elements

    # ---------------------------------------------------------
    # DIAGRAM
    # ---------------------------------------------------------

    def _diagram(self, slide):
        elements = self._title(slide)

        visual = self._find_visual(slide)

        if visual:
            elements.append(
                self._visual(
                    slide,
                    visual,
                    Rect(
                        1.0,
                        1.6,
                        11.33,
                        5.3,
                    ),
                )
            )

        return elements

    # ---------------------------------------------------------
    # CHART
    # ---------------------------------------------------------

    def _chart(self, slide):
        elements = self._title(slide)

        visual = self._find_visual(slide)

        if visual:
            elements.append(
                self._visual(
                    slide,
                    visual,
                    Rect(
                        1.0,
                        1.7,
                        11.33,
                        5.0,
                    ),
                )
            )

        return elements

    # ---------------------------------------------------------
    # CARDS
    # ---------------------------------------------------------

    def _cards(self, slide):
        elements = self._title(slide)

        bullets = self._bullets(slide)

        count = min(len(bullets), 6)

        if count == 0:
            return elements

        columns = 3
        rows = (count + columns - 1) // columns

        width = (
            SLIDE_WIDTH
            - 2 * self.MARGIN
            - (columns - 1) * self.GAP
        ) / columns

        height = 1.8

        for index in range(count):
            column = index % columns
            row = index // columns

            x = (
                self.MARGIN
                + column * (width + self.GAP)
            )

            y = 1.7 + row * (height + self.GAP)

            elements.append(
                self._text(
                    slide,
                    f"card-{index + 1}",
                    bullets[index].content,
                    Rect(
                        x,
                        y,
                        width,
                        height,
                    ),
                    17,
                    False,
                    "center",
                )
            )

        return elements

    # ---------------------------------------------------------
    # SECTION DIVIDER
    # ---------------------------------------------------------

    def _section_divider(self, slide):
        elements = []

        title = self._find_title(slide)

        if title:
            elements.append(
                self._text(
                    slide,
                    "title",
                    title.content,
                    Rect(
                        1.0,
                        2.7,
                        11.33,
                        1.2,
                    ),
                    40,
                    True,
                    "center",
                )
            )

        return elements

    # ---------------------------------------------------------
    # HELPERS
    # ---------------------------------------------------------

    def _title(self, slide):
        title = self._find_title(slide)

        if not title:
            return []

        return [
            self._text(
                slide,
                "title",
                title.content,
                Rect(
                    self.MARGIN,
                    0.45,
                    SLIDE_WIDTH - 2 * self.MARGIN,
                    0.8,
                ),
                30,
                True,
            )
        ]

    def _add_bullets(
        self,
        elements,
        slide,
        x,
        y,
        width,
    ):
        bullets = self._bullets(slide)

        for index, bullet in enumerate(bullets):
            elements.append(
                self._text(
                    slide,
                    f"bullet-{index + 1}",
                    bullet.content,
                    Rect(
                        x,
                        y + index * 0.85,
                        width,
                        0.65,
                    ),
                    19,
                )
            )

    @staticmethod
    def _bullets(slide):
        return [
            element
            for element in slide.elements
            if element.role == "bullet"
        ]

    @staticmethod
    def _find_title(slide):
        for element in slide.elements:
            if element.role == "title":
                return element

        return None

    @staticmethod
    def _find_visual(slide):
        for element in slide.elements:
            if element.role == "visual":
                return element

        return None

    @staticmethod
    def _text(
        slide,
        element_id,
        content,
        rect,
        font_size,
        bold=False,
        alignment="left",
    ):
        return PositionedElement(
            element_id=f"{slide.id}-{element_id}",
            element_type="text",
            role=element_id,
            content=content,
            rect=rect,
            style=TextStyle(
                font_size=font_size,
                bold=bold,
                alignment=alignment,
            ),
        )

    @staticmethod
    def _visual(
        slide,
        visual,
        rect,
    ):
        return PositionedElement(
            element_id=f"{slide.id}-visual",
            element_type=visual.type,
            role="visual",
            content=visual.content,
            rect=rect,
            asset_id=visual.asset_id,
        )

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    @staticmethod
    def _validate_bounds(elements):
        """
        Ensure every positioned element remains inside
        the PowerPoint slide boundaries.
        """

        for element in elements:
            rect = element.rect

            if rect.x < 0:
                raise ValueError(
                    f"{element.element_id} exceeds left boundary."
                )

            if rect.y < 0:
                raise ValueError(
                    f"{element.element_id} exceeds top boundary."
                )

            if rect.right > SLIDE_WIDTH:
                raise ValueError(
                    f"{element.element_id} exceeds right boundary."
                )

            if rect.bottom > SLIDE_HEIGHT:
                raise ValueError(
                    f"{element.element_id} exceeds bottom boundary."
                )

    @staticmethod
    def _rectangles_overlap(a: Rect, b: Rect) -> bool:
        """
        Return True if two rectangles overlap.

        Touching edges are not considered a collision.
        """

        return not (
            a.right <= b.x
            or b.right <= a.x
            or a.bottom <= b.y
            or b.bottom <= a.y
        )

    @classmethod
    def _validate_collisions(cls, elements):
        """
        Detect accidental overlap between positioned elements.

        Visual elements are ignored here because a visual may
        intentionally sit behind or alongside another element,
        especially in layouts such as full_image.
        """

        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                first = elements[i]
                second = elements[j]

                # Visuals may intentionally overlap text.
                if (
                    first.role == "visual"
                    or second.role == "visual"
                ):
                    continue

                if cls._rectangles_overlap(
                    first.rect,
                    second.rect,
                ):
                    raise ValueError(
                        "Layout collision detected between "
                        f"{first.element_id} and "
                        f"{second.element_id}"
                    )