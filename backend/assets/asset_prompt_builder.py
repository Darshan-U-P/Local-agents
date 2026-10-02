from typing import Any


class AssetPromptBuilder:
    """
    Converts a structured asset specification into a
    detailed prompt for the appropriate asset generator.

    Input:

        {
            "type": "image",
            "description": "...",
            "visual_spec": {
                "purpose": "...",
                "subject": "...",
                "composition": "...",
                "style": "...",
                "color_palette": [...],
                "must_show": [...],
                "must_avoid": [...],
                "text_policy": "..."
            }
        }

    Output:

        A detailed prompt suitable for the image generator.

    Important:
        This class does NOT generate the asset.
        It only converts structured visual requirements
        into a generator-ready prompt.
    """

    # =========================================================
    # PUBLIC API
    # =========================================================

    def build(
        self,
        asset: dict[str, Any],
    ) -> str:
        """
        Build a generator-ready prompt from an asset definition.

        The method supports:

            image
            illustration
            photo
            icon
            diagram
            chart
            none

        Image-generation prompts are optimized for FLUX.

        Diagram/chart/icon prompts are descriptive and can later
        be consumed by their specialized generators.
        """

        if not isinstance(asset, dict):
            raise ValueError(
                "Asset must be a dictionary."
            )

        asset_type = str(
            asset.get("type", "image")
        ).strip().lower()

        description = str(
            asset.get("description", "")
        ).strip()

        if asset_type == "none":
            return ""

        if not description:
            raise ValueError(
                "Asset description cannot be empty."
            )

        visual_spec = asset.get(
            "visual_spec"
        )

        if not isinstance(
            visual_spec,
            dict,
        ):
            raise ValueError(
                "Asset visual_spec must be a dictionary."
            )

        if asset_type in {
            "image",
            "illustration",
            "photo",
        }:
            return self._build_image_prompt(
                description=description,
                visual_spec=visual_spec,
                asset_type=asset_type,
            )

        if asset_type == "icon":
            return self._build_icon_prompt(
                description=description,
                visual_spec=visual_spec,
            )

        if asset_type == "diagram":
            return self._build_diagram_prompt(
                description=description,
                visual_spec=visual_spec,
            )

        if asset_type == "chart":
            return self._build_chart_prompt(
                description=description,
                visual_spec=visual_spec,
            )

        raise ValueError(
            f"Unsupported asset type: {asset_type}"
        )

    # =========================================================
    # IMAGE PROMPT
    # =========================================================

    def _build_image_prompt(
        self,
        description: str,
        visual_spec: dict[str, Any],
        asset_type: str,
    ) -> str:
        """
        Build a detailed FLUX prompt.

        FLUX should generate the visual itself, while all
        presentation text remains outside the image.
        """

        purpose = self._text(
            visual_spec,
            "purpose",
        )

        subject = self._text(
            visual_spec,
            "subject",
            fallback=description,
        )

        composition = self._text(
            visual_spec,
            "composition",
        )

        style = self._text(
            visual_spec,
            "style",
        )

        color_palette = self._list(
            visual_spec,
            "color_palette",
        )

        must_show = self._list(
            visual_spec,
            "must_show",
        )

        must_avoid = self._list(
            visual_spec,
            "must_avoid",
        )

        text_policy = self._text(
            visual_spec,
            "text_policy",
            fallback="no text",
        )

        lines = []

        # -----------------------------------------------------
        # ROLE
        # -----------------------------------------------------

        lines.append(
            "Create a professional visual asset for a "
            "presentation slide."
        )

        # -----------------------------------------------------
        # ASSET TYPE
        # -----------------------------------------------------

        if asset_type == "photo":
            lines.append(
                "Visual type: realistic professional "
                "photographic scene."
            )

        elif asset_type == "illustration":
            lines.append(
                "Visual type: polished conceptual "
                "editorial illustration."
            )

        else:
            lines.append(
                "Visual type: high-quality conceptual "
                "scientific or technological visualization."
            )

        # -----------------------------------------------------
        # PURPOSE
        # -----------------------------------------------------

        if purpose:
            lines.append(
                f"Purpose: {purpose}"
            )

        # -----------------------------------------------------
        # SUBJECT
        # -----------------------------------------------------

        lines.append(
            f"Main subject: {subject}"
        )

        # -----------------------------------------------------
        # COMPOSITION
        # -----------------------------------------------------

        if composition:
            lines.append(
                f"Composition: {composition}"
            )

        # -----------------------------------------------------
        # STYLE
        # -----------------------------------------------------

        if style:
            lines.append(
                f"Visual style: {style}"
            )

        # -----------------------------------------------------
        # COLOR PALETTE
        # -----------------------------------------------------

        if color_palette:
            lines.append(
                "Color palette: "
                + ", ".join(color_palette)
                + "."
            )

        # -----------------------------------------------------
        # REQUIRED VISUAL ELEMENTS
        # -----------------------------------------------------

        if must_show:
            lines.append(
                "The image must clearly show:"
            )

            for item in must_show:
                lines.append(
                    f"- {item}"
                )

        # -----------------------------------------------------
        # NEGATIVE REQUIREMENTS
        # -----------------------------------------------------

        if must_avoid:
            lines.append(
                "Avoid the following:"
            )

            for item in must_avoid:
                lines.append(
                    f"- {item}"
                )

        # -----------------------------------------------------
        # TEXT POLICY
        # -----------------------------------------------------

        lines.append(
            f"Text policy: {text_policy}."
        )

        # -----------------------------------------------------
        # PRESENTATION-SPECIFIC CONSTRAINTS
        # -----------------------------------------------------

        lines.extend(
            [
                "Do not include watermarks.",
                "Do not include logos unless explicitly required.",
                "Do not include random UI elements.",
                "Do not include unrelated objects.",
                "Keep the composition visually coherent.",
                "Use clear subject separation.",
                "Maintain strong visual hierarchy.",
                "Use clean professional presentation aesthetics.",
                "Do not place important presentation content "
                "inside the generated image.",
                "Leave appropriate negative space where useful "
                "for slide text.",
            ]
        )

        # -----------------------------------------------------
        # IMAGE QUALITY
        # -----------------------------------------------------

        lines.extend(
            [
                "High visual clarity.",
                "Detailed but not cluttered.",
                "Professional presentation quality.",
                "Balanced composition.",
                "Consistent lighting.",
                "High-quality rendering.",
            ]
        )

        return self._join_lines(
            lines
        )

    # =========================================================
    # ICON PROMPT
    # =========================================================

    def _build_icon_prompt(
        self,
        description: str,
        visual_spec: dict[str, Any],
    ) -> str:
        """
        Build a specification for an icon generator.

        Icons will eventually be generated as SVG/vector
        assets rather than relying on FLUX.
        """

        purpose = self._text(
            visual_spec,
            "purpose",
        )

        subject = self._text(
            visual_spec,
            "subject",
            fallback=description,
        )

        composition = self._text(
            visual_spec,
            "composition",
        )

        style = self._text(
            visual_spec,
            "style",
        )

        colors = self._list(
            visual_spec,
            "color_palette",
        )

        must_show = self._list(
            visual_spec,
            "must_show",
        )

        must_avoid = self._list(
            visual_spec,
            "must_avoid",
        )

        lines = [
            "Create a clean vector icon specification.",
            f"Purpose: {purpose}" if purpose else "",
            f"Subject: {subject}",
            f"Composition: {composition}"
            if composition
            else "",
            f"Style: {style}"
            if style
            else "",
        ]

        if colors:
            lines.append(
                "Color palette: "
                + ", ".join(colors)
                + "."
            )

        if must_show:
            lines.append(
                "Required elements:"
            )

            lines.extend(
                f"- {item}"
                for item in must_show
            )

        if must_avoid:
            lines.append(
                "Avoid:"
            )

            lines.extend(
                f"- {item}"
                for item in must_avoid
            )

        lines.extend(
            [
                "Use simple recognizable geometry.",
                "Use a consistent visual language.",
                "Avoid unnecessary detail.",
                "Avoid photorealism.",
                "Avoid gradients unless specifically required.",
                "No watermark.",
                "No unnecessary text.",
                "Designed for use on a presentation slide.",
            ]
        )

        return self._join_lines(
            lines
        )

    # =========================================================
    # DIAGRAM PROMPT
    # =========================================================

    def _build_diagram_prompt(
        self,
        description: str,
        visual_spec: dict[str, Any],
    ) -> str:
        """
        Build a structured diagram specification.

        This is intentionally NOT an image-generation prompt.

        The future diagram generator can convert this
        specification into SVG/PPT shapes.
        """

        purpose = self._text(
            visual_spec,
            "purpose",
        )

        subject = self._text(
            visual_spec,
            "subject",
            fallback=description,
        )

        composition = self._text(
            visual_spec,
            "composition",
        )

        style = self._text(
            visual_spec,
            "style",
        )

        colors = self._list(
            visual_spec,
            "color_palette",
        )

        must_show = self._list(
            visual_spec,
            "must_show",
        )

        must_avoid = self._list(
            visual_spec,
            "must_avoid",
        )

        lines = [
            "Create a clean educational vector diagram specification.",
            f"Purpose: {purpose}" if purpose else "",
            f"Subject: {subject}",
            f"Composition: {composition}"
            if composition
            else "",
            f"Style: {style}"
            if style
            else "",
        ]

        if colors:
            lines.append(
                "Color palette: "
                + ", ".join(colors)
                + "."
            )

        if must_show:
            lines.append(
                "Required diagram elements:"
            )

            lines.extend(
                f"- {item}"
                for item in must_show
            )

        if must_avoid:
            lines.append(
                "Avoid:"
            )

            lines.extend(
                f"- {item}"
                for item in must_avoid
            )

        lines.extend(
            [
                "Use clear nodes, connectors, arrows, "
                "boundaries, and relationships where appropriate.",
                "Use consistent spacing.",
                "Use clear visual hierarchy.",
                "Keep the diagram understandable at slide size.",
                "Prefer vector geometry.",
                "Avoid decorative clutter.",
                "Avoid photorealistic imagery.",
            ]
        )

        return self._join_lines(
            lines
        )

    # =========================================================
    # CHART PROMPT
    # =========================================================

    def _build_chart_prompt(
        self,
        description: str,
        visual_spec: dict[str, Any],
    ) -> str:
        """
        Build a chart specification.

        Charts should eventually be generated using real chart
        data and rendered as editable/vector presentation
        elements rather than generated by FLUX.
        """

        purpose = self._text(
            visual_spec,
            "purpose",
        )

        subject = self._text(
            visual_spec,
            "subject",
            fallback=description,
        )

        composition = self._text(
            visual_spec,
            "composition",
        )

        style = self._text(
            visual_spec,
            "style",
        )

        colors = self._list(
            visual_spec,
            "color_palette",
        )

        must_show = self._list(
            visual_spec,
            "must_show",
        )

        must_avoid = self._list(
            visual_spec,
            "must_avoid",
        )

        lines = [
            "Create a presentation chart specification.",
            f"Purpose: {purpose}" if purpose else "",
            f"Chart subject: {subject}",
            f"Composition: {composition}"
            if composition
            else "",
            f"Style: {style}"
            if style
            else "",
        ]

        if colors:
            lines.append(
                "Color palette: "
                + ", ".join(colors)
                + "."
            )

        if must_show:
            lines.append(
                "Required chart elements:"
            )

            lines.extend(
                f"- {item}"
                for item in must_show
            )

        if must_avoid:
            lines.append(
                "Avoid:"
            )

            lines.extend(
                f"- {item}"
                for item in must_avoid
            )

        lines.extend(
            [
                "Use accurate data only.",
                "Do not fabricate numerical values.",
                "Use editable chart labels.",
                "Use clear axes.",
                "Use readable legends when necessary.",
                "Use a clean presentation style.",
                "Avoid unnecessary decoration.",
            ]
        )

        return self._join_lines(
            lines
        )

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _text(
        data: dict[str, Any],
        key: str,
        fallback: str = "",
    ) -> str:

        value = data.get(
            key,
            fallback,
        )

        if value is None:
            return fallback

        return str(value).strip()

    @staticmethod
    def _list(
        data: dict[str, Any],
        key: str,
    ) -> list[str]:

        value = data.get(
            key,
            [],
        )

        if value is None:
            return []

        if not isinstance(
            value,
            list,
        ):
            return [
                str(value).strip()
            ]

        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    @staticmethod
    def _join_lines(
        lines: list[str],
    ) -> str:

        return "\n".join(
            line
            for line in lines
            if line
        ).strip()