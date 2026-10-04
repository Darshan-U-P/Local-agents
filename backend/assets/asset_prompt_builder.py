from typing import Any


class AssetPromptBuilder:
    """
    Converts structured asset specifications into generator-ready prompts.

    The full visual specification is preserved by the calling system.

    For FLUX image generation, this builder deliberately creates a
    compact prompt because the current text encoder has a 77-token
    input limit.

    Priority for FLUX prompts:

        1. Subject
        2. Composition
        3. Style
        4. Important visual elements
        5. Color palette
        6. Important constraints

    The builder does NOT generate assets.
    It only converts structured visual requirements into prompts.
    """

    # =========================================================
    # CONFIGURATION
    # =========================================================

    # FLUX is using a CLIP-based text encoder with a 77-token limit
    # in the current pipeline.
    #
    # We intentionally stay below that limit rather than trying to
    # use the full 77 tokens.
    MAX_FLUX_WORDS = 65

    # Keep only the most important items from list-based fields.
    MAX_MUST_SHOW = 4
    MAX_MUST_AVOID = 3
    MAX_COLORS = 4

    # =========================================================
    # PUBLIC API
    # =========================================================

    def build(
        self,
        asset: dict[str, Any],
    ) -> str:
        """
        Build a generator-ready prompt from an asset definition.

        Supported types:

            image
            illustration
            photo
            icon
            diagram
            chart
            none
        """

        if not isinstance(asset, dict):
            raise ValueError(
                "Asset must be a dictionary."
            )

        asset_type = str(
            asset.get(
                "type",
                "image",
            )
        ).strip().lower()

        description = str(
            asset.get(
                "description",
                "",
            )
        ).strip()

        # -----------------------------------------------------
        # No visual asset required.
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Image-like assets
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Icon
        # -----------------------------------------------------

        if asset_type == "icon":
            return self._build_icon_prompt(
                description=description,
                visual_spec=visual_spec,
            )

        # -----------------------------------------------------
        # Diagram
        # -----------------------------------------------------

        if asset_type == "diagram":
            return self._build_diagram_prompt(
                description=description,
                visual_spec=visual_spec,
            )

        # -----------------------------------------------------
        # Chart
        # -----------------------------------------------------

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
        Build a compact FLUX prompt.

        IMPORTANT:

        Do not add long generic instructions here.

        The visual specification already contains the information
        needed to describe the image. Repeating generic instructions
        wastes the limited CLIP prompt capacity.

        Target:

            <= approximately 65 whitespace-separated words

        This provides safety margin below the 77-token CLIP limit.
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
        )[: self.MAX_COLORS]

        must_show = self._list(
            visual_spec,
            "must_show",
        )[: self.MAX_MUST_SHOW]

        must_avoid = self._list(
            visual_spec,
            "must_avoid",
        )[: self.MAX_MUST_AVOID]

        text_policy = self._text(
            visual_spec,
            "text_policy",
            fallback="no text",
        )

        # -----------------------------------------------------
        # Asset type descriptor
        # -----------------------------------------------------

        if asset_type == "photo":

            visual_type = (
                "realistic professional photograph"
            )

        elif asset_type == "illustration":

            visual_type = (
                "polished conceptual illustration"
            )

        else:

            visual_type = (
                "high-quality scientific or "
                "technological visualization"
            )

        # -----------------------------------------------------
        # Build compact prompt components.
        #
        # Order matters.
        #
        # Subject and composition are more important than generic
        # quality language.
        # -----------------------------------------------------

        parts: list[str] = []

        # Visual type
        parts.append(
            visual_type
        )

        # Subject
        if subject:
            parts.append(
                subject
            )

        # Composition
        if composition:
            parts.append(
                composition
            )

        # Style
        if style:
            parts.append(
                style
            )

        # Required visual elements
        if must_show:

            parts.append(
                "show "
                + self._join_items(
                    must_show
                )
            )

        # Color palette
        if colors:

            parts.append(
                "colors "
                + self._join_items(
                    colors
                )
            )

        # Text policy
        if text_policy:

            normalized_text_policy = (
                text_policy.lower()
            )

            if (
                "no text"
                in normalized_text_policy
                or "without text"
                in normalized_text_policy
            ):

                parts.append(
                    "no text or labels"
                )

            else:

                parts.append(
                    text_policy
                )

        # Important negative constraints only.
        #
        # We intentionally don't append every generic constraint.
        # Long lists are harmful to prompt quality.
        if must_avoid:

            avoid_items = [
                item
                for item in must_avoid
                if item
            ]

            if avoid_items:

                parts.append(
                    "avoid "
                    + self._join_items(
                        avoid_items
                    )
                )

        # -----------------------------------------------------
        # Convert to a single prompt.
        # -----------------------------------------------------

        prompt = self._compact_text(
            ", ".join(parts)
        )

        # -----------------------------------------------------
        # If the prompt is still too long, progressively remove
        # lower-priority information.
        # -----------------------------------------------------

        if self._word_count(prompt) > self.MAX_FLUX_WORDS:

            parts = [
                visual_type,
                subject,
                composition,
                style,
            ]

            prompt = self._compact_text(
                ", ".join(
                    part
                    for part in parts
                    if part
                )
            )

        # -----------------------------------------------------
        # Second fallback:
        # subject + composition + style
        # -----------------------------------------------------

        if self._word_count(prompt) > self.MAX_FLUX_WORDS:

            parts = [
                subject,
                composition,
                style,
            ]

            prompt = self._compact_text(
                ", ".join(
                    part
                    for part in parts
                    if part
                )
            )

        # -----------------------------------------------------
        # Final fallback:
        # subject only.
        # -----------------------------------------------------

        if self._word_count(prompt) > self.MAX_FLUX_WORDS:

            prompt = self._compact_text(
                subject
            )

        return prompt

    # =========================================================
    # ICON PROMPT
    # =========================================================

    def _build_icon_prompt(
        self,
        description: str,
        visual_spec: dict[str, Any],
    ) -> str:
        """
        Build a structured specification for an icon generator.

        Icons should eventually be generated as SVG/vector assets
        rather than through FLUX.
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
        )[: self.MAX_COLORS]

        must_show = self._list(
            visual_spec,
            "must_show",
        )[: self.MAX_MUST_SHOW]

        must_avoid = self._list(
            visual_spec,
            "must_avoid",
        )[: self.MAX_MUST_AVOID]

        lines: list[str] = [
            "Clean vector icon specification.",
        ]

        if purpose:
            lines.append(
                f"Purpose: {purpose}"
            )

        lines.append(
            f"Subject: {subject}"
        )

        if composition:
            lines.append(
                f"Composition: {composition}"
            )

        if style:
            lines.append(
                f"Style: {style}"
            )

        if colors:
            lines.append(
                "Colors: "
                + self._join_items(colors)
            )

        if must_show:
            lines.append(
                "Required: "
                + self._join_items(must_show)
            )

        if must_avoid:
            lines.append(
                "Avoid: "
                + self._join_items(must_avoid)
            )

        lines.extend(
            [
                "Simple recognizable geometry.",
                "Consistent visual language.",
                "No photorealism.",
                "No unnecessary text.",
                "No watermark.",
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

        This is intentionally NOT a FLUX image prompt.

        The future diagram generator can convert this specification
        into SVG or editable PowerPoint elements.
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
        )[: self.MAX_COLORS]

        must_show = self._list(
            visual_spec,
            "must_show",
        )[: self.MAX_MUST_SHOW]

        must_avoid = self._list(
            visual_spec,
            "must_avoid",
        )[: self.MAX_MUST_AVOID]

        lines: list[str] = [
            "Clean educational vector diagram specification.",
        ]

        if purpose:
            lines.append(
                f"Purpose: {purpose}"
            )

        lines.append(
            f"Subject: {subject}"
        )

        if composition:
            lines.append(
                f"Composition: {composition}"
            )

        if style:
            lines.append(
                f"Style: {style}"
            )

        if colors:
            lines.append(
                "Colors: "
                + self._join_items(colors)
            )

        if must_show:
            lines.append(
                "Required: "
                + self._join_items(must_show)
            )

        if must_avoid:
            lines.append(
                "Avoid: "
                + self._join_items(must_avoid)
            )

        lines.extend(
            [
                "Use clear nodes, connectors and relationships.",
                "Use consistent spacing.",
                "Maintain clear visual hierarchy.",
                "Readable at slide size.",
                "Prefer vector geometry.",
                "Avoid decorative clutter.",
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
        Build a structured chart specification.

        Charts should eventually be generated using real data and
        rendered as editable/vector presentation elements rather
        than generated by FLUX.
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
        )[: self.MAX_COLORS]

        must_show = self._list(
            visual_spec,
            "must_show",
        )[: self.MAX_MUST_SHOW]

        must_avoid = self._list(
            visual_spec,
            "must_avoid",
        )[: self.MAX_MUST_AVOID]

        lines: list[str] = [
            "Presentation chart specification.",
        ]

        if purpose:
            lines.append(
                f"Purpose: {purpose}"
            )

        lines.append(
            f"Chart subject: {subject}"
        )

        if composition:
            lines.append(
                f"Composition: {composition}"
            )

        if style:
            lines.append(
                f"Style: {style}"
            )

        if colors:
            lines.append(
                "Colors: "
                + self._join_items(colors)
            )

        if must_show:
            lines.append(
                "Required: "
                + self._join_items(must_show)
            )

        if must_avoid:
            lines.append(
                "Avoid: "
                + self._join_items(must_avoid)
            )

        lines.extend(
            [
                "Use accurate data only.",
                "Do not fabricate numerical values.",
                "Use editable chart labels.",
                "Use clear axes.",
                "Use readable legends when needed.",
                "Avoid unnecessary decoration.",
            ]
        )

        return self._join_lines(
            lines
        )

    # =========================================================
    # TEXT HELPER
    # =========================================================

    @staticmethod
    def _text(
        data: dict[str, Any],
        key: str,
        fallback: str = "",
    ) -> str:
        """
        Safely extract a text field.
        """

        value = data.get(
            key,
            fallback,
        )

        if value is None:
            return fallback

        return str(value).strip()

    # =========================================================
    # LIST HELPER
    # =========================================================

    @staticmethod
    def _list(
        data: dict[str, Any],
        key: str,
    ) -> list[str]:
        """
        Safely extract a list field.

        Non-list values are converted into a single-item list.
        """

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
            value = [
                value
            ]

        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    # =========================================================
    # JOIN ITEMS
    # =========================================================

    @staticmethod
    def _join_items(
        items: list[str],
    ) -> str:
        """
        Join short visual-spec items compactly.
        """

        cleaned = [
            item.strip()
            for item in items
            if item
            and item.strip()
        ]

        if not cleaned:
            return ""

        return ", ".join(
            cleaned
        )

    # =========================================================
    # COMPACT TEXT
    # =========================================================

    @staticmethod
    def _compact_text(
        text: str,
    ) -> str:
        """
        Normalize whitespace so the prompt remains compact.
        """

        return " ".join(
            text.split()
        ).strip()

    # =========================================================
    # WORD COUNT
    # =========================================================

    @staticmethod
    def _word_count(
        text: str,
    ) -> int:
        """
        Approximate prompt length.

        This is deliberately conservative. CLIP uses tokenizer
        tokens rather than whitespace-separated words, so this
        count is only a safety heuristic.
        """

        if not text:
            return 0

        return len(
            text.split()
        )

    # =========================================================
    # JOIN LINES
    # =========================================================

    @staticmethod
    def _join_lines(
        lines: list[str],
    ) -> str:
        """
        Join non-empty lines.
        """

        return "\n".join(
            line.strip()
            for line in lines
            if line
            and line.strip()
        ).strip()