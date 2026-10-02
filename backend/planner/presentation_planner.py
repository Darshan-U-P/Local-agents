import json
import re

from backend.ai.model_manager import ModelManager


class PresentationPlanner:
    """
    Creates structured presentation plans using the local Qwen model.

    The planner is responsible for:
        1. Presentation structure
        2. Slide content
        3. Layout selection
        4. Asset requirements
        5. Detailed visual specifications

    The planner does NOT generate images.

    Example pipeline:

        User topic
            ↓
        Qwen3-4B
            ↓
        Presentation Plan
            ↓
        Asset Visual Specification
            ↓
        Asset Prompt Builder
            ↓
        FLUX / SVG / Chart / PPT shapes
    """

    ALLOWED_LAYOUTS = {
        "hero",
        "title_content",
        "two_column",
        "three_column",
        "image_left",
        "image_right",
        "big_stat",
        "timeline",
        "process",
        "comparison",
        "quote",
        "full_image",
        "diagram",
        "chart",
        "cards",
        "section_divider",
    }

    ALLOWED_ASSET_TYPES = {
        "image",
        "icon",
        "diagram",
        "chart",
        "illustration",
        "photo",
        "none",
    }

    def __init__(self):
        self.model_manager = ModelManager()

    # =========================================================
    # CREATE PRESENTATION PLAN
    # =========================================================

    def create_plan(
        self,
        topic: str,
        slide_count: int = 6,
    ) -> dict:

        if not topic or not topic.strip():
            raise ValueError(
                "Presentation topic cannot be empty."
            )

        if slide_count < 1:
            raise ValueError(
                "slide_count must be at least 1."
            )

        if slide_count > 30:
            raise ValueError(
                "slide_count cannot exceed 30."
            )

        # =====================================================
        # SYSTEM PROMPT
        # =====================================================

        system_prompt = """
You are a professional AI presentation planning system.

Your job is to transform a user's presentation topic into a
structured presentation plan that can later be rendered into
an editable PowerPoint presentation.

You are NOT generating the PowerPoint file.

You are NOT generating the actual images.

You are creating the structured content and visual requirements
that downstream systems will use.

============================================================
IMPORTANT OUTPUT RULES
============================================================

1. Return ONLY valid JSON.
2. Do NOT return Markdown.
3. Do NOT use ``` code fences.
4. Do NOT write explanations before or after the JSON.
5. Do NOT write <think> tags.
6. Generate exactly the requested number of slides.
7. Keep slide content concise and presentation-friendly.
8. Every slide must have a clear purpose.
9. Use only supported layouts.
10. Do not invent unnecessary statistics.
11. Do not fabricate precise numerical data.
12. Do not request an AI-generated image when a native
    PowerPoint element would be more appropriate.
13. Do not put important readable text inside an AI image.
14. AI image assets should generally contain NO text.
15. Charts should be represented as chart requirements,
    not as AI-generated images.
16. Diagrams should be represented as diagram requirements,
    not as AI-generated images.
17. Timelines, processes, comparisons, cards, and statistics
    should preferably use editable PowerPoint elements.
18. Assets must explain WHY the visual is needed.
19. Visual specifications must be detailed enough for a
    downstream asset generator to create the correct visual.
20. The final output must be directly parseable using
    Python json.loads().

============================================================
SUPPORTED SLIDE LAYOUTS
============================================================

hero
title_content
two_column
three_column
image_left
image_right
big_stat
timeline
process
comparison
quote
full_image
diagram
chart
cards
section_divider

============================================================
SUPPORTED ASSET TYPES
============================================================

image
icon
diagram
chart
illustration
photo
none

============================================================
ASSET TYPE RULES
============================================================

IMAGE:
Use for conceptual, scientific, technological, environmental,
architectural, or other visual imagery.

ILLUSTRATION:
Use for stylized conceptual visuals.

PHOTO:
Use when a realistic photographic appearance is appropriate.

ICON:
Use for small symbolic visual elements.

DIAGRAM:
Use when relationships, architecture, processes, components,
or concepts must be visually explained.

CHART:
Use when numerical or comparative data must be visualized.

NONE:
Use when the slide does not require a visual asset.

IMPORTANT:
Do NOT use IMAGE to represent:
- timelines
- charts
- process diagrams
- comparison tables
- statistics
- technical diagrams
when those elements can be rendered directly as editable
PowerPoint elements.

============================================================
VISUAL SPECIFICATION
============================================================

Every non-"none" asset must contain:

"visual_spec": {
    "purpose": "why this visual is needed",
    "subject": "main subject",
    "composition": "how the visual should be composed",
    "style": "visual style",
    "color_palette": [
        "color 1",
        "color 2"
    ],
    "must_show": [
        "important visual element"
    ],
    "must_avoid": [
        "unwanted element"
    ],
    "text_policy": "no text"
}

The visual specification must be specific.

BAD:

"subject": "quantum computer"

GOOD:

"subject": "a superconducting quantum processor inside a
cryogenic dilution refrigerator with visible control wiring
and layered metallic structures"

BAD:

"composition": "nice"

GOOD:

"composition": "center the quantum processor in the lower
middle of the frame, with cryogenic components surrounding it,
leaving clean negative space on the upper-left side for
presentation text"

============================================================
AI IMAGE RULES
============================================================

For image, illustration, and photo assets:

- Do not request text inside the image.
- Do not request labels.
- Do not request paragraphs.
- Do not request numerical annotations.
- Do not request logos.
- Do not request watermarks.
- Do not request readable UI screenshots unless explicitly
  required by the user.
- Prefer a clean composition suitable for presentation use.
- Specify the subject clearly.
- Specify the visual style.
- Specify lighting when relevant.
- Specify composition.
- Specify important visual elements.
- Specify unwanted elements.

============================================================
DIAGRAM RULES
============================================================

For diagrams:

Describe the conceptual relationships.

Example:

"visual_spec": {
    "purpose": "Explain how superposition and entanglement
    enable quantum information processing",
    "subject": "two qubits showing superposition and an
    entangled connection",
    "composition": "place two qubit nodes horizontally with
    a clear connection between them; show superposition as
    two possible state paths",
    "style": "clean educational scientific vector diagram",
    "color_palette": [
        "deep blue",
        "cyan",
        "white"
    ],
    "must_show": [
        "Qubit A",
        "Qubit B",
        "superposition states",
        "entanglement connection"
    ],
    "must_avoid": [
        "photorealism",
        "decorative clutter",
        "unnecessary text"
    ],
    "text_policy": "minimal labels only"
}

============================================================
CHART RULES
============================================================

For charts:

- Do not ask an image model to draw the chart.
- Describe the chart structure.
- Never invent precise data unless the user supplied it.
- If exact data is unavailable, describe the intended
  relationship without fabricating numbers.

Example:

"visual_spec": {
    "purpose": "Show how qubit stability changes as
    temperature increases",
    "subject": "qubit stability versus temperature",
    "composition": "single clean line chart with temperature
    on the horizontal axis and stability on the vertical axis",
    "style": "minimal scientific presentation chart",
    "color_palette": [
        "blue",
        "dark gray",
        "white"
    ],
    "must_show": [
        "temperature axis",
        "stability axis",
        "declining stability trend"
    ],
    "must_avoid": [
        "fabricated measurements",
        "unverified numerical values"
    ],
    "text_policy": "editable chart labels"
}

============================================================
ICON RULES
============================================================

Icons should represent the concept clearly and simply.

Example:

Cryptography:
- lock
- key
- secure digital connection

Biology:
- DNA
- molecule
- biological structure

Finance:
- graph
- financial network
- currency symbol

Avoid creating one large image containing multiple unrelated
icons when separate editable/vector icons are more appropriate.

============================================================
SLIDE CONTENT RULES
============================================================

Each slide should normally contain around 3 key points.

Key points must be:
- short
- understandable
- presentation-friendly
- directly related to the slide purpose

Avoid paragraphs.

============================================================
OUTPUT JSON STRUCTURE
============================================================

Return exactly this structure:

{
    "title": "string",
    "subtitle": "string",
    "slide_count": 6,
    "slides": [
        {
            "slide_number": 1,
            "title": "string",
            "purpose": "string",
            "layout": "hero",
            "key_points": [
                "string",
                "string",
                "string"
            ],
            "assets": [
                {
                    "type": "image",
                    "description": "short description of the asset",
                    "visual_spec": {
                        "purpose": "why this visual is needed",
                        "subject": "main visual subject",
                        "composition": "composition instructions",
                        "style": "visual style",
                        "color_palette": [
                            "color 1",
                            "color 2"
                        ],
                        "must_show": [
                            "important element"
                        ],
                        "must_avoid": [
                            "unwanted element"
                        ],
                        "text_policy": "no text"
                    }
                }
            ]
        }
    ]
}

For slides without a visual:

"assets": [
    {
        "type": "none",
        "description": "No visual asset required",
        "visual_spec": null
    }
]

Do not add fields outside this schema.
"""

        # =====================================================
        # USER PROMPT
        # =====================================================

        user_prompt = f"""
Create a {slide_count}-slide presentation about:

{topic}

Requirements:

- Exactly {slide_count} slides.
- Create a logical narrative from introduction to conclusion.
- Each slide must have a clear purpose.
- Choose the most appropriate layout for each slide.
- Keep key points short and presentation-friendly.
- Use approximately 3 key points per slide.
- Identify only useful visual assets.
- Every visual asset must have a detailed visual_spec.
- Make visual specifications concrete and actionable.
- Do not put important presentation text inside AI-generated
  images.
- Prefer editable PowerPoint elements for timelines, charts,
  processes, comparisons, and statistics.
- Use diagrams for conceptual relationships.
- Use icons for simple symbolic concepts.
- Use images/illustrations/photos for visual storytelling.
- Do not fabricate numerical data.
- Do not create unnecessary assets.
- Return JSON only.
"""

        # =====================================================
        # GENERATE PLAN
        # =====================================================

        response = self.model_manager.generate(
            prompt=user_prompt,
            max_tokens=4000,
            temperature=0.2,
            system_prompt=system_prompt,
        )

        return self._parse_json(response)

    # =========================================================
    # UNLOAD MODEL
    # =========================================================

    def unload(self):
        """
        Unload the Qwen model used by the presentation planner.

        The planner owns the ModelManager instance, so the
        planner is responsible for releasing it when planning
        is finished.
        """

        if self.model_manager is None:
            return

        print()
        print("================================")
        print("UNLOADING PRESENTATION PLANNER")
        print("================================")

        self.model_manager.unload()

        print("Presentation planner model released.")
        print("================================")

    # =========================================================
    # MODEL STATUS
    # =========================================================

    def is_loaded(self) -> bool:
        """
        Return True if the Qwen model is currently loaded.
        """

        if self.model_manager is None:
            return False

        return self.model_manager.is_loaded()

    # =========================================================
    # JSON PARSER
    # =========================================================

    def _parse_json(
        self,
        response: str,
    ) -> dict:

        if not response:
            raise ValueError(
                "Planner returned an empty response."
            )

        response = response.strip()

        # -----------------------------------------------------
        # Remove Qwen thinking output
        # -----------------------------------------------------

        if "<think>" in response:

            if "</think>" in response:

                response = response.split(
                    "</think>",
                    1,
                )[1].strip()

            else:

                response = response.split(
                    "<think>",
                    1,
                )[0].strip()

        # -----------------------------------------------------
        # Remove Markdown code fences
        # -----------------------------------------------------

        response = re.sub(
            r"^```(?:json)?\s*",
            "",
            response,
            flags=re.IGNORECASE,
        )

        response = re.sub(
            r"\s*```$",
            "",
            response,
        )

        response = response.strip()

        # -----------------------------------------------------
        # Extract outermost JSON object
        # -----------------------------------------------------

        if not response.startswith("{"):

            start = response.find("{")

            if start == -1:
                raise ValueError(
                    "Planner response does not contain "
                    "a JSON object."
                )

            response = response[start:]

        end = response.rfind("}")

        if end == -1:
            raise ValueError(
                "Planner response contains an incomplete "
                "JSON object."
            )

        response = response[:end + 1]

        # -----------------------------------------------------
        # Parse JSON
        # -----------------------------------------------------

        try:

            plan = json.loads(response)

        except json.JSONDecodeError as exc:

            raise ValueError(
                "Planner generated invalid JSON.\n\n"
                f"JSON error: {exc}\n\n"
                f"Model response:\n{response}"
            ) from exc

        # -----------------------------------------------------
        # Validate
        # -----------------------------------------------------

        self._validate_plan(plan)

        return plan

    # =========================================================
    # VALIDATE PLAN
    # =========================================================

    def _validate_plan(
        self,
        plan: dict,
    ):

        if not isinstance(plan, dict):
            raise ValueError(
                "Planner output must be a JSON object."
            )

        # -----------------------------------------------------
        # Top-level fields
        # -----------------------------------------------------

        required_fields = [
            "title",
            "subtitle",
            "slide_count",
            "slides",
        ]

        for field in required_fields:

            if field not in plan:
                raise ValueError(
                    f"Missing required field: {field}"
                )

        # -----------------------------------------------------
        # Title
        # -----------------------------------------------------

        if not isinstance(
            plan["title"],
            str,
        ):
            raise ValueError(
                "'title' must be a string."
            )

        # -----------------------------------------------------
        # Subtitle
        # -----------------------------------------------------

        if not isinstance(
            plan["subtitle"],
            str,
        ):
            raise ValueError(
                "'subtitle' must be a string."
            )

        # -----------------------------------------------------
        # Slide count
        # -----------------------------------------------------

        if not isinstance(
            plan["slide_count"],
            int,
        ):
            raise ValueError(
                "'slide_count' must be an integer."
            )

        # -----------------------------------------------------
        # Slides
        # -----------------------------------------------------

        if not isinstance(
            plan["slides"],
            list,
        ):
            raise ValueError(
                "'slides' must be a list."
            )

        if plan["slide_count"] != len(
            plan["slides"]
        ):
            raise ValueError(
                "slide_count does not match "
                "the number of slides."
            )

        # -----------------------------------------------------
        # Validate every slide
        # -----------------------------------------------------

        for index, slide in enumerate(
            plan["slides"],
            start=1,
        ):

            if not isinstance(
                slide,
                dict,
            ):
                raise ValueError(
                    f"Slide {index} must be a JSON object."
                )

            required_slide_fields = [
                "slide_number",
                "title",
                "purpose",
                "layout",
                "key_points",
                "assets",
            ]

            for field in required_slide_fields:

                if field not in slide:
                    raise ValueError(
                        f"Slide {index} is missing "
                        f"required field: {field}"
                    )

            # -------------------------------------------------
            # Slide number
            # -------------------------------------------------

            if slide["slide_number"] != index:

                raise ValueError(
                    f"Slide numbering error: "
                    f"expected {index}, "
                    f"got {slide['slide_number']}."
                )

            # -------------------------------------------------
            # Text fields
            # -------------------------------------------------

            for field in [
                "title",
                "purpose",
            ]:

                if not isinstance(
                    slide[field],
                    str,
                ):
                    raise ValueError(
                        f"Slide {index} '{field}' "
                        "must be a string."
                    )

            # -------------------------------------------------
            # Layout
            # -------------------------------------------------

            if slide["layout"] not in self.ALLOWED_LAYOUTS:

                raise ValueError(
                    f"Slide {index} uses unsupported "
                    f"layout: {slide['layout']}"
                )

            # -------------------------------------------------
            # Key points
            # -------------------------------------------------

            if not isinstance(
                slide["key_points"],
                list,
            ):
                raise ValueError(
                    f"Slide {index} 'key_points' "
                    "must be a list."
                )

            for point in slide["key_points"]:

                if not isinstance(
                    point,
                    str,
                ):
                    raise ValueError(
                        f"Slide {index} contains a "
                        "non-string key point."
                    )

            # -------------------------------------------------
            # Assets
            # -------------------------------------------------

            if not isinstance(
                slide["assets"],
                list,
            ):
                raise ValueError(
                    f"Slide {index} 'assets' "
                    "must be a list."
                )

            for asset in slide["assets"]:

                self._validate_asset(
                    slide_index=index,
                    asset=asset,
                )

    # =========================================================
    # VALIDATE ASSET
    # =========================================================

    def _validate_asset(
        self,
        slide_index: int,
        asset: dict,
    ):

        if not isinstance(
            asset,
            dict,
        ):
            raise ValueError(
                f"Slide {slide_index} contains "
                "an invalid asset."
            )

        # -----------------------------------------------------
        # Required asset fields
        # -----------------------------------------------------

        required_fields = [
            "type",
            "description",
            "visual_spec",
        ]

        for field in required_fields:

            if field not in asset:

                raise ValueError(
                    f"Slide {slide_index} asset "
                    f"is missing '{field}'."
                )

        # -----------------------------------------------------
        # Asset type
        # -----------------------------------------------------

        if asset["type"] not in self.ALLOWED_ASSET_TYPES:

            raise ValueError(
                f"Slide {slide_index} contains "
                f"unsupported asset type: "
                f"{asset['type']}"
            )

        # -----------------------------------------------------
        # Description
        # -----------------------------------------------------

        if not isinstance(
            asset["description"],
            str,
        ):
            raise ValueError(
                f"Slide {slide_index} asset "
                "description must be a string."
            )

        # -----------------------------------------------------
        # NONE asset
        # -----------------------------------------------------

        if asset["type"] == "none":

            if asset["visual_spec"] is not None:

                raise ValueError(
                    f"Slide {slide_index} asset "
                    "of type 'none' must have "
                    "'visual_spec': null."
                )

            return

        # -----------------------------------------------------
        # Visual specification
        # -----------------------------------------------------

        visual_spec = asset["visual_spec"]

        if not isinstance(
            visual_spec,
            dict,
        ):
            raise ValueError(
                f"Slide {slide_index} asset "
                "'visual_spec' must be an object."
            )

        required_visual_fields = [
            "purpose",
            "subject",
            "composition",
            "style",
            "color_palette",
            "must_show",
            "must_avoid",
            "text_policy",
        ]

        for field in required_visual_fields:

            if field not in visual_spec:

                raise ValueError(
                    f"Slide {slide_index} asset "
                    f"visual_spec is missing "
                    f"'{field}'."
                )

        # -----------------------------------------------------
        # Visual text fields
        # -----------------------------------------------------

        for field in [
            "purpose",
            "subject",
            "composition",
            "style",
            "text_policy",
        ]:

            if not isinstance(
                visual_spec[field],
                str,
            ):
                raise ValueError(
                    f"Slide {slide_index} asset "
                    f"visual_spec '{field}' "
                    "must be a string."
                )

        # -----------------------------------------------------
        # Color palette
        # -----------------------------------------------------

        if not isinstance(
            visual_spec["color_palette"],
            list,
        ):
            raise ValueError(
                f"Slide {slide_index} asset "
                "'color_palette' must be a list."
            )

        for color in visual_spec["color_palette"]:

            if not isinstance(
                color,
                str,
            ):
                raise ValueError(
                    f"Slide {slide_index} asset "
                    "contains a non-string color."
                )

        # -----------------------------------------------------
        # Must show
        # -----------------------------------------------------

        if not isinstance(
            visual_spec["must_show"],
            list,
        ):
            raise ValueError(
                f"Slide {slide_index} asset "
                "'must_show' must be a list."
            )

        for item in visual_spec["must_show"]:

            if not isinstance(
                item,
                str,
            ):
                raise ValueError(
                    f"Slide {slide_index} asset "
                    "'must_show' contains a "
                    "non-string value."
                )

        # -----------------------------------------------------
        # Must avoid
        # -----------------------------------------------------

        if not isinstance(
            visual_spec["must_avoid"],
            list,
        ):
            raise ValueError(
                f"Slide {slide_index} asset "
                "'must_avoid' must be a list."
            )

        for item in visual_spec["must_avoid"]:

            if not isinstance(
                item,
                str,
            ):
                raise ValueError(
                    f"Slide {slide_index} asset "
                    "'must_avoid' contains a "
                    "non-string value."
                )

    # =========================================================
    # PRINT PLAN
    # =========================================================

    def print_plan(
        self,
        plan: dict,
    ):

        print(
            "\n===== PRESENTATION PLAN =====\n"
        )

        print(
            f"Title: {plan['title']}"
        )

        print(
            f"Subtitle: {plan['subtitle']}"
        )

        print(
            f"Slides: {plan['slide_count']}"
        )

        for slide in plan["slides"]:

            print(
                f"\n--- Slide "
                f"{slide['slide_number']} ---"
            )

            print(
                f"Title: {slide['title']}"
            )

            print(
                f"Purpose: {slide['purpose']}"
            )

            print(
                f"Layout: {slide['layout']}"
            )

            print("Key Points:")

            for point in slide["key_points"]:

                print(
                    f"  - {point}"
                )

            print("Assets:")

            for asset in slide["assets"]:

                print(
                    f"  - [{asset['type']}] "
                    f"{asset['description']}"
                )

                visual_spec = asset.get(
                    "visual_spec"
                )

                if visual_spec:

                    print(
                        "    Visual purpose: "
                        f"{visual_spec['purpose']}"
                    )

                    print(
                        "    Subject: "
                        f"{visual_spec['subject']}"
                    )

                    print(
                        "    Composition: "
                        f"{visual_spec['composition']}"
                    )

                    print(
                        "    Style: "
                        f"{visual_spec['style']}"
                    )

                    print(
                        "    Colors: "
                        f"{', '.join(visual_spec['color_palette'])}"
                    )

                    print("    Must show:")

                    for item in visual_spec["must_show"]:

                        print(
                            f"      - {item}"
                        )

                    print("    Must avoid:")

                    for item in visual_spec["must_avoid"]:

                        print(
                            f"      - {item}"
                        )

                    print(
                        "    Text policy: "
                        f"{visual_spec['text_policy']}"
                    )