import json
import re

from backend.ai.model_manager import ModelManager


class PresentationPlanner:

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

    def __init__(self):
        self.model_manager = ModelManager()

    def create_plan(
        self,
        topic: str,
        slide_count: int = 6,
    ) -> dict:

        if not topic or not topic.strip():
            raise ValueError("Presentation topic cannot be empty.")

        if slide_count < 1:
            raise ValueError("slide_count must be at least 1.")

        if slide_count > 30:
            raise ValueError("slide_count cannot exceed 30.")

        system_prompt = """
You are a professional presentation planning AI.

Your task is to convert a user's presentation request into a
structured presentation plan.

IMPORTANT RULES:

1. Return ONLY valid JSON.
2. Do NOT return Markdown.
3. Do NOT use ``` code fences.
4. Do NOT write explanations before or after the JSON.
5. Do NOT write <think> tags.
6. Generate exactly the requested number of slides.
7. Keep slide content concise.
8. Do not invent unnecessary statistics.
9. Do not use unsupported layouts.
10. Every slide must have a clear purpose.
11. Assets should describe what visual content is needed.
12. The final output must be directly parseable by Python json.loads().

Use this exact JSON structure:

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
          "description": "string"
        }
      ]
    }
  ]
}

Allowed layouts:

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

Allowed asset types:

image
icon
diagram
chart
illustration
photo
none

Do not include fields that are not specified in the schema.
"""

        user_prompt = f"""
Create a {slide_count}-slide presentation about:

{topic}

Requirements:

- Exactly {slide_count} slides.
- Create a logical narrative from introduction to conclusion.
- Each slide must have a clear purpose.
- Choose the most appropriate layout for each slide.
- Keep each key point short and presentation-friendly.
- Use approximately 3 key points per slide.
- Identify useful visual assets.
- Do not create unnecessary assets.
- Return JSON only.
"""

        response = self.model_manager.generate(
            prompt=user_prompt,
            max_tokens=3000,
            temperature=0.2,
            system_prompt=system_prompt,
        )

        return self._parse_json(response)

    def _parse_json(self, response: str) -> dict:

        if not response:
            raise ValueError(
                "Planner returned an empty response."
            )

        response = response.strip()

        # ---------------------------------------------------------
        # Remove Qwen thinking output if it somehow survives
        # ModelManager's response cleaning.
        # ---------------------------------------------------------

        if "<think>" in response:

            if "</think>" in response:
                response = response.split(
                    "</think>",
                    1
                )[1].strip()

            else:
                response = response.split(
                    "<think>",
                    1
                )[0].strip()

        # ---------------------------------------------------------
        # Remove Markdown code fences
        # ---------------------------------------------------------

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

        # ---------------------------------------------------------
        # If the model added text before/after JSON, extract the
        # outermost JSON object.
        # ---------------------------------------------------------

        if not response.startswith("{"):

            start = response.find("{")

            if start == -1:
                raise ValueError(
                    "Planner response does not contain a JSON object."
                )

            response = response[start:]

        # Find the final closing brace.
        end = response.rfind("}")

        if end == -1:
            raise ValueError(
                "Planner response contains an incomplete JSON object."
            )

        response = response[:end + 1]

        # ---------------------------------------------------------
        # Parse JSON
        # ---------------------------------------------------------

        try:

            plan = json.loads(response)

        except json.JSONDecodeError as exc:

            raise ValueError(
                "Planner generated invalid JSON.\n\n"
                f"JSON error: {exc}\n\n"
                f"Model response:\n{response}"
            ) from exc

        # ---------------------------------------------------------
        # Validate structure
        # ---------------------------------------------------------

        self._validate_plan(plan)

        return plan

    def _validate_plan(self, plan: dict):

        if not isinstance(plan, dict):
            raise ValueError(
                "Planner output must be a JSON object."
            )

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

        # ---------------------------------------------------------
        # Validate top-level fields
        # ---------------------------------------------------------

        if not isinstance(plan["title"], str):
            raise ValueError(
                "'title' must be a string."
            )

        if not isinstance(plan["subtitle"], str):
            raise ValueError(
                "'subtitle' must be a string."
            )

        if not isinstance(plan["slide_count"], int):
            raise ValueError(
                "'slide_count' must be an integer."
            )

        if not isinstance(plan["slides"], list):
            raise ValueError(
                "'slides' must be a list."
            )

        if plan["slide_count"] != len(plan["slides"]):
            raise ValueError(
                "slide_count does not match the number of slides."
            )

        # ---------------------------------------------------------
        # Validate every slide
        # ---------------------------------------------------------

        for index, slide in enumerate(plan["slides"], start=1):

            if not isinstance(slide, dict):
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

            # Slide number
            if slide["slide_number"] != index:
                raise ValueError(
                    f"Slide numbering error: expected {index}, "
                    f"got {slide['slide_number']}."
                )

            # Text fields
            for field in [
                "title",
                "purpose",
            ]:

                if not isinstance(slide[field], str):
                    raise ValueError(
                        f"Slide {index} '{field}' "
                        "must be a string."
                    )

            # Layout
            if slide["layout"] not in self.ALLOWED_LAYOUTS:
                raise ValueError(
                    f"Slide {index} uses unsupported layout: "
                    f"{slide['layout']}"
                )

            # Key points
            if not isinstance(slide["key_points"], list):
                raise ValueError(
                    f"Slide {index} 'key_points' "
                    "must be a list."
                )

            for point in slide["key_points"]:

                if not isinstance(point, str):
                    raise ValueError(
                        f"Slide {index} contains a "
                        "non-string key point."
                    )

            # Assets
            if not isinstance(slide["assets"], list):
                raise ValueError(
                    f"Slide {index} 'assets' must be a list."
                )

            for asset in slide["assets"]:

                if not isinstance(asset, dict):
                    raise ValueError(
                        f"Slide {index} contains "
                        "an invalid asset."
                    )

                if "type" not in asset:
                    raise ValueError(
                        f"Slide {index} asset is missing 'type'."
                    )

                if "description" not in asset:
                    raise ValueError(
                        f"Slide {index} asset is missing "
                        "'description'."
                    )

                allowed_asset_types = {
                    "image",
                    "icon",
                    "diagram",
                    "chart",
                    "illustration",
                    "photo",
                    "none",
                }

                if asset["type"] not in allowed_asset_types:
                    raise ValueError(
                        f"Slide {index} contains unsupported "
                        f"asset type: {asset['type']}"
                    )

                if not isinstance(asset["description"], str):
                    raise ValueError(
                        f"Slide {index} asset description "
                        "must be a string."
                    )

    def print_plan(self, plan: dict):

        print("\n===== PRESENTATION PLAN =====\n")

        print(f"Title: {plan['title']}")
        print(f"Subtitle: {plan['subtitle']}")
        print(f"Slides: {plan['slide_count']}")

        for slide in plan["slides"]:

            print(
                f"\n--- Slide {slide['slide_number']} ---"
            )

            print(f"Title: {slide['title']}")
            print(f"Purpose: {slide['purpose']}")
            print(f"Layout: {slide['layout']}")

            print("Key Points:")

            for point in slide["key_points"]:
                print(f"  - {point}")

            print("Assets:")

            for asset in slide["assets"]:
                print(
                    f"  - [{asset['type']}] "
                    f"{asset['description']}"
                )