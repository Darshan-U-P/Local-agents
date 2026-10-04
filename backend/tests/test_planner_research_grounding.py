from __future__ import annotations

import json
import unittest

from backend.planner.presentation_planner import PresentationPlanner


class CapturingModelManager:
    def __init__(self, response: str) -> None:
        self.response = response
        self.arguments: dict = {}

    def generate(self, **arguments) -> str:
        self.arguments = arguments
        return self.response


class PlannerResearchGroundingTests(unittest.TestCase):
    def test_future_target_does_not_license_unsolved_problem_claim(self):
        fact_id = "source-002-fact-05"
        fact_text = (
            "The company’s longer-term goal is to build a system with "
            "2,000 logical qubits capable of running 1 billion gates "
            "by 2033."
        )
        safe_plan = {
            "title": "Quantum Computing",
            "subtitle": "A research-grounded overview",
            "slide_count": 1,
            "slides": [
                {
                    "slide_number": 1,
                    "title": "A future system target",
                    "purpose": "Summarize the stated system target",
                    "layout": "title_content",
                    "key_points": [
                        {
                            "text": (
                                "The company targets 2,000 logical "
                                "qubits by 2033"
                            ),
                            "sources": [fact_id],
                        }
                    ],
                    "assets": [],
                }
            ],
        }
        model_manager = CapturingModelManager(
            json.dumps(safe_plan)
        )
        planner = PresentationPlanner.__new__(PresentationPlanner)
        planner.model_manager = model_manager
        planner.evidence_validation_results = []

        planner.create_plan(
            topic="Quantum computing",
            slide_count=1,
            research_context={
                "sources": [
                    {
                        "source_id": "source-002",
                        "facts": [
                            {
                                "fact_id": fact_id,
                                "text": fact_text,
                            }
                        ],
                    }
                ]
            },
        )

        prompts = (
            model_manager.arguments["system_prompt"]
            + "\n"
            + model_manager.arguments["prompt"]
        )
        self.assertIn(fact_id, prompts)
        self.assertIn(fact_text, prompts)
        self.assertIn(
            "Never infer a general problem or limitation from a future target",
            prompts,
        )
        self.assertIn(
            '"error correction remains unsolved"',
            prompts,
        )
        self.assertIn(
            '"scalability remains unsolved"',
            prompts,
        )
        self.assertIn(
            '"decoherence is the main challenge"',
            prompts,
        )
        self.assertIn(
            "unless the research explicitly states those things",
            prompts,
        )
        self.assertIn(
            "Do not use pretrained knowledge to fill evidence gaps.",
            prompts,
        )
        self.assertNotIn(
            "Quantum error correction remains unsolved",
            planner.model_manager.arguments["prompt"],
        )


if __name__ == "__main__":
    unittest.main()
