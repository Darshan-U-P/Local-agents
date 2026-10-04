from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO

from backend.planner.presentation_planner import PresentationPlanner


class UnsupportedClaimFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = PresentationPlanner.__new__(PresentationPlanner)
        self.planner.evidence_validation_results = []
        self.planner.unsupported_claim_removals = []

    def test_removes_unsupported_claim_and_preserves_other_points_and_slide(self):
        unsupported_claim = "Error correction remains a critical challenge"
        plan = {
            "slides": [
                {
                    "slide_number": 1,
                    "title": "Challenges",
                    "purpose": "Summarize the cited research",
                    "layout": "title_content",
                    "key_points": [
                        {
                            "text": "Qubits encode quantum states",
                            "sources": ["source-supported-fact-01"],
                        },
                        {
                            "text": unsupported_claim,
                            "sources": ["source-002-fact-05"],
                        },
                        {
                            "text": "A related claim remains weak",
                            "sources": ["source-weak-fact-01"],
                        },
                    ],
                    "assets": [{"type": "none"}],
                }
            ]
        }
        original_slide_structure = {
            key: value
            for key, value in plan["slides"][0].items()
            if key != "key_points"
        }
        self.planner.evidence_validation_results = [
            {
                "slide_index": 1,
                "point_index": 1,
                "claim": "Qubits encode quantum states",
                "status": "SUPPORTED",
                "original_citations": ["source-supported-fact-01"],
            },
            {
                "slide_index": 1,
                "point_index": 2,
                "claim": unsupported_claim,
                "status": "UNSUPPORTED",
                "original_citations": ["source-002-fact-05"],
            },
            {
                "slide_index": 1,
                "point_index": 3,
                "claim": "A related claim remains weak",
                "status": "WEAK_SUPPORT",
                "original_citations": ["source-weak-fact-01"],
            },
        ]

        output = StringIO()
        with redirect_stdout(output):
            removals = self.planner._remove_unsupported_key_points(plan)

        points = plan["slides"][0]["key_points"]
        self.assertEqual(
            [point["text"] for point in points],
            [
                "Qubits encode quantum states",
                "A related claim remains weak",
            ],
        )
        self.assertEqual(
            {
                key: value
                for key, value in plan["slides"][0].items()
                if key != "key_points"
            },
            original_slide_structure,
        )
        self.assertEqual(
            removals,
            [
                {
                    "slide_number": 1,
                    "point_number": 2,
                    "claim": unsupported_claim,
                    "original_citations": ["source-002-fact-05"],
                    "reason": "UNSUPPORTED",
                }
            ],
        )
        self.assertEqual(
            self.planner.unsupported_claim_removals,
            removals,
        )
        self.assertIn(unsupported_claim, output.getvalue())
        self.assertIn("slide 1, point 2", output.getvalue())
        self.assertIn("reason: UNSUPPORTED", output.getvalue())


if __name__ == "__main__":
    unittest.main()
