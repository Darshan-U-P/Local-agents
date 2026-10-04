from __future__ import annotations

import unittest

from backend.planner.presentation_planner import PresentationPlanner


class EvidenceConceptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = PresentationPlanner.__new__(PresentationPlanner)

    def evaluate(self, claim: str, fact_text: str, concepts=None) -> dict:
        fact = {"fact_text": fact_text}
        if concepts is not None:
            fact["concepts"] = concepts
        return self.planner._evaluate_claim_against_fact(
            claim=claim,
            fact=fact,
        )

    def test_exact_concept_overlap_scores_full_coverage(self):
        result = self.evaluate(
            "quantum entanglement",
            "Unrelated evidence sentence.",
            ["quantum entanglement"],
        )

        self.assertEqual(result["concept_overlap_score"], 1.0)
        self.assertEqual(result["status"], "UNSUPPORTED")

    def test_partial_concept_overlap_scores_part_of_claim(self):
        result = self.evaluate(
            "quantum entanglement processing",
            "Unrelated evidence sentence.",
            ["quantum entanglement"],
        )

        self.assertAlmostEqual(
            result["concept_overlap_score"],
            2 / 3,
        )

    def test_unrelated_concepts_have_zero_overlap(self):
        result = self.evaluate(
            "quantum entanglement",
            "Unrelated evidence sentence.",
            ["classical computer"],
        )

        self.assertEqual(result["concept_overlap_score"], 0.0)

    def test_missing_concepts_preserves_text_assessment(self):
        claim = "Qubits represent quantum states."
        fact_text = "Qubits represent quantum states."

        without_concepts = self.evaluate(claim, fact_text)
        with_empty_concepts = self.evaluate(claim, fact_text, [])

        self.assertEqual(
            without_concepts["status"],
            with_empty_concepts["status"],
        )
        self.assertEqual(
            without_concepts["claim_term_coverage"],
            with_empty_concepts["claim_term_coverage"],
        )
        self.assertEqual(
            without_concepts["term_similarity"],
            with_empty_concepts["term_similarity"],
        )
        self.assertEqual(
            without_concepts["concept_overlap_score"],
            0.0,
        )

    def test_concepts_break_similar_lexical_candidate_ranking(self):
        claim = "Qubits encode quantum states."
        plan = {
            "slides": [
                {
                    "key_points": [
                        {
                            "text": claim,
                            "sources": ["source-wrong-fact-01"],
                        }
                    ]
                }
            ]
        }
        research = {
            "sources": [
                {
                    "facts": [
                        {
                            "fact_id": "source-wrong-fact-01",
                            "text": "Classical systems model physical states.",
                        },
                        {
                            "fact_id": "source-candidate-a-fact-01",
                            "text": "Qubits encode quantum states.",
                            "concepts": ["quantum"],
                        },
                        {
                            "fact_id": "source-candidate-b-fact-01",
                            "text": "Qubits encode quantum states.",
                            "concepts": [
                                "qubits",
                                "encode",
                                "quantum states",
                            ],
                        },
                    ]
                }
            ]
        }

        self.planner._validate_evidence(plan, research)

        self.assertEqual(
            plan["slides"][0]["key_points"][0]["sources"],
            ["source-candidate-b-fact-01"],
        )
        self.assertEqual(
            self.planner.evidence_validation_results[0]["citations"][0][
                "concept_overlap_score"
            ],
            1.0,
        )

    def test_repair_thresholds_still_block_weak_or_non_improving_candidates(self):
        self.assertEqual(self.planner.MIN_REPAIR_COVERAGE, 0.60)
        self.assertEqual(self.planner.MIN_REPAIR_SCORE, 0.55)
        self.assertEqual(self.planner.MIN_REPAIR_MARGIN, 0.10)

        low_coverage_plan = {
            "slides": [
                {
                    "key_points": [
                        {
                            "text": "Qubits enable parallel quantum computation",
                            "sources": ["source-wrong-fact-01"],
                        }
                    ]
                }
            ]
        }
        low_coverage_research = {
            "sources": [
                {
                    "facts": [
                        {
                            "fact_id": "source-wrong-fact-01",
                            "text": "Classical systems model physical states.",
                        },
                        {
                            "fact_id": "source-low-coverage-fact-01",
                            "text": "Qubits may process parallel information.",
                            "concepts": [
                                "qubits",
                                "parallel quantum computation",
                            ],
                        },
                    ]
                }
            ]
        }
        self.planner._validate_evidence(
            low_coverage_plan,
            low_coverage_research,
        )
        self.assertEqual(
            low_coverage_plan["slides"][0]["key_points"][0]["sources"],
            ["source-wrong-fact-01"],
        )
        self.assertIn(
            "below_coverage",
            {
                item["reason"]
                for item in self.planner.evidence_validation_results[0][
                    "rejected_candidates"
                ]
            },
        )

        low_score_plan = {
            "slides": [
                {
                    "key_points": [
                        {
                            "text": "Qubits encode parallel quantum states",
                            "sources": ["source-wrong-fact-02"],
                        }
                    ]
                }
            ]
        }
        low_score_research = {
            "sources": [
                {
                    "facts": [
                        {
                            "fact_id": "source-wrong-fact-02",
                            "text": "Classical systems model physical states.",
                        },
                        {
                            "fact_id": "source-low-score-fact-01",
                            "text": (
                                "Qubits encode parallel information across "
                                "hardware architectures and physical "
                                "implementations."
                            ),
                            "concepts": [
                                "qubits",
                                "encode",
                                "parallel",
                                "quantum states",
                            ],
                        },
                    ]
                }
            ]
        }
        self.planner._validate_evidence(
            low_score_plan,
            low_score_research,
        )
        self.assertEqual(
            low_score_plan["slides"][0]["key_points"][0]["sources"],
            ["source-wrong-fact-02"],
        )
        self.assertIn(
            "below_score",
            {
                item["reason"]
                for item in self.planner.evidence_validation_results[0][
                    "rejected_candidates"
                ]
            },
        )

        no_margin_plan = {
            "slides": [
                {
                    "key_points": [
                        {
                            "text": "Qubits represent quantum states.",
                            "sources": ["source-current-fact-01"],
                        }
                    ]
                }
            ]
        }
        no_margin_research = {
            "sources": [
                {
                    "facts": [
                        {
                            "fact_id": "source-current-fact-01",
                            "text": "Qubits represent quantum states.",
                        },
                        {
                            "fact_id": "source-candidate-fact-01",
                            "text": "Qubits represent quantum states.",
                            "concepts": [
                                "qubits",
                                "quantum states",
                            ],
                        },
                    ]
                }
            ]
        }
        self.planner._validate_evidence(
            no_margin_plan,
            no_margin_research,
        )
        self.assertEqual(
            no_margin_plan["slides"][0]["key_points"][0]["sources"],
            ["source-current-fact-01"],
        )
        self.assertIn(
            "insufficient_margin",
            {
                item["reason"]
                for item in self.planner.evidence_validation_results[0][
                    "rejected_candidates"
                ]
            },
        )


if __name__ == "__main__":
    unittest.main()
