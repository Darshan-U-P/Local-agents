from __future__ import annotations

import json
import unittest

from backend.research.research_extractor import ResearchExtractor


class ResearchExtractorConceptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = ResearchExtractor()
        self.fact_text = (
            "Quantum computers use qubits to represent quantum states, "
            "while entanglement correlates multiple qubits."
        )

    def test_fact_id_and_text_are_preserved_with_explicit_concepts(self):
        research = self.extractor.extract(
            {
                "topic": "Quantum computing",
                "sources": [
                    {
                        "source_id": "source-004",
                        "title": "Quantum systems",
                        "url": "https://example.test/research",
                        "domain": "example.test",
                        "content": self.fact_text,
                    }
                ],
            }
        )

        fact = research["sources"][0]["facts"][0]

        self.assertEqual(
            fact["fact_id"],
            "source-004-fact-01",
        )
        self.assertEqual(fact["text"], self.fact_text)
        self.assertTrue(fact["concepts"])
        self.assertTrue(
            all(
                concept.casefold() in fact["text"].casefold()
                for concept in fact["concepts"]
            )
        )
        self.assertIn(
            "qubits",
            " ".join(fact["concepts"]).casefold(),
        )
        self.assertEqual(
            fact["concepts"],
            self.extractor._extract_concepts(fact["text"]),
        )
        self.assertNotIn(
            "quantum system",
            [concept.casefold() for concept in fact["concepts"]],
        )

    def test_compact_serialization_includes_fact_concepts(self):
        source = self.extractor._extract_source(
            {
                "source_id": "source-004",
                "content": self.fact_text,
            },
            fallback_source_id="source-001",
        )

        serialized_json = json.dumps(source)
        serialized_text = self.extractor._source_to_text(source)

        self.assertIn('"concepts"', serialized_json)
        self.assertIn("fact_id: source-004-fact-01", serialized_text)
        self.assertIn(f"text: {self.fact_text}", serialized_text)
        self.assertIn("concepts: [", serialized_text)
        for concept in source["facts"][0]["concepts"]:
            self.assertIn(concept, serialized_json)


if __name__ == "__main__":
    unittest.main()
