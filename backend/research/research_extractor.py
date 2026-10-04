from __future__ import annotations

import re
from typing import Any


class ResearchExtractor:
    """
    Converts large raw web research results into compact,
    source-grounded research suitable for the Qwen presentation planner.

    Design goals:
    - Reduce research context aggressively.
    - Preserve source identity.
    - Preserve stable fact IDs.
    - Keep factual evidence useful for presentation planning.
    - Avoid webpage boilerplate and duplicate facts.
    - Remain dependency-free.
    """

    # ------------------------------------------------------------------
    # CONFIGURATION
    # ------------------------------------------------------------------

    NOISE_PATTERNS = (
        "cookie",
        "privacy policy",
        "terms of service",
        "sign in",
        "log in",
        "subscribe",
        "newsletter",
        "advertisement",
        "advertising",
        "copyright",
        "all rights reserved",
        "accept cookies",
        "manage cookies",
        "skip to content",
        "table of contents",
        "share this",
        "follow us",
        "related articles",
    )

    SENTENCE_PATTERN = re.compile(
        r"(?<=[.!?])\s+"
    )

    FACT_KEYWORDS = (
        "is",
        "are",
        "was",
        "were",
        "means",
        "defined",
        "refers",
        "uses",
        "used",
        "consists",
        "contains",
        "includes",
        "allows",
        "enables",
        "requires",
        "based on",
        "developed",
        "introduced",
        "measured",
        "called",
        "known as",
        "such as",
        "for example",
        "%",
        "million",
        "billion",
        "thousand",
        "year",
        "years",
        "2020",
        "2021",
        "2022",
        "2023",
        "2024",
        "2025",
        "2026",
    )

    CONCEPT_STOPWORDS = frozenset(
        """
        a an and are as at be been being by can could did do does for
        from had has have in into is it its may might must of on or
        should that the their them they this those through to was were
        will with would about after also based become called cause causes
        describe describes develop developed enable enables explain
        explains include includes included involving involve known make
        made mean means measure measured occur occurs provide provides
        represent represents require requires result results show shows
        use used uses using allow allows allowed allowing correlate
        correlates correlated correlating explain enable
        """.split()
    )
    CONCEPT_CONNECTORS = frozenset(
        "a an and as at by for from in into of on the through to with".split()
    )
    CONCEPT_WORD_PATTERN = re.compile(
        r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*"
    )
    MAX_CONCEPTS_PER_FACT = 8

    # ------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------

    def __init__(
        self,
        max_chars_per_source: int = 900,
        max_facts_per_source: int = 5,
        max_total_chars: int = 5200,
    ):
        """
        Configure the research compression budget.

        Defaults are intentionally conservative because the compact
        research is inserted directly into the Qwen planner prompt.

        Approximate target:

            6 sources
            × up to ~900 chars
            capped at 5.2K compact research

        This leaves substantially more room for the planner's
        system prompt and JSON output inside an 8192-token context.
        """

        self.max_chars_per_source = max_chars_per_source
        self.max_facts_per_source = max_facts_per_source
        self.max_total_chars = max_total_chars

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def extract(
        self,
        research: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Compress a ResearchManager result while preserving provenance.

        Expected input:

        {
            "topic": "...",
            "source_count": 6,
            "sources": [
                {
                    "source_id": "source-001",
                    "title": "...",
                    "url": "...",
                    "domain": "...",
                    "snippet": "...",
                    "content": "..."
                }
            ]
        }

        Output:

        {
            "topic": "...",
            "source_count": 6,
            "raw_character_count": 123456,
            "compact_character_count": 7000,
            "sources": [
                {
                    "source_id": "source-001",
                    "title": "...",
                    "url": "...",
                    "domain": "...",
                    "facts": [
                        {
                            "fact_id": "source-001-fact-01",
                            "text": "..."
                            "concepts": ["..."]
                        }
                    ]
                }
            ]
        }

        The fact_id remains stable and allows the planner to
        attach source attribution to presentation claims.
        """

        if not isinstance(research, dict):
            raise TypeError(
                "research must be a dictionary"
            )

        topic = str(
            research.get("topic", "")
        ).strip()

        raw_sources = research.get(
            "sources",
            [],
        )

        if not isinstance(raw_sources, list):
            raise ValueError(
                "research['sources'] must be a list"
            )

        compact_sources: list[dict[str, Any]] = []

        raw_characters = 0
        compact_characters = 0

        # --------------------------------------------------------------
        # PROCESS EACH SOURCE
        # --------------------------------------------------------------

        for index, source in enumerate(
            raw_sources,
            start=1,
        ):
            if not isinstance(source, dict):
                continue

            compact_source = self._extract_source(
                source=source,
                fallback_source_id=f"source-{index:03d}",
            )

            raw_content = str(
                source.get("content", "")
            )

            raw_characters += len(raw_content)

            source_text = self._source_to_text(
                compact_source
            )

            # ----------------------------------------------------------
            # RESPECT TOTAL CONTEXT BUDGET
            # ----------------------------------------------------------

            remaining = (
                self.max_total_chars
                - compact_characters
            )

            if remaining <= 0:
                break

            # ----------------------------------------------------------
            # TRUNCATE SOURCE IF NECESSARY
            # ----------------------------------------------------------

            if len(source_text) > remaining:

                compact_source["facts"] = (
                    self._truncate_facts(
                        compact_source.get(
                            "facts",
                            [],
                        ),
                        remaining_chars=remaining,
                    )
                )

                source_text = self._source_to_text(
                    compact_source
                )

            if not source_text:
                continue

            compact_sources.append(
                compact_source
            )

            compact_characters += len(
                source_text
            )

        # --------------------------------------------------------------
        # FINAL RESULT
        # --------------------------------------------------------------

        result = {
            "topic": topic,
            "source_count": len(
                compact_sources
            ),
            "raw_character_count": raw_characters,
            "compact_character_count": compact_characters,
            "sources": compact_sources,
        }

        return result

    # ------------------------------------------------------------------
    # SOURCE EXTRACTION
    # ------------------------------------------------------------------

    def _extract_source(
        self,
        source: dict[str, Any],
        fallback_source_id: str,
    ) -> dict[str, Any]:
        """
        Extract one compact source while preserving source identity.
        """

        source_id = str(
            source.get("source_id")
            or fallback_source_id
        ).strip()

        title = self._clean_text(
            str(
                source.get(
                    "title",
                    "",
                )
            )
        )

        url = str(
            source.get(
                "url",
                "",
            )
        ).strip()

        domain = str(
            source.get(
                "domain",
                "",
            )
        ).strip()

        snippet = self._clean_text(
            str(
                source.get(
                    "snippet",
                    "",
                )
            )
        )

        content = self._clean_text(
            str(
                source.get(
                    "content",
                    "",
                )
            )
        )

        # --------------------------------------------------------------
        # COMBINE SNIPPET + PAGE CONTENT
        # --------------------------------------------------------------

        combined_text = "\n".join(
            part
            for part in (
                snippet,
                content,
            )
            if part
        )

        # --------------------------------------------------------------
        # SENTENCE PROCESSING
        # --------------------------------------------------------------

        sentences = self._split_sentences(
            combined_text
        )

        sentences = self._remove_noise(
            sentences
        )

        sentences = self._deduplicate(
            sentences
        )

        scored_sentences = (
            self._score_sentences(
                sentences
            )
        )

        selected = [
            sentence
            for sentence, _score in scored_sentences[
                : self.max_facts_per_source
            ]
        ]

        # --------------------------------------------------------------
        # KEEP ORIGINAL SNIPPET IF USEFUL
        # --------------------------------------------------------------

        if (
            snippet
            and snippet not in selected
            and len(selected) < self.max_facts_per_source
        ):
            selected.insert(
                0,
                snippet
            )

        selected = self._deduplicate(
            selected
        )

        selected = selected[
            : self.max_facts_per_source
        ]

        # --------------------------------------------------------------
        # APPLY PER-SOURCE CHARACTER BUDGET
        # --------------------------------------------------------------

        selected = self._fit_sentences_to_budget(
            selected,
            max_chars=self.max_chars_per_source,
        )

        # --------------------------------------------------------------
        # ASSIGN STABLE FACT IDS
        # --------------------------------------------------------------

        facts: list[dict[str, Any]] = []

        for fact_index, text in enumerate(
            selected,
            start=1,
        ):
            facts.append(
                {
                    "fact_id": (
                        f"{source_id}"
                        f"-fact-{fact_index:02d}"
                    ),
                    "text": text,
                    "concepts": self._extract_concepts(text),
                }
            )

        # --------------------------------------------------------------
        # SOURCE OBJECT
        # --------------------------------------------------------------

        return {
            "source_id": source_id,
            "title": title,
            "url": url,
            "domain": domain,
            "facts": facts,
        }

    # ------------------------------------------------------------------
    # TEXT PROCESSING
    # ------------------------------------------------------------------

    def _clean_text(
        self,
        text: str,
    ) -> str:
        """
        Normalize whitespace and remove obvious webpage artifacts.
        """

        if not text:
            return ""

        # Normalize common whitespace.
        text = text.replace(
            "\xa0",
            " ",
        )

        text = text.replace(
            "\r",
            " ",
        )

        text = text.replace(
            "\n",
            " ",
        )

        # Collapse whitespace.
        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        # Remove repeated separators.
        text = re.sub(
            r"\s*[|•·]\s*",
            " ",
            text,
        )

        return text.strip()

    def _split_sentences(
        self,
        text: str,
    ) -> list[str]:
        """
        Split text into reasonably sized sentences.
        """

        if not text:
            return []

        parts = self.SENTENCE_PATTERN.split(
            text
        )

        cleaned: list[str] = []

        for part in parts:

            part = part.strip()

            if not part:
                continue

            # Ignore extremely short fragments.
            if len(part) < 30:
                continue

            # Prevent giant paragraphs from dominating
            # the context.
            if len(part) > 700:

                chunks = (
                    self._split_long_sentence(
                        part
                    )
                )

                cleaned.extend(
                    chunks
                )

            else:
                cleaned.append(
                    part
                )

        return cleaned

    def _split_long_sentence(
        self,
        text: str,
    ) -> list[str]:
        """
        Split oversized blocks on commas,
        semicolons and colons.
        """

        pieces = re.split(
            r"(?<=[,;:])\s+",
            text,
        )

        result: list[str] = []

        current = ""

        for piece in pieces:

            piece = piece.strip()

            if not piece:
                continue

            if (
                len(current)
                + len(piece)
                <= 500
            ):
                current = (
                    f"{current} {piece}"
                    .strip()
                )

            else:

                if len(current) >= 30:
                    result.append(
                        current
                    )

                current = piece

        if len(current) >= 30:
            result.append(
                current
            )

        return result

    # ------------------------------------------------------------------
    # NOISE FILTERING
    # ------------------------------------------------------------------

    def _remove_noise(
        self,
        sentences: list[str],
    ) -> list[str]:

        result: list[str] = []

        for sentence in sentences:

            lower = sentence.lower()

            # Remove obvious webpage boilerplate.
            if any(
                pattern in lower
                for pattern in self.NOISE_PATTERNS
            ):
                continue

            # Remove URLs-only fragments.
            if re.fullmatch(
                r"https?://\S+",
                sentence,
            ):
                continue

            # Remove navigation-like fragments.
            words = sentence.split()

            if len(words) <= 4:
                continue

            result.append(
                sentence
            )

        return result

    # ------------------------------------------------------------------
    # DEDUPLICATION
    # ------------------------------------------------------------------

    def _normalize_for_comparison(
        self,
        text: str,
    ) -> str:

        text = text.lower()

        text = re.sub(
            r"[^a-z0-9\s]",
            "",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    def _deduplicate(
        self,
        sentences: list[str],
    ) -> list[str]:

        seen: set[str] = set()

        result: list[str] = []

        for sentence in sentences:

            normalized = (
                self._normalize_for_comparison(
                    sentence
                )
            )

            if not normalized:
                continue

            # Exact duplicate.
            if normalized in seen:
                continue

            # Detect very similar repeated sentences.
            if self._is_near_duplicate(
                normalized,
                seen,
            ):
                continue

            seen.add(
                normalized
            )

            result.append(
                sentence
            )

        return result

    def _is_near_duplicate(
        self,
        text: str,
        seen: set[str],
    ) -> bool:

        words = set(
            text.split()
        )

        if len(words) < 8:
            return False

        for existing in seen:

            existing_words = set(
                existing.split()
            )

            if len(existing_words) < 8:
                continue

            intersection = (
                words & existing_words
            )

            similarity = (
                len(intersection)
                / max(
                    len(
                        words
                        | existing_words
                    ),
                    1,
                )
            )

            if similarity >= 0.82:
                return True

        return False

    # ------------------------------------------------------------------
    # SENTENCE SCORING
    # ------------------------------------------------------------------

    def _score_sentences(
        self,
        sentences: list[str],
    ) -> list[tuple[str, float]]:

        scored: list[
            tuple[str, float]
        ] = []

        for sentence in sentences:

            lower = sentence.lower()

            score = 0.0

            # ----------------------------------------------------------
            # FACTUAL LANGUAGE
            # ----------------------------------------------------------

            for keyword in self.FACT_KEYWORDS:

                if keyword in lower:
                    score += 1.0

            # ----------------------------------------------------------
            # NUMBERS
            # ----------------------------------------------------------

            if re.search(
                r"\b\d+(?:\.\d+)?\b",
                sentence,
            ):
                score += 2.0

            # ----------------------------------------------------------
            # TECHNICAL EXPLANATION
            # ----------------------------------------------------------

            if any(
                marker in lower
                for marker in (
                    "because",
                    "therefore",
                    "how",
                    "why",
                    "process",
                    "method",
                    "algorithm",
                    "system",
                    "technology",
                    "principle",
                    "advantage",
                    "limitation",
                    "application",
                    "example",
                )
            ):
                score += 1.5

            # ----------------------------------------------------------
            # USEFUL SENTENCE LENGTH
            # ----------------------------------------------------------

            length = len(
                sentence
            )

            if 80 <= length <= 350:
                score += 1.5

            elif length < 50:
                score -= 1.0

            elif length > 500:
                score -= 0.5

            scored.append(
                (
                    sentence,
                    score,
                )
            )

        # Highest scoring first.
        scored.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return scored

    # ------------------------------------------------------------------
    # BUDGET CONTROL
    # ------------------------------------------------------------------

    def _fit_sentences_to_budget(
        self,
        sentences: list[str],
        max_chars: int,
    ) -> list[str]:
        """
        Keep the highest-priority selected sentences inside
        the per-source character budget.

        We preserve complete sentences rather than blindly
        cutting text in the middle.
        """

        if max_chars <= 0:
            return []

        result: list[str] = []
        used = 0

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            cost = len(sentence)

            if result:
                cost += 1

            if used + cost > max_chars:

                # If nothing fits yet, allow a shortened version
                # so the source still has evidence.
                if not result:

                    available = max_chars

                    if available >= 80:

                        shortened = (
                            sentence[:available]
                            .rsplit(" ", 1)[0]
                            .strip()
                        )

                        if len(shortened) >= 50:

                            result.append(
                                shortened
                            )

                break

            result.append(
                sentence
            )

            used += cost

        return result

    # ------------------------------------------------------------------
    # OUTPUT HELPERS
    # ------------------------------------------------------------------

    def _source_to_text(
        self,
        source: dict[str, Any],
    ) -> str:
        """
        Convert a compact source into text for
        context-size accounting.

        Fact objects are converted into labeled fact_id, text, and
        concepts fields for compact-size accounting.
        """

        parts: list[str] = [
            str(
                source.get(
                    "source_id",
                    "",
                )
            ),
            str(
                source.get(
                    "title",
                    "",
                )
            ),
            str(
                source.get(
                    "url",
                    "",
                )
            ),
            str(
                source.get(
                    "domain",
                    "",
                )
            ),
        ]

        for fact in source.get(
            "facts",
            [],
        ):

            if isinstance(
                fact,
                dict,
            ):

                fact_id = str(
                    fact.get(
                        "fact_id",
                        "",
                    )
                ).strip()

                fact_text = str(
                    fact.get(
                        "text",
                        "",
                    )
                ).strip()
                concepts = fact.get("concepts", [])
                if not isinstance(concepts, list):
                    concepts = []

                if fact_id:
                    parts.append(
                        "\n".join(
                            (
                                f"fact_id: {fact_id}",
                                f"text: {fact_text}",
                                "concepts: ["
                                + ", ".join(
                                    str(concept)
                                    for concept in concepts
                                )
                                + "]",
                            )
                        )
                    )

                elif fact_text:
                    parts.append(
                        "\n".join(
                            (
                                f"text: {fact_text}",
                                "concepts: ["
                                + ", ".join(
                                    str(concept)
                                    for concept in concepts
                                )
                                + "]",
                            )
                        )
                    )

            else:

                # Backward compatibility.
                parts.append(
                    str(fact)
                )

        return "\n".join(
            part
            for part in parts
            if part
        )

    def _truncate_facts(
        self,
        facts: list[Any],
        remaining_chars: int,
    ) -> list[Any]:
        """
        Truncate facts while preserving
        fact_id + text structure.
        """

        result: list[Any] = []

        used = 0

        for fact in facts:

            if isinstance(
                fact,
                dict,
            ):

                fact_id = str(
                    fact.get(
                        "fact_id",
                        "",
                    )
                ).strip()

                fact_text = str(
                    fact.get(
                        "text",
                        "",
                    )
                ).strip()

                concepts = fact.get(
                    "concepts",
                    [],
                )
                if not isinstance(concepts, list):
                    concepts = []

                formatted = self._format_fact(
                    fact_id=fact_id,
                    fact_text=fact_text,
                    concepts=concepts,
                )

                cost = (
                    len(formatted)
                    + 1
                )

                if (
                    used + cost
                    > remaining_chars
                ):
                    break

                result.append(
                    {
                        "fact_id": fact_id,
                        "text": fact_text,
                        "concepts": list(concepts),
                    }
                )

                used += cost

            else:

                fact_text = str(
                    fact
                ).strip()

                cost = (
                    len(fact_text)
                    + 1
                )

                if (
                    used + cost
                    > remaining_chars
                ):
                    break

                result.append(
                    fact_text
                )

                used += cost

        return result

    def _extract_concepts(
        self,
        fact_text: str,
    ) -> list[str]:
        """Select concise, deterministic concept spans verbatim from text."""
        tokens = list(
            self.CONCEPT_WORD_PATTERN.finditer(fact_text)
        )
        if not tokens:
            return []

        content_positions = [
            index
            for index, match in enumerate(tokens)
            if match.group(0).lower()
            not in self.CONCEPT_STOPWORDS
            and len(match.group(0)) > 2
        ]

        candidates: dict[str, tuple[float, int]] = {}

        for position in content_positions:
            start = tokens[position].start()
            end = tokens[position].end()
            candidates[
                fact_text[start:end]
            ] = (
                self._concept_score(
                    fact_text[start:end],
                    1,
                )
                + fact_text.casefold().count(
                    fact_text[start:end].casefold()
                ),
                start,
            )

            for next_position in content_positions:
                if next_position <= position:
                    continue

                word_gap = next_position - position
                if word_gap > 4:
                    break

                phrase = fact_text[
                    start:tokens[next_position].end()
                ]
                phrase_words = [
                    token.group(0)
                    for token in tokens[position:next_position + 1]
                ]
                content_word_count = sum(
                    word.lower() not in self.CONCEPT_CONNECTORS
                    and word.lower() not in self.CONCEPT_STOPWORDS
                    for word in phrase_words
                )
                if (
                    len(phrase_words) > 1
                    and len(phrase_words) <= 5
                    and content_word_count >= 2
                    and re.fullmatch(
                        r"[A-Za-z0-9\s'-]+",
                        phrase,
                    )
                ):
                    candidates[phrase] = (
                        self._concept_score(
                            phrase,
                            content_word_count,
                        )
                        + fact_text.casefold().count(
                            phrase.casefold()
                        ),
                        start,
                    )

        ranked = sorted(
            candidates,
            key=lambda concept: (
                -candidates[concept][0],
                candidates[concept][1],
                concept.lower(),
            ),
        )

        selected: list[str] = []
        for concept in ranked:
            normalized = concept.casefold()
            if any(
                re.search(
                    rf"\b{re.escape(normalized)}\b",
                    existing.casefold(),
                )
                for existing in selected
            ):
                continue
            selected.append(concept)
            if len(selected) >= self.MAX_CONCEPTS_PER_FACT:
                break

        return selected

    def _concept_score(
        self,
        concept: str,
        content_word_count: int,
    ) -> float:
        """Rank explicit text spans, favoring concise multiword terms."""
        words = self.CONCEPT_WORD_PATTERN.findall(concept)
        score = float(content_word_count * 2)
        score += sum(
            1
            for word in words
            if len(word) >= 7
        )
        score += sum(
            1
            for word in words
            if word.isupper() and len(word) > 1
        )
        return score

    @staticmethod
    def _format_fact(
        fact_id: str,
        fact_text: str,
        concepts: list[Any],
    ) -> str:
        """Format the retained fact fields for compact-size accounting."""
        lines = []
        if fact_id:
            lines.append(f"fact_id: {fact_id}")
        lines.append(f"text: {fact_text}")
        lines.append(
            "concepts: ["
            + ", ".join(str(concept) for concept in concepts)
            + "]"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # DEBUG / REPORTING
    # ------------------------------------------------------------------

    def print_summary(
        self,
        research: dict[str, Any],
    ) -> None:

        print()

        print(
            "=" * 70
        )

        print(
            "RESEARCH EXTRACTION"
        )

        print(
            "=" * 70
        )

        print(
            f"Topic              : "
            f"{research.get('topic', '')}"
        )

        print(
            f"Sources            : "
            f"{research.get('source_count', 0)}"
        )

        print(
            f"Raw characters     : "
            f"{research.get('raw_character_count', 0):,}"
        )

        print(
            f"Compact characters : "
            f"{research.get('compact_character_count', 0):,}"
        )

        raw = research.get(
            "raw_character_count",
            0,
        )

        compact = research.get(
            "compact_character_count",
            0,
        )

        if raw > 0:

            reduction = (
                1
                - (
                    compact
                    / raw
                )
            ) * 100

            print(
                f"Reduction          : "
                f"{reduction:.1f}%"
            )

        # --------------------------------------------------------------
        # SOURCE / FACT REPORT
        # --------------------------------------------------------------

        print()

        for source in research.get(
            "sources",
            [],
        ):

            source_id = source.get(
                "source_id",
                "unknown",
            )

            title = source.get(
                "title",
                "",
            )

            facts = source.get(
                "facts",
                [],
            )

            print(
                f"[{source_id}] "
                f"{title}"
            )

            print(
                f"    Facts: "
                f"{len(facts)}"
            )

            for fact in facts:

                if isinstance(
                    fact,
                    dict,
                ):

                    print(
                        f"    "
                        f"{fact.get('fact_id', '')}: "
                        f"{fact.get('text', '')}"
                    )

                else:

                    print(
                        f"    {fact}"
                    )

        print(
            "=" * 70
        )


# ----------------------------------------------------------------------
# SIMPLE MANUAL TEST
# ----------------------------------------------------------------------

if __name__ == "__main__":

    extractor = ResearchExtractor()

    sample_research = {
        "topic": "Quantum Computing",
        "source_count": 2,
        "sources": [
            {
                "source_id": "source-001",
                "title": "IBM Quantum Computing",
                "url": "https://example.com/ibm",
                "domain": "example.com",
                "snippet": (
                    "Quantum computing uses quantum "
                    "mechanical phenomena such as "
                    "superposition and entanglement."
                ),
                "content": (
                    "Quantum computers use qubits "
                    "instead of classical bits. "
                    "Qubits can represent quantum "
                    "states using superposition."
                ),
            },
            {
                "source_id": "source-002",
                "title": "NIST Quantum Computing",
                "url": "https://example.com/nist",
                "domain": "example.com",
                "snippet": (
                    "Quantum computing is an emerging "
                    "computing technology."
                ),
                "content": (
                    "Quantum computers may provide "
                    "advantages for specific problems."
                ),
            },
        ],
    }

    extracted = extractor.extract(
        sample_research
    )

    extractor.print_summary(
        extracted
    )