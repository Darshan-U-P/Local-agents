from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

from backend.research.web_search import (
    WebSearch,
    WebPageFetcher,
)

from backend.research.research_extractor import (
    ResearchExtractor,
)


class ResearchManager:

    def __init__(
        self,
        output_dir: str | Path = "generated/research",
    ):

        self.search = WebSearch()
        self.fetcher = WebPageFetcher()

        self.extractor = ResearchExtractor()

        self.output_dir = Path(
            output_dir
        )

    # ========================================================
    # Research
    # ========================================================

    def research(
        self,
        topic: str,
        max_sources: int = 6,
        fetch_pages: bool = True,
    ) -> dict:

        if not topic.strip():
            raise ValueError(
                "Research topic cannot be empty."
            )

        print()
        print("=" * 70)
        print("WEB RESEARCH")
        print("=" * 70)

        print(f"Topic: {topic}")
        print()

        # ----------------------------------------------------
        # Search
        # ----------------------------------------------------

        results = self.search.search(
            topic,
            max_results=max_sources,
        )

        print(
            f"Search results found: {len(results)}"
        )

        sources = []

        # ----------------------------------------------------
        # Fetch pages
        # ----------------------------------------------------

        for index, result in enumerate(
            results,
            start=1,
        ):

            print()
            print(
                f"[{index}/{len(results)}] "
                f"{result.title}"
            )

            domain = urlparse(
                result.url
            ).netloc

            source = {
                "source_id": f"source-{index:03d}",
                "title": result.title,
                "url": result.url,
                "domain": domain,
                "snippet": result.snippet,
                "content": "",
            }

            if fetch_pages:

                try:

                    content = self.fetcher.fetch(
                        result.url
                    )

                    source["content"] = content

                    print(
                        f"    Fetched: "
                        f"{len(content):,} characters"
                    )

                except Exception as exc:

                    print(
                        f"    Fetch failed: {exc}"
                    )

            sources.append(source)

        # ----------------------------------------------------
        # Raw research object
        # ----------------------------------------------------

        raw_research = {
            "topic": topic,
            "source_count": len(sources),
            "sources": sources,
        }

        # ----------------------------------------------------
        # Extract / compress research
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("EXTRACTING RESEARCH")
        print("=" * 70)

        compact_research = self.extractor.extract(
            raw_research
        )

        # ----------------------------------------------------
        # Print extraction statistics
        # ----------------------------------------------------

        raw_characters = compact_research.get(
            "raw_character_count",
            0,
        )

        compact_characters = compact_research.get(
            "compact_character_count",
            0,
        )

        print(
            f"Raw characters     : "
            f"{raw_characters:,}"
        )

        print(
            f"Compact characters : "
            f"{compact_characters:,}"
        )

        if raw_characters > 0:

            reduction = (
                1
                - (
                    compact_characters
                    / raw_characters
                )
            ) * 100

            print(
                f"Reduction          : "
                f"{reduction:.1f}%"
            )

        print(
            f"Sources retained   : "
            f"{compact_research.get('source_count', 0)}"
        )

        print("=" * 70)

        return compact_research

    # ========================================================
    # Save
    # ========================================================

    def save(
        self,
        research: dict,
        filename: str | None = None,
    ) -> Path:

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if filename is None:

            filename = (
                self._safe_filename(
                    research["topic"]
                )
                + ".json"
            )

        output_path = (
            self.output_dir
            / filename
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                research,
                file,
                indent=2,
                ensure_ascii=False,
            )

        return output_path

    # ========================================================
    # Helpers
    # ========================================================

    @staticmethod
    def _safe_filename(
        text: str,
    ) -> str:

        text = text.lower()

        text = re.sub(
            r"[^a-z0-9]+",
            "-",
            text,
        )

        return text.strip("-")