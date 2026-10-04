from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from backend.research.research_manager import (
    ResearchManager,
)


TOPIC = "Quantum Computing"


def main():

    print("=" * 70)
    print("WEB RESEARCH TEST")
    print("=" * 70)

    manager = ResearchManager()

    research = manager.research(
        topic=TOPIC,
        max_sources=6,
        fetch_pages=True,
    )

    output = manager.save(
        research
    )

    print()
    print("=" * 70)
    print("RESEARCH RESULTS")
    print("=" * 70)

    print(
        f"Topic   : {research['topic']}"
    )

    print(
        f"Sources : {research['source_count']}"
    )

    print()

    for source in research["sources"]:

        print(
            f"[{source['source_id']}] "
            f"{source['title']}"
        )

        print(
            f"Domain : {source['domain']}"
        )

        print(
            f"URL    : {source['url']}"
        )

        print(
            f"Content: "
            f"{len(source['content']):,} chars"
        )

        print()

    print(
        f"Saved research to:\n{output}"
    )

    print()
    print("=" * 70)
    print("WEB RESEARCH TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()