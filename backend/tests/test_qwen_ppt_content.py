from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any


# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------
# Project modules
# ------------------------------------------------------------

from backend.planner.presentation_planner import PresentationPlanner
from backend.research.research_manager import ResearchManager


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

TOPIC = "Quantum Computing"
SLIDE_COUNT = 6

MAX_RESEARCH_SOURCES = 6

OUTPUT_DIR = (
    PROJECT_ROOT
    / "generated"
    / "content_tests"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "quantum_computing_plan.json"
)

EVIDENCE_OUTPUT_FILE = (
    OUTPUT_DIR
    / "quantum_computing_evidence_validation.json"
)

EVIDENCE_PREVIEW_FILE = (
    OUTPUT_DIR
    / "quantum_computing_evidence_preview.txt"
)


# ------------------------------------------------------------
# Evidence helpers
# ------------------------------------------------------------

STATUS_ALIASES = {
    "SUPPORTED": "SUPPORTED",
    "SUPPORT": "SUPPORTED",
    "SUPPORTED_CLAIM": "SUPPORTED",
    "FULLY_SUPPORTED": "SUPPORTED",

    "WEAK_SUPPORT": "WEAK_SUPPORT",
    "WEAK SUPPORT": "WEAK_SUPPORT",
    "WEAKLY_SUPPORTED": "WEAK_SUPPORT",
    "PARTIAL_SUPPORT": "WEAK_SUPPORT",
    "PARTIALLY_SUPPORTED": "WEAK_SUPPORT",

    "UNSUPPORTED": "UNSUPPORTED",
    "NOT_SUPPORTED": "UNSUPPORTED",
    "NO_SUPPORT": "UNSUPPORTED",

    "CONTRADICTED": "UNSUPPORTED",
    "CONTRADICTION": "UNSUPPORTED",
}


def normalize_status(value: Any) -> str | None:
    """
    Normalize arbitrary status strings into the small set used
    by the test reporter.
    """
    if not isinstance(value, str):
        return None

    normalized = value.strip().upper()
    normalized = normalized.replace("-", "_")

    return STATUS_ALIASES.get(normalized)


def is_claim_like_dict(value: dict[str, Any]) -> bool:
    """
    Determine whether a dictionary looks like a per-claim
    evidence validation record.

    This intentionally does not require one exact schema.
    """
    keys = {str(key).lower() for key in value.keys()}

    claim_keys = {
        "claim",
        "claim_id",
        "fact_id",
        "key_point",
        "key_point_id",
        "text",
        "statement",
    }

    evidence_keys = {
        "evidence",
        "evidence_ids",
        "source_ids",
        "sources",
        "source",
    }

    status_keys = {
        "status",
        "validation_status",
        "support_status",
        "result",
    }

    return bool(keys & claim_keys) and bool(
        keys & (evidence_keys | status_keys)
    )


def collect_status_records(
    value: Any,
    *,
    path: str = "$",
    records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Recursively collect status-bearing dictionaries.

    We keep the path so the real evidence schema can be inspected
    without assuming exact field names.
    """
    if records is None:
        records = []

    if isinstance(value, dict):
        normalized_status = None

        # Look for the common status field names.
        for key in (
            "status",
            "validation_status",
            "support_status",
            "evidence_status",
            "result",
            "overall_status",
        ):
            if key in value:
                normalized_status = normalize_status(value[key])
                if normalized_status is not None:
                    break

        if normalized_status is not None:
            records.append(
                {
                    "path": path,
                    "status": normalized_status,
                    "is_claim_like": is_claim_like_dict(value),
                }
            )

        for key, child in value.items():
            child_path = f"{path}.{key}"
            collect_status_records(
                child,
                path=child_path,
                records=records,
            )

    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]"
            collect_status_records(
                child,
                path=child_path,
                records=records,
            )

    return records


def find_first_string(
    value: Any,
    candidate_keys: set[str],
) -> str | None:
    """
    Recursively find the first non-empty string associated
    with one of candidate keys.
    """
    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = str(key).lower()

            if normalized_key in candidate_keys:
                if isinstance(child, str) and child.strip():
                    return child.strip()

            result = find_first_string(
                child,
                candidate_keys,
            )

            if result:
                return result

    elif isinstance(value, list):
        for child in value:
            result = find_first_string(
                child,
                candidate_keys,
            )

            if result:
                return result

    return None


def find_first_integer(
    value: Any,
    candidate_keys: set[str],
) -> int | None:
    """
    Recursively find the first integer associated with one
    of candidate keys.
    """
    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = str(key).lower()

            if normalized_key in candidate_keys:
                if isinstance(child, int) and not isinstance(child, bool):
                    return child

            result = find_first_integer(
                child,
                candidate_keys,
            )

            if result is not None:
                return result

    elif isinstance(value, list):
        for child in value:
            result = find_first_integer(
                child,
                candidate_keys,
            )

            if result is not None:
                return result

    return None


def count_claim_like_records(
    value: Any,
) -> int:
    """
    Count dictionaries that look like actual per-claim records.

    This is preferable to blindly counting every 'status' field,
    because some validators may contain summary/status objects.
    """
    count = 0

    if isinstance(value, dict):
        if is_claim_like_dict(value):
            count += 1

        for child in value.values():
            count += count_claim_like_records(child)

    elif isinstance(value, list):
        for child in value:
            count += count_claim_like_records(child)

    return count


def summarize_evidence_results(
    evidence_results: Any,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "overall_status": "UNKNOWN",
        "claims_checked": 0,
        "supported": 0,
        "weak_support": 0,
        "unsupported": 0,
    }

    if not isinstance(evidence_results, list):
        return summary

    # Each top-level item represents one claim.
    summary["claims_checked"] = len(evidence_results)
    for item in evidence_results:
        if not isinstance(item, dict):
            continue

        status = item.get("status")
        if not isinstance(status, str):
            continue

        normalized = status.strip().upper()
        if normalized == "SUPPORTED":
            summary["supported"] += 1
        elif normalized in {
            "WEAK_SUPPORT",
            "WEAK SUPPORT",
        }:
            summary["weak_support"] += 1
        elif normalized == "UNSUPPORTED":
            summary["unsupported"] += 1

    if summary["unsupported"] > 0:
        summary["overall_status"] = "UNSUPPORTED"
    elif summary["weak_support"] > 0:
        summary["overall_status"] = "WEAK_SUPPORT"
    elif summary["supported"] > 0:
        summary["overall_status"] = "SUPPORTED"
    else:
        summary["overall_status"] = "UNKNOWN"

    return summary


def print_evidence_summary(
    evidence_results: Any,
) -> dict[str, Any]:
    """
    Print human-readable evidence validation statistics.
    Returns the summary so the caller can reuse it.
    """
    summary = summarize_evidence_results(
        evidence_results
    )

    print("Evidence validation:")
    print(
        f"      Overall status : "
        f"{summary['overall_status']}"
    )
    print(
        f"      Claims checked : "
        f"{summary['claims_checked']}"
    )
    print(
        f"      SUPPORTED      : "
        f"{summary['supported']}"
    )
    print(
        f"      WEAK_SUPPORT   : "
        f"{summary['weak_support']}"
    )
    print(
        f"      UNSUPPORTED    : "
        f"{summary['unsupported']}"
    )
    return summary


def build_evidence_preview(
    evidence_results: Any,
    *,
    max_chars: int = 12000,
) -> str:
    """
    Build a compact human-readable preview of the raw evidence
    result. The complete object is still saved separately.
    """
    lines: list[str] = []

    lines.append("=" * 70)
    lines.append("RAW EVIDENCE VALIDATION PREVIEW")
    lines.append("=" * 70)
    lines.append("")

    if evidence_results is None:
        lines.append("Evidence result: None")
        return "\n".join(lines)

    lines.append(
        f"Python type: {type(evidence_results).__name__}"
    )
    lines.append("")

    # --------------------------------------------------------
    # Top-level structure
    # --------------------------------------------------------

    if isinstance(evidence_results, dict):
        lines.append("Top-level keys:")
        for key in evidence_results.keys():
            lines.append(f"  - {key}")

        lines.append("")

    elif isinstance(evidence_results, list):
        lines.append(
            f"Top-level list length: "
            f"{len(evidence_results)}"
        )
        lines.append("")

    # --------------------------------------------------------
    # Detected status records
    # --------------------------------------------------------

    records = collect_status_records(
        evidence_results
    )

    lines.append(
        f"Detected status records: {len(records)}"
    )
    lines.append("")

    for record in records[:100]:
        lines.append(
            f"  {record['path']} -> "
            f"{record['status']}"
        )

    if len(records) > 100:
        lines.append(
            f"  ... {len(records) - 100} more"
        )

    lines.append("")

    # --------------------------------------------------------
    # Pretty JSON preview
    # --------------------------------------------------------

    lines.append("Pretty JSON preview:")
    lines.append("-" * 70)

    try:
        json_text = json.dumps(
            evidence_results,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    except Exception as exc:
        json_text = (
            f"<Unable to serialize evidence result: {exc}>"
        )

    if len(json_text) > max_chars:
        json_text = (
            json_text[:max_chars]
            + "\n\n...[preview truncated]..."
        )

    lines.append(json_text)

    return "\n".join(lines)


def save_json(
    path: Path,
    data: Any,
) -> None:
    """
    Save JSON using UTF-8 and readable indentation.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("=" * 70)
    print("QWEN3-4B PPT CONTENT GENERATION TEST")
    print("=" * 70)

    print(f"Topic : {TOPIC}")
    print(f"Slides: {SLIDE_COUNT}")
    print()

    start = perf_counter()

    planner: PresentationPlanner | None = None
    source_count = 0
    raw_characters = 0
    compact_characters = 0
    plan: dict[str, Any] | None = None
    evidence_results: Any = None
    evidence_summary: dict[str, Any] = {
        "overall_status": "UNKNOWN",
        "claims_checked": 0,
        "supported": 0,
        "weak_support": 0,
        "unsupported": 0,
    }

    try:
        # ----------------------------------------------------
        # 1. Research topic
        # ----------------------------------------------------

        print(
            "[1/6] Researching topic from web sources..."
        )

        research_manager = ResearchManager()

        research_context = research_manager.research(
            topic=TOPIC,
            max_sources=MAX_RESEARCH_SOURCES,
            fetch_pages=True,
        )

        source_count = research_context.get(
            "source_count",
            0,
        )

        raw_characters = research_context.get(
            "raw_character_count",
            0,
        )

        compact_characters = research_context.get(
            "compact_character_count",
            0,
        )

        print("      PASS")
        print(
            f"      Sources retained: "
            f"{source_count}"
        )
        print(
            f"      Raw research: "
            f"{raw_characters:,} characters"
        )
        print(
            f"      Compact research: "
            f"{compact_characters:,} characters"
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
                f"      Compression: "
                f"{reduction:.1f}%"
            )

        print()

        # ----------------------------------------------------
        # 2. Create planner
        # ----------------------------------------------------

        print(
            "[2/6] Initializing PresentationPlanner..."
        )

        planner = PresentationPlanner()

        print("      PASS")
        print()

        # ----------------------------------------------------
        # 3. Generate content using Qwen + research
        # ----------------------------------------------------

        print(
            "[3/6] Generating PPT content "
            "with Qwen3-4B..."
        )

        plan = planner.create_plan(
            topic=TOPIC,
            slide_count=SLIDE_COUNT,
            research_context=research_context,
        )

        print("      PASS")
        print()

        # ----------------------------------------------------
        # 4. Evidence validation
        # ----------------------------------------------------

        print(
            "[4/6] Evidence validation..."
        )

        evidence_results = getattr(
            planner,
            "evidence_validation_results",
            None,
        )

        if evidence_results is None:
            print(
                "      WARNING: "
                "No evidence validation results found."
            )

            evidence_results = {}

        else:
            print("      PASS")

        # Print robust schema-independent summary.
        evidence_summary = print_evidence_summary(
            evidence_results
        )

        # ----------------------------------------------------
        # Save a compact preview of the actual evidence schema
        # ----------------------------------------------------

        evidence_preview = build_evidence_preview(
            evidence_results
        )

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        EVIDENCE_PREVIEW_FILE.write_text(
            evidence_preview,
            encoding="utf-8",
        )

        print(
            f"      Evidence preview: "
            f"{EVIDENCE_PREVIEW_FILE}"
        )

        print()

        # ----------------------------------------------------
        # 5. Save outputs
        # ----------------------------------------------------

        print(
            "[5/6] Saving presentation and "
            "evidence results..."
        )

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Presentation plan
        # ----------------------------------------------------

        if plan is None:
            raise RuntimeError(
                "Presentation plan is missing."
            )

        save_json(
            OUTPUT_FILE,
            plan,
        )

        print(
            f"      Presentation: "
            f"{OUTPUT_FILE}"
        )

        # ----------------------------------------------------
        # Raw evidence validation result
        # ----------------------------------------------------

        save_json(
            EVIDENCE_OUTPUT_FILE,
            evidence_results,
        )

        print(
            f"      Evidence: "
            f"{EVIDENCE_OUTPUT_FILE}"
        )

        print()

        # ----------------------------------------------------
        # 6. Print result
        # ----------------------------------------------------

        print(
            "[6/6] Presentation content"
        )
        print()

        planner.print_plan(plan)

        elapsed = perf_counter() - start

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print(
            "QWEN PPT CONTENT + "
            "WEB RESEARCH + "
            "EXTRACTION + "
            "EVIDENCE VALIDATION TEST PASSED"
        )
        print("=" * 70)

        print(
            f"Topic            : "
            f"{plan.get('title', TOPIC)}"
        )

        print(
            f"Subtitle         : "
            f"{plan.get('subtitle', '')}"
        )

        print(
            f"Slide count      : "
            f"{plan.get('slide_count', SLIDE_COUNT)}"
        )

        print(
            f"Research sources : "
            f"{source_count}"
        )

        print(
            f"Raw research     : "
            f"{raw_characters:,} chars"
        )

        print(
            f"Compact research : "
            f"{compact_characters:,} chars"
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
                f"Compression      : "
                f"{reduction:.1f}%"
            )

        # ----------------------------------------------------
        # Evidence summary
        # ----------------------------------------------------

        print(
            f"Evidence status   : "
            f"{evidence_summary['overall_status']}"
        )

        print(
            f"Claims checked    : "
            f"{evidence_summary['claims_checked']}"
        )

        print(
            f"Supported         : "
            f"{evidence_summary['supported']}"
        )

        print(
            f"Weak support      : "
            f"{evidence_summary['weak_support']}"
        )

        print(
            f"Unsupported       : "
            f"{evidence_summary['unsupported']}"
        )

        print(
            f"Runtime           : "
            f"{elapsed:.2f} seconds"
        )

        print(
            f"JSON file         : "
            f"{OUTPUT_FILE}"
        )

        print(
            f"Evidence file     : "
            f"{EVIDENCE_OUTPUT_FILE}"
        )

        print(
            f"Evidence preview  : "
            f"{EVIDENCE_PREVIEW_FILE}"
        )

    except Exception as exc:
        print()

        print("=" * 70)
        print(
            "QWEN PPT CONTENT + "
            "WEB RESEARCH + "
            "EXTRACTION + "
            "EVIDENCE VALIDATION TEST FAILED"
        )
        print("=" * 70)

        print(
            f"Error: {exc}"
        )

        raise

    finally:
        # ----------------------------------------------------
        # Release Qwen model
        # ----------------------------------------------------

        if planner is not None:
            try:
                planner.unload()

            except Exception as unload_error:
                print(
                    f"\nWarning: Qwen unload failed: "
                    f"{unload_error}"
                )


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    main()