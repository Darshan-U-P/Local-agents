import json
import re
from typing import Any

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
        6. Source attribution

    The planner does NOT generate images.

    Example pipeline:

        User topic
            ↓
        Research
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

    MIN_REPAIR_COVERAGE = 0.60
    MIN_REPAIR_SCORE = 0.55
    MIN_REPAIR_MARGIN = 0.10

    def __init__(self):
        self.model_manager = ModelManager()
        self.evidence_validation_results: list[dict] = []
        self.unsupported_claim_removals: list[dict] = []

    # =========================================================
    # CREATE PRESENTATION PLAN
    # =========================================================

    def create_plan(
        self,
        topic: str,
        slide_count: int = 6,
        research_context: dict | None = None,
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

        has_research = bool(
            research_context
            and isinstance(research_context, dict)
            and research_context.get("sources")
        )

        # =====================================================
        # SYSTEM PROMPT
        # =====================================================

        system_prompt = """
You create concise, editable PowerPoint presentation plans; do not create
the deck or images. Return only complete, valid JSON (no Markdown or
explanations), using exactly the requested slide count and this schema:
top-level title, subtitle, slide_count, slides; each slide has
slide_number, title, purpose, layout, key_points, assets; every
key_points item is an object with text and sources; each asset has type,
description, source_ids, visual_spec. Add no fields.

Use only these layouts:
hero, title_content, two_column, three_column, image_left, image_right,
big_stat, timeline, process, comparison, quote, full_image, diagram,
chart, cards, section_divider.
Asset types: image, icon, diagram, chart, illustration, photo, none.
Layouts and asset types are separate namespaces: layout names go only
in slide["layout"], and asset types go only in asset["type"]. In
particular, big_stat belongs only in slide["layout"]; statistics do not
use an asset type called "big_stat". If a big_stat slide needs no visual
asset, use an asset with type "none"; if it needs a visual, use only an
allowed asset type such as image, icon, diagram, chart, illustration,
photo, or none. Never use a layout name as an asset type.
Timeline, process, comparison, and cards are also layouts, not asset
types. Visualize these with editable elements or an allowed diagram/chart
asset, never an image when an editable element is appropriate.
For a slide whose layout is "timeline", include a diagram asset.

FACTUAL CLAIM RULES
- Each factual key point expresses ONE primary factual assertion.
- Follow this fact-first workflow for every factual key point:
  STEP 1: Select one exact research fact_id that directly supports the
  intended claim.
  STEP 2: Write one conservative sentence that can be directly
  paraphrased from that fact.
  STEP 3: Put that exact fact_id in the claim's sources list.
- Never write a factual claim first and then search for a related
  citation. Select the supporting fact first.
- A fact_id is valid only if the cited fact supports the complete meaning
  of the sentence.
- Every factual key point must be directly expressible from one or more
  supplied research facts. Before writing it, identify the exact
  supporting fact or facts.
- Do not introduce a new problem, limitation, challenge, cause, effect,
  comparison, capability, trend, or conclusion unless the supplied
  research facts explicitly establish it.
- Never infer a general problem or limitation from a future target,
  roadmap, benchmark, project goal, or development milestone.
- A fact mentioning a target such as "2,000 logical qubits by 2033" does
  NOT support claims such as "error correction remains unsolved",
  "scalability remains unsolved", or "decoherence is the main challenge"
  unless the research explicitly states those things.
- Do not add concepts that are not explicitly present in the selected
  fact. Do not infer mechanisms, applications, benefits, limitations,
  performance, causality, or implications.
- Normally, one factual key point uses one fact_id. Do not combine
  information from different facts into one factual sentence; make
  separate claims when facts support separate key points.
- Prefer a narrower claim with strong evidence over a broader claim with
  weak evidence.
- If no research fact supports a useful claim, omit that claim.

RESEARCH-ONLY GENERATION
- Supplied research is the factual knowledge boundary; do not add facts
  simply because they are generally true or infer implications.
- Do not use pretrained knowledge to fill evidence gaps.
- Do not add scientific explanations, applications, performance claims,
  risks, limitations, or historical context unless explicitly supported.
- Omit useful-sounding claims that are not supported by research.
- Prefer authoritative source content over snippets; represent source
  disagreement rather than silently choosing a value.
- Use qualified wording when a claim has conditions. Avoid promotional
  terms such as revolutionize, revolutionary, game-changing, critical,
  dramatically faster, and solve everything; avoid unsupported absolutes.

CITATION RULE
For every factual key point, follow the fact-first workflow above and
cite only its exact supporting fact_id in "sources". Do not cite a fact
merely because it is related to the topic. If no supplied fact supports
the complete claim, do not generate it.
Never invent fact IDs. Non-factual organizational text may use an empty
"sources" list.

TIMELINES
Every year, date, period, and milestone must be directly supported by
research. Never invent dates, balance a timeline with placeholders, or
present a historical theory as a modern milestone. Distinguish historical
events, current achievements, and future targets where relevant.

ASSETS AND SOURCES
Use assets only when they add value; prefer editable PowerPoint visuals
for charts, timelines, processes, comparisons, and statistics. A
research-backed asset's source_ids contain source IDs only (never fact
IDs); use [] when no source directly informs it. Researched chart data
requires its source IDs. Every non-none asset MUST contain a visual_spec
object with ALL EIGHT fields. Never omit any field:
purpose, subject, composition, style, color_palette, must_show,
must_avoid, text_policy.
Use this exact object shape:
"visual_spec": {
  "purpose": "...",
  "subject": "...",
  "composition": "...",
  "style": "...",
  "color_palette": ["...", "..."],
  "must_show": ["..."],
  "must_avoid": ["..."],
  "text_policy": "..."
}
Keep values concise; use 2-3 colors, up to 3 must_show items, and up to
2 must_avoid items. Image/illustration/photo assets must not contain
text, labels, numbers, logos, or watermarks.
For type "none", use source_ids: [] and visual_spec: null.

FINAL ASSET CHECKLIST
Before returning JSON, verify for every asset:
- type is one of the allowed asset types;
- if type != "none", visual_spec exists and contains all eight fields;
- no required field is omitted and no unknown asset type is used;
- layout names are never used as asset types.
Do not finish the JSON until every non-none asset contains all eight
visual_spec fields.

For quantum topics, do not imply entanglement enables faster-than-light
communication or that quantum computers universally outperform classical
computers. Qualify claims such as Shor's algorithm's cryptographic impact
with their research-supported conditions.
"""

        # =====================================================
        # USER PROMPT
        # =====================================================

        user_prompt = f"""
Create a {slide_count}-slide presentation about:

{topic}

RESEARCH MATERIAL
=================

The following information was collected from web sources.

Use this research as the factual basis for the presentation.

Do NOT invent facts or statistics.

If the research does not support a specific claim,
do not make that claim.

FACTUAL CLAIM WORKFLOW:
1. Select one exact research fact_id that directly supports the intended
   claim.
2. Write one conservative sentence that can be directly paraphrased
   from that fact.
3. Put that exact fact_id in the claim's "sources" list.
Never write a factual claim first and then search for a related citation.
The cited fact must support the complete meaning of the sentence. Do not
add concepts or infer mechanisms, applications, benefits, limitations,
performance, causality, or implications that are not explicitly present
in the selected fact. Prefer a narrower claim with strong evidence over
a broader claim with weak evidence. If no research fact supports a
useful claim, omit it.
Every factual key point must be directly expressible from supplied facts;
identify the exact supporting fact or facts before writing it. Do not
introduce a new problem, limitation, challenge, cause, effect, comparison,
capability, trend, or conclusion unless the supplied facts explicitly
establish it. Do not infer a general problem or limitation from a future
target, roadmap, benchmark, project goal, or development milestone.
Specifically, a fact mentioning "2,000 logical qubits by 2033" does NOT
support "error correction remains unsolved", "scalability remains
unsolved", or "decoherence is the main challenge" unless the research
explicitly states those things. Do not use pretrained knowledge to fill
evidence gaps. Keep claims concise and omit unsupported claims rather
than filling a slide with general knowledge.

When a visual directly depends on a research source,
include the exact source_id in the asset's "source_ids" list.

Never invent source IDs or fact IDs.

Research data:

{json.dumps(
    research_context or {},
    indent=2,
    ensure_ascii=False,
)}

Requirements:

- Exactly {slide_count} slides.
- Create a logical introduction-to-conclusion narrative; every slide
  needs a clear purpose and a supported layout.
- Follow the factual-claim, research-only, and citation rules above;
  cite visual assets with source_id values. Never invent IDs, dates,
  statistics, or claims. Distinguish historic, current, and future
  milestones.
- Use about 3 key points per slide and only assets that add value.
- For visuals, prefer editable elements; keep visual_spec concise,
  actionable, and within the specified field/item limits.
- Entanglement does not enable faster-than-light communication; do not
  claim universal quantum advantage.

OUTPUT LENGTH RULES:
- Keep every key point under 16 words.
- Use at most 3 key points per slide.
- Use at most 1 asset per slide.
- Keep asset descriptions concise.
- Keep each visual_spec field concise.
- Do not repeat information between purpose, key_points, and assets.
- The JSON must be fully closed and complete.
- Prefer concise wording over additional detail.
- Use the minimum amount of text necessary to satisfy the schema.
  Do not add explanations outside the JSON.
- Return JSON only.
"""

        # =====================================================
        # GENERATE PLAN
        # =====================================================

        response = self.model_manager.generate(
            prompt=user_prompt,
            max_tokens=3200,
            temperature=0.2,
            system_prompt=system_prompt,
        )

        return self._parse_json(
            response,
            research_context=research_context,
        )

    # =========================================================
    # UNLOAD MODEL
    # =========================================================

    def unload(self):
        """
        Unload the Qwen model used by the presentation planner.
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
        research_context: dict | None = None,
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

        response = response[: end + 1]

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
        # Normalize and Validate
        # -----------------------------------------------------

        self._normalize_plan(plan)
        self._validate_citations(
            plan=plan,
            research_context=research_context,
        )
        self._validate_semantics(plan)
        self._validate_evidence(
            plan=plan,
            research_context=research_context,
        )
        self._remove_unsupported_key_points(plan)
        self._validate_plan(plan)

        return plan

    def _remove_unsupported_key_points(
        self,
        plan: dict,
    ) -> list[dict]:
        """Remove only key points marked UNSUPPORTED by evidence validation."""
        evidence_by_position = {
            (result.get("slide_index"), result.get("point_index")): result
            for result in self.evidence_validation_results
            if isinstance(result, dict)
        }
        removals: list[dict] = []
        slides = plan.get("slides", [])
        if not isinstance(slides, list):
            self.unsupported_claim_removals = removals
            return removals

        for slide_index, slide in enumerate(slides, start=1):
            if not isinstance(slide, dict):
                continue

            key_points = slide.get("key_points", [])
            if not isinstance(key_points, list):
                continue

            retained_key_points = []
            for point_index, point in enumerate(key_points, start=1):
                evidence_result = evidence_by_position.get(
                    (slide_index, point_index)
                )
                if (
                    isinstance(evidence_result, dict)
                    and evidence_result.get("status") == "UNSUPPORTED"
                ):
                    original_citations = evidence_result.get(
                        "original_citations",
                        point.get("sources", [])
                        if isinstance(point, dict)
                        else [],
                    )
                    if not isinstance(original_citations, list):
                        original_citations = []

                    removal = {
                        "slide_number": slide_index,
                        "point_number": point_index,
                        "claim": (
                            point.get("text", "")
                            if isinstance(point, dict)
                            else ""
                        ),
                        "original_citations": list(original_citations),
                        "reason": "UNSUPPORTED",
                    }
                    removals.append(removal)
                    print(
                        "[Planner] Removed unsupported key point "
                        f"on slide {slide_index}, point {point_index}: "
                        f"{removal['claim']} "
                        f"(original citations: "
                        f"{removal['original_citations']}; "
                        "reason: UNSUPPORTED)"
                    )
                    continue

                retained_key_points.append(point)

            slide["key_points"] = retained_key_points

        self.unsupported_claim_removals = removals
        return removals

    def _validate_citations(
        self,
        plan: dict,
        research_context: dict | None,
    ) -> None:
        if research_context is None or not isinstance(
            research_context,
            dict,
        ):
            return

        sources = research_context.get("sources", [])
        if not isinstance(sources, list):
            sources = []

        valid_source_ids: set[str] = set()
        valid_fact_ids: set[str] = set()

        for source in sources:
            if not isinstance(source, dict):
                continue

            source_id = source.get("source_id")
            if isinstance(source_id, str) and source_id.strip():
                valid_source_ids.add(source_id)

            facts = source.get("facts", [])
            if not isinstance(facts, list):
                continue

            for fact in facts:
                if not isinstance(fact, dict):
                    continue

                fact_id = fact.get("fact_id")
                if isinstance(fact_id, str) and fact_id.strip():
                    valid_fact_ids.add(fact_id)

        slides = plan.get("slides", [])
        if not isinstance(slides, list):
            return

        for slide_index, slide in enumerate(
            slides,
            start=1,
        ):
            if not isinstance(slide, dict):
                continue

            key_points = slide.get("key_points", [])
            if isinstance(key_points, list):
                for point_index, point in enumerate(
                    key_points,
                    start=1,
                ):
                    if not isinstance(point, dict):
                        continue

                    citations = point.get("sources", [])
                    if not isinstance(citations, list):
                        continue

                    for fact_id in citations:
                        if (
                            isinstance(fact_id, str)
                            and fact_id not in valid_fact_ids
                        ):
                            raise ValueError(
                                f"Slide {slide_index} key point "
                                f"{point_index} cites unknown "
                                f"fact_id: {fact_id!r}."
                            )

            assets = slide.get("assets", [])
            if isinstance(assets, list):
                for asset_index, asset in enumerate(
                    assets,
                    start=1,
                ):
                    if not isinstance(asset, dict):
                        continue

                    citations = asset.get("source_ids", [])
                    if not isinstance(citations, list):
                        continue

                    for source_id in citations:
                        if not isinstance(source_id, str):
                            raise ValueError(
                                f"Slide {slide_index} asset "
                                f"{asset_index} has a non-string "
                                "source_id."
                            )

                        if source_id in valid_source_ids:
                            continue

                        # Qwen sometimes puts a fact_id into an
                        # asset source_ids list. Convert only known
                        # fact IDs to their verified parent source.
                        if source_id in valid_fact_ids:
                            parent_source_id = source_id.split(
                                "-fact-",
                                1,
                            )[0]
                            if parent_source_id in valid_source_ids:
                                asset["source_ids"] = [
                                    parent_source_id
                                    if item == source_id
                                    else item
                                    for item in asset["source_ids"]
                                ]
                                continue

                        raise ValueError(
                            f"Slide {slide_index} asset "
                            f"{asset_index} cites unknown source_id: "
                            f"'{source_id}'."
                        )

    def _validate_evidence(
        self,
        plan: dict,
        research_context: dict | None,
    ) -> list[dict]:
        """
        Classify claim support from fact text and rank repair candidates.

        Fact text determines support status. Supplied concepts add a
        diagnostic overlap score and a secondary candidate-ranking signal.
        Results are exposed on ``evidence_validation_results``; citations
        are repaired only after the existing checks and final revalidation.
        """

        evidence_by_fact_id: dict[str, dict[str, Any]] = {}
        if not isinstance(research_context, dict):
            self.evidence_validation_results = []
            return []

        sources = research_context.get("sources", [])
        if isinstance(sources, list):
            for source in sources:
                if not isinstance(source, dict):
                    continue

                facts = source.get("facts", [])
                if not isinstance(facts, list):
                    continue

                for fact in facts:
                    if not isinstance(fact, dict):
                        continue

                    fact_id = fact.get("fact_id")
                    fact_text = fact.get("text")
                    concepts = fact.get("concepts", [])
                    if (
                        isinstance(fact_id, str)
                        and fact_id.strip()
                        and isinstance(fact_text, str)
                    ):
                        evidence_by_fact_id[fact_id] = {
                            "fact_text": fact_text,
                            "concepts": (
                                concepts
                                if isinstance(concepts, list)
                                else []
                            ),
                        }

        results: list[dict] = []
        slides = plan.get("slides", [])
        if not isinstance(slides, list):
            self.evidence_validation_results = results
            return results

        for slide_index, slide in enumerate(
            slides,
            start=1,
        ):
            if not isinstance(slide, dict):
                continue

            key_points = slide.get("key_points", [])
            if not isinstance(key_points, list):
                continue

            for point_index, point in enumerate(
                key_points,
                start=1,
            ):
                if not isinstance(point, dict):
                    continue

                claim = point.get("text", "")
                if not isinstance(claim, str):
                    continue

                citations = point.get("sources", [])
                if not isinstance(citations, list):
                    continue

                original_citations = list(citations)
                repaired_from: list[str] = []
                rejected_candidates: list[dict] = []
                repaired_citations: list[str] = []
                for fact_id in citations:
                    current_evidence = evidence_by_fact_id.get(
                        fact_id if isinstance(fact_id, str) else "",
                        {},
                    )
                    current_fact = {
                        "fact_id": fact_id,
                        **current_evidence,
                    }
                    current_evaluation = self._evaluate_claim_against_fact(
                        claim=claim,
                        fact=current_fact,
                    )
                    best_candidate: dict | None = None
                    best_candidate_rank: tuple[float, float] | None = None
                    current_score = (
                        0.75
                        * current_evaluation["claim_term_coverage"]
                        + 0.25
                        * current_evaluation["term_similarity"]
                    )

                    for candidate_fact_id, candidate_evidence in (
                        evidence_by_fact_id.items()
                    ):
                        if candidate_fact_id == fact_id:
                            continue
                        candidate_fact_text = candidate_evidence[
                            "fact_text"
                        ]
                        if not candidate_fact_text:
                            rejected_candidates.append(
                                {
                                    "rejected": True,
                                    "reason": "empty_fact_text",
                                    "candidate_fact_id": candidate_fact_id,
                                }
                            )
                            continue

                        candidate = {
                            "fact_id": candidate_fact_id,
                            **candidate_evidence,
                        }
                        candidate_evaluation = self._evaluate_claim_against_fact(
                            claim=claim,
                            fact=candidate,
                        )
                        candidate_coverage = candidate_evaluation[
                            "claim_term_coverage"
                        ]
                        if candidate_coverage < self.MIN_REPAIR_COVERAGE:
                            rejected_candidates.append(
                                {
                                    "rejected": True,
                                    "reason": "below_coverage",
                                    "candidate_fact_id": candidate_fact_id,
                                    "candidate_coverage": candidate_coverage,
                                    "candidate_concept_overlap_score": (
                                        candidate_evaluation[
                                            "concept_overlap_score"
                                        ]
                                    ),
                                }
                            )
                            continue

                        candidate_similarity = candidate_evaluation[
                            "term_similarity"
                        ]
                        candidate_score = (
                            0.75 * candidate_coverage
                            + 0.25 * candidate_similarity
                        )
                        if candidate_score < self.MIN_REPAIR_SCORE:
                            rejected_candidates.append(
                                {
                                    "rejected": True,
                                    "reason": "below_score",
                                    "candidate_fact_id": candidate_fact_id,
                                    "candidate_score": candidate_score,
                                    "candidate_concept_overlap_score": (
                                        candidate_evaluation[
                                            "concept_overlap_score"
                                        ]
                                    ),
                                }
                            )
                            continue
                        if candidate_score < (
                            current_score + self.MIN_REPAIR_MARGIN
                        ):
                            rejected_candidates.append(
                                {
                                    "rejected": True,
                                    "reason": "insufficient_margin",
                                    "candidate_fact_id": candidate_fact_id,
                                    "candidate_score": candidate_score,
                                    "current_score": current_score,
                                    "candidate_concept_overlap_score": (
                                        candidate_evaluation[
                                            "concept_overlap_score"
                                        ]
                                    ),
                                }
                            )
                            continue
                        if not self._repair_candidate_is_compatible(
                            claim,
                            candidate_fact_text,
                        ):
                            rejected_candidates.append(
                                {
                                    "rejected": True,
                                    "reason": "missing_required_anchor",
                                    "candidate_fact_id": candidate_fact_id,
                                    "candidate_concept_overlap_score": (
                                        candidate_evaluation[
                                            "concept_overlap_score"
                                        ]
                                    ),
                                }
                            )
                            continue

                        revalidated = self._evaluate_claim_against_fact(
                            claim=claim,
                            fact=candidate,
                        )
                        if revalidated["status"] != "SUPPORTED":
                            rejected_candidates.append(
                                {
                                    "rejected": True,
                                    "reason": "not_supported",
                                    "candidate_fact_id": candidate_fact_id,
                                    "candidate_status": revalidated["status"],
                                    "candidate_concept_overlap_score": (
                                        revalidated[
                                            "concept_overlap_score"
                                        ]
                                    ),
                                }
                            )
                            continue

                        candidate_rank = self._repair_candidate_rank(
                            candidate_score,
                            candidate_evaluation[
                                "concept_overlap_score"
                            ],
                        )
                        if (
                            best_candidate_rank is None
                            or candidate_rank > best_candidate_rank
                        ):
                            best_candidate = candidate
                            best_candidate_rank = candidate_rank

                    replacement_id = (
                        best_candidate["fact_id"]
                        if best_candidate is not None
                        else fact_id
                    )
                    if replacement_id not in repaired_citations:
                        repaired_citations.append(replacement_id)
                    if (
                        best_candidate is not None
                        and replacement_id != fact_id
                    ):
                        repaired_from.append(fact_id)

                if repaired_from:
                    point["sources"] = repaired_citations
                    citations = repaired_citations

                fact_results: list[dict] = []
                for fact_id in citations:
                    evidence = evidence_by_fact_id.get(
                        fact_id if isinstance(fact_id, str) else "",
                        {},
                    )
                    fact_text = evidence.get("fact_text", "")
                    evaluation = self._evaluate_claim_against_fact(
                        claim=claim,
                        fact={
                            "fact_id": fact_id,
                            "fact_text": fact_text,
                            **evidence,
                        },
                    )
                    fact_results.append(
                        {
                            "fact_id": fact_id,
                            "fact_text": fact_text,
                            **evaluation,
                        }
                    )

                if not fact_results:
                    overall_status = "UNSUPPORTED"
                elif any(
                    result["status"] == "SUPPORTED"
                    for result in fact_results
                ):
                    overall_status = "SUPPORTED"
                elif any(
                    result["status"] == "WEAK_SUPPORT"
                    for result in fact_results
                ):
                    overall_status = "WEAK_SUPPORT"
                else:
                    overall_status = "UNSUPPORTED"

                results.append(
                    {
                        "slide_index": slide_index,
                        "point_index": point_index,
                        "claim": claim,
                        "status": overall_status,
                        "original_citations": original_citations,
                        "citation_repaired_from": repaired_from,
                        "rejected_candidates": rejected_candidates,
                        "citations": fact_results,
                    }
                )

        self.evidence_validation_results = results
        return results

    def _assess_claim_support(
        self,
        claim: str,
        evidence: str,
    ) -> tuple[str, float, float, list[str]]:
        stop_words = {
            "a", "an", "and", "are", "as", "at", "be", "been",
            "being", "by", "can", "for", "from", "in", "into",
            "is", "it", "its", "of", "on", "or", "s", "that",
            "the", "their", "this", "to", "was", "were", "with",
        }
        synonym_groups = (
            {"algorithm", "algorithms"},
            {"factor", "factors", "factored", "factoring"},
            {"threat", "threats", "threaten", "threatens", "threatening"},
            {"cryptography", "cryptographic", "encryption", "encrypt"},
            {"computer", "computers"},
            {"quantum"},
            {"classical"},
            {"integer", "integers"},
        )
        canonical_terms = {
            term: min(group)
            for group in synonym_groups
            for term in group
        }

        def terms(text: str) -> set[str]:
            result = set()
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                if token in stop_words:
                    continue
                result.add(canonical_terms.get(token, token))
            return result

        claim_terms = terms(claim)
        evidence_terms = terms(evidence)
        matched_terms = sorted(claim_terms & evidence_terms)

        if not claim_terms:
            claim_term_coverage = 0.0
        else:
            claim_term_coverage = (
                len(matched_terms) / len(claim_terms)
            )

        union_terms = claim_terms | evidence_terms
        if not union_terms:
            term_similarity = 0.0
        else:
            term_similarity = (
                len(matched_terms) / len(union_terms)
            )

        if (
            claim_term_coverage >= 0.7
            and len(matched_terms) >= 2
        ):
            status = "SUPPORTED"
        elif (
            claim_term_coverage >= 0.3
            and matched_terms
        ):
            status = "WEAK_SUPPORT"
        else:
            status = "UNSUPPORTED"

        return (
            status,
            claim_term_coverage,
            term_similarity,
            matched_terms,
        )

    def _extract_required_terms(
        self,
        claim: str,
    ) -> set[str]:
        """
        Extract important terms that should be preserved by a
        citation repair candidate using simple lexical rules.
        """
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "but",
            "by", "can", "could", "for", "from", "has", "have",
            "in", "into", "is", "it", "of", "on", "or", "that",
            "the", "their", "this", "to", "using", "with",
            "may", "might", "than", "these", "those",
        }
        tokens = re.findall(
            r"[a-z0-9]+",
            claim.lower(),
        )
        return {
            token
            for token in tokens
            if len(token) >= 4 and token not in stopwords
        }

    def _repair_candidate_is_compatible(
        self,
        claim: str,
        fact_text: str,
    ) -> bool:
        claim_text = claim.lower().replace("-", " ")
        fact = fact_text.lower().replace("-", " ")

        concept_groups = [
            ({"shor", "shor's"}, {"shor", "shor's"}),
            (
                {"cryptography", "cryptographic"},
                {"cryptography", "cryptographic"},
            ),
            ({"entanglement", "entangled"}, {"entanglement", "entangled"}),
            ({"superposition"}, {"superposition"}),
            ({"decoherence"}, {"decoherence"}),
            ({"annealing"}, {"annealing"}),
            (
                {"optimization", "optimisation"},
                {"optimization", "optimisation"},
            ),
            (
                {"material science", "materials science"},
                {"material science", "materials science"},
            ),
            ({"logistics"}, {"logistics"}),
            ({"finance", "financial"}, {"finance", "financial"}),
            ({"teleportation"}, {"teleportation"}),
            ({"qiskit"}, {"qiskit"}),
            ({"ibm"}, {"ibm"}),
            ({"d wave", "dwave"}, {"d wave", "dwave"}),
            (
                {"error correction", "error-correction"},
                {"error correction", "error-correction"},
            ),
            ({"qubit", "qubits"}, {"qubit", "qubits"}),
        ]

        required_groups: list[set[str]] = []
        for claim_aliases, fact_aliases in concept_groups:
            if any(alias in claim_text for alias in claim_aliases):
                required_groups.append(fact_aliases)

        for aliases in required_groups:
            if not any(alias in fact for alias in aliases):
                return False

        if not self._extract_numeric_tokens(claim).issubset(
            self._extract_numeric_tokens(fact_text)
        ):
            return False

        special_phrase_groups = (
            (
                {"faster than light", "faster-than-light"},
                {"faster than light", "faster-than-light"},
            ),
            (
                {"public key", "public-key"},
                {"public key", "public-key"},
            ),
        )
        for claim_aliases, fact_aliases in special_phrase_groups:
            if any(alias in claim_text for alias in claim_aliases):
                if not any(alias in fact for alias in fact_aliases):
                    return False

        required_terms = self._extract_required_terms(claim)
        fact_terms = self._extract_required_terms(fact_text)
        if required_terms and not required_terms.intersection(fact_terms):
            return False

        return True

    def _extract_numeric_tokens(
        self,
        text: str,
    ) -> set[str]:
        """
        Extract meaningful numeric tokens and normalize commas.

        Examples:
            2,000 -> 2000
            2033  -> 2033
            105   -> 105
        """
        matches = re.findall(
            r"\b\d[\d,]*(?:\.\d+)?\b",
            text.lower(),
        )
        normalized: set[str] = set()
        for value in matches:
            cleaned = value.replace(",", "")
            if cleaned.isdigit():
                normalized.add(cleaned)
            elif re.fullmatch(r"\d+\.\d+", cleaned):
                normalized.add(cleaned)
        return normalized

    def _evaluate_claim_against_fact(
        self,
        claim: str,
        fact: dict,
    ) -> dict:
        fact_text = fact.get("fact_text", "")
        if not isinstance(fact_text, str):
            fact_text = ""

        (
            status,
            claim_term_coverage,
            term_similarity,
            matched_terms,
        ) = self._assess_claim_support(
            claim=claim,
            evidence=fact_text,
        )
        concepts = fact.get("concepts", [])
        if not isinstance(concepts, list):
            concepts = []

        return {
            "status": status,
            "claim_term_coverage": claim_term_coverage,
            "term_similarity": term_similarity,
            "matched_terms": matched_terms,
            "concept_overlap_score": self._concept_overlap_score(
                claim,
                concepts,
            ),
        }

    @staticmethod
    def _concept_overlap_score(
        claim: str,
        concepts: list[Any],
    ) -> float:
        """Measure claim-token coverage by supplied concept text only."""
        ignored_terms = {
            "a", "an", "and", "are", "as", "at", "be", "been",
            "being", "by", "can", "could", "did", "do", "does",
            "for", "from", "had", "has", "have", "in", "into",
            "is", "it", "its", "may", "might", "must", "of", "on",
            "or", "should", "that", "the", "their", "them", "they",
            "this", "those", "through", "to", "was", "were", "will",
            "with", "would",
        }

        def meaningful_tokens(text: str) -> set[str]:
            return {
                token
                for token in re.findall(
                    r"[a-z0-9]+",
                    text.lower().replace("-", " "),
                )
                if len(token) > 1 and token not in ignored_terms
            }

        claim_tokens = meaningful_tokens(claim)
        if not claim_tokens:
            return 0.0

        concept_tokens: set[str] = set()
        for concept in concepts:
            if isinstance(concept, str):
                concept_tokens.update(
                    meaningful_tokens(concept)
                )

        return len(claim_tokens & concept_tokens) / len(claim_tokens)

    @staticmethod
    def _repair_candidate_rank(
        lexical_score: float,
        concept_overlap_score: float,
    ) -> tuple[float, float]:
        """Use concept overlap as a small ranking tie-break signal."""
        return (
            lexical_score + 0.05 * concept_overlap_score,
            lexical_score,
        )

    @staticmethod
    def _evidence_rank(
        assessment: tuple[str, float, float, list[str]],
    ) -> tuple[int, float, float]:
        status, claim_term_coverage, term_similarity, _ = assessment
        status_rank = {
            "UNSUPPORTED": 0,
            "WEAK_SUPPORT": 1,
            "SUPPORTED": 2,
        }.get(status, 0)
        return (
            status_rank,
            claim_term_coverage,
            term_similarity,
        )

    # =========================================================
    # NORMALIZE PLAN
    # =========================================================

    def _normalize_plan(self, plan: dict) -> None:
        
        slides = plan.get("slides", [])
        if not isinstance(slides, list):
            return
            
        for slide in slides:
            if not isinstance(slide, dict):
                continue
                
            assets = slide.get("assets", [])
            if not isinstance(assets, list):
                continue
                
            for asset in assets:
                if not isinstance(asset, dict):
                    continue
                    
                asset_type = asset.get("type")
                
                # -------------------------------------------------
                # Convert timeline → diagram
                # -------------------------------------------------
                if asset_type == "timeline":
                    asset["type"] = "diagram"
                    description = asset.get("description", "").strip()
                    if description:
                        asset["description"] = (
                            f"Timeline diagram: {description}"
                        )
                    else:
                        asset["description"] = (
                            "Editable timeline diagram"
                        )
                        
                # -------------------------------------------------
                # Convert process → diagram
                # -------------------------------------------------
                elif asset_type == "process":
                    asset["type"] = "diagram"
                    description = asset.get("description", "").strip()
                    if description:
                        asset["description"] = (
                            f"Process diagram: {description}"
                        )
                    else:
                        asset["description"] = (
                            "Editable process diagram"
                        )
                        
                # -------------------------------------------------
                # Convert comparison → diagram
                # -------------------------------------------------
                elif asset_type == "comparison":
                    asset["type"] = "diagram"
                    description = asset.get("description", "").strip()
                    if description:
                        asset["description"] = (
                            f"Comparison diagram: {description}"
                        )
                    else:
                        asset["description"] = (
                            "Editable comparison diagram"
                        )

    # =========================================================
    # SEMANTIC VALIDATION
    # =========================================================

    def _validate_semantics(self, plan: dict) -> None:
        """
        Validate semantic quality of generated presentation content.

        This catches common problems that JSON/schema validation
        cannot detect, such as overly broad scientific claims,
        quantitative charts without cited numerical data, and
        timelines that cannot be rendered with editable labels.
        """

        slides = plan.get("slides", [])
        if not isinstance(slides, list):
            return

        for slide_index, slide in enumerate(
            slides,
            start=1,
        ):
            if not isinstance(slide, dict):
                continue

            self._validate_semantic_key_points(
                slide_index=slide_index,
                slide=slide,
            )
            self._validate_semantic_assets(
                slide_index=slide_index,
                slide=slide,
            )

    def _validate_semantic_key_points(
        self,
        slide_index: int,
        slide: dict,
    ) -> None:
        key_points = slide.get("key_points", [])
        if not isinstance(key_points, list):
            return

        valid_key_points = []
        for point_index, point in enumerate(key_points, start=1):
            if not isinstance(point, dict):
                valid_key_points.append(point)
                continue

            text = point.get("text")
            if not isinstance(text, str):
                valid_key_points.append(point)
                continue

            try:
                self._validate_semantic_key_point(
                    slide_index=slide_index,
                    point_index=point_index,
                    point=point,
                    text=text,
                )
            except ValueError as exc:
                reason = str(exc)
                point["_semantic_rejected"] = True
                point["_semantic_rejection_reason"] = reason
                print(
                    f"[Planner] Rejected slide {slide_index} "
                    f"key point {point_index}: {reason}"
                )
                continue

            point.pop("_semantic_rejected", None)
            point.pop("_semantic_rejection_reason", None)
            valid_key_points.append(point)

        slide["key_points"] = valid_key_points

    def _validate_semantic_key_point(
        self,
        slide_index: int,
        point_index: int,
        point: dict,
        text: str,
    ) -> None:
        point["text"] = self._repair_semantic_key_point(text)
        normalized = point["text"].strip().lower()

        risky_phrases = (
            "revolutionize",
            "revolutionary",
            "game-changing",
            "game changer",
            "transform computing",
            "transform computation",
            "critical for",
            "always faster",
            "dramatically faster",
            "solve all",
            "solves everything",
            "grow exponentially with particles",
            "exponential growth with particles",
        )
        for phrase in risky_phrases:
            if phrase in normalized:
                raise ValueError(
                    f"Slide {slide_index} key point "
                    f"{point_index} contains overly broad "
                    f"or promotional wording: '{phrase}'. "
                    "Use precise, research-grounded wording."
                )

        if (
            "quantum annealing" in normalized
            and "solves optimization problems faster"
            in normalized
        ):
            point["text"] = (
                "Quantum annealing is designed to address certain "
                "optimization problems using quantum effects."
            )
            normalized = point["text"].strip().lower()

        broad_speed_claim = (
            (
                "quantum computers" in normalized
                or "quantum computing" in normalized
            )
            and any(
                phrase in normalized
                for phrase in (
                    "solve problems faster",
                    "solves problems faster",
                    "are faster than classical",
                    "faster than classical computers",
                    "outperform classical computers",
                )
            )
        )
        if broad_speed_claim:
            point["text"] = (
                "Quantum computers may outperform classical "
                "computers for certain problem classes."
            )
            normalized = point["text"].strip().lower()

        if (
            "quantum computers solve" in normalized
            and "faster" in normalized
        ):
            raise ValueError(
                f"Slide {slide_index} key point {point_index} "
                "contains an overly broad quantum speed claim. "
                "Use qualified wording such as "
                "'may outperform classical computers for "
                "certain problem classes'."
            )

        if (
            "quantum computers" in normalized
            and "always faster" in normalized
        ):
            raise ValueError(
                f"Slide {slide_index} key point {point_index} "
                "contains an absolute performance claim."
            )

        positive_ftl_claim = (
            "enables faster-than-light communication"
            in normalized
            or "allows faster-than-light communication"
            in normalized
            or "permits faster-than-light communication"
            in normalized
            or "instant communication across distances"
            in normalized
            or "communicate faster than light"
            in normalized
        )
        if positive_ftl_claim:
            raise ValueError(
                f"Slide {slide_index} key point {point_index} "
                "contains a prohibited faster-than-light "
                "communication claim."
            )

        if (
            "shor" in normalized
            and "breaks encryption" in normalized
        ):
            raise ValueError(
                f"Slide {slide_index} key point {point_index} "
                "contains an overly absolute Shor's algorithm claim."
            )

    def _repair_semantic_key_point(
        self,
        text: str,
    ) -> str:
        normalized = text.strip().lower()

        if (
            "quantum annealing" in normalized
            and "faster" in normalized
        ):
            return (
                "Quantum annealing is designed to address certain "
                "optimization problems using quantum effects."
            )

        if (
            "quantum algorithms" in normalized
            and "factor large integers" in normalized
        ):
            return (
                "Shor's algorithm can factor large integers "
                "efficiently on sufficiently capable "
                "fault-tolerant quantum computers."
            )

        if (
            "quantum computers solve" in normalized
            and "faster" in normalized
        ):
            return (
                "Quantum computers may outperform classical "
                "computers for certain problem classes."
            )

        if (
            "shor" in normalized
            and "breaks encryption" in normalized
        ):
            return (
                "Shor's algorithm can efficiently factor large "
                "integers on sufficiently capable fault-tolerant "
                "quantum computers, threatening some public-key "
                "cryptography."
            )

        return text

    def _validate_semantic_assets(
        self,
        slide_index: int,
        slide: dict,
    ) -> None:
        assets = slide.get("assets", [])
        if not isinstance(assets, list):
            return

        layout = slide.get("layout", "")

        if layout == "timeline" and not assets:
            raise ValueError(
                f"Slide {slide_index} is a timeline "
                "but has no diagram asset."
            )

        for asset in assets:
            if not isinstance(asset, dict):
                continue

            asset_type = asset.get("type")
            description = str(
                asset.get("description", "")
            ).lower()
            visual_spec = asset.get("visual_spec")

            if layout == "timeline":
                if asset_type == "none":
                    raise ValueError(
                        f"Slide {slide_index} is a timeline "
                        "but has no diagram asset."
                    )

                asset["type"] = "diagram"
                if not description.startswith("timeline diagram:"):
                    original_description = asset.get(
                        "description",
                        "Timeline",
                    )
                    asset["description"] = (
                        "Timeline diagram: "
                        f"{original_description}"
                    )

                if isinstance(visual_spec, dict):
                    text_policy = str(
                        visual_spec.get(
                            "text_policy",
                            "",
                        )
                    ).strip().lower()

                    if text_policy in {
                        "no text",
                        "none",
                        "no labels",
                    }:
                        visual_spec["text_policy"] = (
                            "editable labels"
                        )
                continue

            if asset_type != "chart":
                continue

            composition = ""
            if isinstance(visual_spec, dict):
                composition = str(
                    visual_spec.get(
                        "composition",
                        "",
                    )
                ).lower()

            quantitative_words = (
                "bar chart",
                "line chart",
                "percentage",
                "percent",
                "measurement",
                "numerical",
                "numeric",
                "values",
                "performance",
                "speed difference",
            )

            combined = description + " " + composition
            if not any(
                word in combined
                for word in quantitative_words
            ):
                continue

            asset["type"] = "diagram"
            original_description = asset.get(
                "description",
                "comparison",
            )
            asset["description"] = (
                "Qualitative comparison diagram: "
                f"{original_description}"
            )

            if isinstance(visual_spec, dict):
                visual_spec["subject"] = (
                    "Qualitative algorithm comparison"
                )
                visual_spec["composition"] = (
                    "Two-column comparison of classical and quantum "
                    "approaches using qualitative labels only; no "
                    "numerical performance values."
                )
                visual_spec["style"] = (
                    "Clean scientific comparison diagram"
                )
                visual_spec["text_policy"] = "editable labels"

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

            for point_index, point in enumerate(
                slide["key_points"],
                start=1,
            ):

                self._validate_key_point(
                    slide_index=index,
                    point_index=point_index,
                    point=point,
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

            for asset_index, asset in enumerate(
                slide["assets"],
                start=1,
            ):

                self._validate_asset(
                    slide_index=index,
                    asset_index=asset_index,
                    asset=asset,
                )

    # =========================================================
    # VALIDATE KEY POINT
    # =========================================================

    def _validate_key_point(
        self,
        slide_index: int,
        point_index: int,
        point: dict,
    ):
        if not isinstance(point, dict):
            raise ValueError(
                f"Slide {slide_index}, key_point {point_index} "
                "must be a dictionary."
            )

        if "text" not in point or not isinstance(point["text"], str):
            raise ValueError(
                f"Slide {slide_index}, key_point {point_index} "
                "is missing a valid 'text' string."
            )

        if not point["text"].strip():
            raise ValueError(
                f"Slide {slide_index}, key_point {point_index} "
                "'text' cannot be empty."
            )

        if "sources" not in point or not isinstance(point["sources"], list):
            raise ValueError(
                f"Slide {slide_index}, key_point {point_index} "
                "is missing a valid 'sources' list."
            )

        for source_id in point["sources"]:
            if not isinstance(source_id, str):
                raise ValueError(
                    f"Slide {slide_index}, key_point {point_index} "
                    "contains a non-string source ID."
                )

            if not source_id.strip():
                raise ValueError(
                    f"Slide {slide_index}, key_point {point_index} "
                    "contains an empty source ID."
                )

    # =========================================================
    # VALIDATE ASSET
    # =========================================================

    def _validate_asset(
        self,
        slide_index: int,
        asset_index: int,
        asset: dict,
    ):
        if not isinstance(asset, dict):
            raise ValueError(
                f"Slide {slide_index}, asset {asset_index} "
                "must be a dictionary."
            )
            
        if "type" not in asset or asset["type"] not in self.ALLOWED_ASSET_TYPES:
            raise ValueError(
                f"Slide {slide_index}, asset {asset_index} has an unsupported "
                f"asset type: {asset.get('type')}."
            )

        if asset["type"] == "none":
            return

        visual_spec = asset.get("visual_spec")
        if not isinstance(visual_spec, dict):
            raise ValueError(
                f"Slide {slide_index}, asset {asset_index} is missing "
                "a valid 'visual_spec' object."
            )

        required_visual_fields = (
            "purpose",
            "subject",
            "composition",
            "style",
            "color_palette",
            "must_show",
            "must_avoid",
            "text_policy",
        )
        for field in required_visual_fields:
            if field not in visual_spec:
                raise ValueError(
                    f"Slide {slide_index}, asset {asset_index} "
                    f"visual_spec is missing required field '{field}'."
                )

    def print_plan(self, plan: dict):
        """
        Print a human-readable presentation plan.
        """

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
                if isinstance(point, dict):
                    print(f"  - {point['text']}")

                    sources = point.get("sources", [])

                    if sources:
                        print(
                            "    Sources: "
                            + ", ".join(sources)
                        )
                else:
                    # Backward compatibility
                    print(f"  - {point}")

            print("Assets:")

            for asset in slide["assets"]:
                print(
                    f"  - [{asset.get('type', 'unknown')}] "
                    f"{asset.get('description', '')}"
                )

                source_ids = asset.get(
                    "source_ids",
                    [],
                )

                if source_ids:
                    print(
                        "    Sources: "
                        + ", ".join(source_ids)
                    )

                visual_spec = asset.get(
                    "visual_spec"
                )

                if isinstance(visual_spec, dict):
                    print(
                        "    Visual purpose: "
                        f"{visual_spec.get('purpose', '')}"
                    )

                    print(
                        "    Subject: "
                        f"{visual_spec.get('subject', '')}"
                    )

                    print(
                        "    Composition: "
                        f"{visual_spec.get('composition', '')}"
                    )

                    print(
                        "    Style: "
                        f"{visual_spec.get('style', '')}"
                    )

                    print(
                        "    Colors: "
                        + ", ".join(
                            visual_spec.get("color_palette", [])
                        )
                    )

                    print("    Must show:")

                    for item in visual_spec.get("must_show", []):
                        print(f"      - {item}")

                    print("    Must avoid:")

                    for item in visual_spec.get("must_avoid", []):
                        print(f"      - {item}")

                    print(
                        "    Text policy: "
                        f"{visual_spec.get('text_policy', '')}"
                    )