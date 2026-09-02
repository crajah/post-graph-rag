"""Community report generation.

A report is an LLM summary of one cluster of the entity graph. Corpus-level
questions are answered from these rather than from individual passages, because
the answer to "what are the main themes?" exists in no single passage.
"""
import json
import logging
import re
from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from post_graph_rag.errors import ExtractionError
from post_graph_rag.llm import LLMService

logger = logging.getLogger(__name__)


class Finding(BaseModel):
    summary: str = Field(..., description="Short headline for this finding")
    explanation: str = Field(..., description="Supporting explanation grounded in the listed entities and relations")


class CommunityReport(BaseModel):
    title: str = Field(..., description="Short, specific name for what this community is about")
    summary: str = Field(..., description="Executive summary of the community as a whole")
    findings: List[Finding] = Field(default_factory=list, description="Key insights within the community")
    rating: float = Field(
        default=5.0,
        description="0-10 importance of this community to understanding the corpus",
    )


REPORT_SYSTEM_PROMPT = """You are an analyst summarising one community of a knowledge graph.

You are given the entities in the community and the relations between them. Write a
report that lets a reader understand what this community is about without reading
the source documents.

RULES:
- Ground every statement in the supplied entities and relations. Do not introduce
  outside knowledge, and do not speculate beyond what the relations state.
- A relation marked NOT means the text explicitly denies it. Never report a denied
  relation as if it holds.
- Title: specific and concrete. Name the actual subject, not "Community 3".
- Summary: what this cluster is, and why its members belong together.
- Findings: the most significant insights, each with a short headline and an
  explanation referencing the entities involved.
- Rating: 0-10 for how important this community is to understanding the corpus.

Return your response strictly according to the required schema.
"""


def render_community(
    entities: Sequence[Dict[str, object]],
    relations: Sequence[Dict[str, object]],
    max_entities: int = 60,
    max_relations: int = 80,
) -> str:
    """Render a community's members and relations as prompt input."""
    lines = ["Entities:"]
    for e in entities[:max_entities]:
        name = e.get("name") or ""
        etype = e.get("type") or "Concept"
        desc = (e.get("description") or "")
        lines.append(f"- {name} ({etype}): {desc}"[:400])

    lines.append("")
    lines.append("Relations:")
    for r in relations[:max_relations]:
        neg = "NOT " if r.get("negated") else ""
        weight = r.get("weight", 1)
        desc = (r.get("description") or "")
        lines.append(
            f"- ({r.get('src')}) --[{neg}{r.get('predicate')}]--> ({r.get('tgt')})"
            f" (weight={weight}): {desc}"[:400]
        )
    return "\n".join(lines)


class CommunityReporter:
    """Turns a community's subgraph into a :class:`CommunityReport`."""

    def __init__(self, llm_service: LLMService, system_prompt: Optional[str] = None):
        self.llm_service = llm_service
        self.system_prompt = system_prompt or REPORT_SYSTEM_PROMPT

    async def summarise_reports(self, children: Sequence[CommunityReport]) -> CommunityReport:
        """Synthesise a parent report from child reports.

        A theme-level report should synthesise findings, not re-derive them
        from raw relations -- and summarising reports rather than subgraphs is
        what keeps the hierarchy's LLM cost proportional to the number of
        clusters, not to the graph.
        """
        lines = []
        for c in children:
            finds = "; ".join(f"{f.summary}: {f.explanation}" for f in c.findings)
            lines.append(f"- {c.title} (importance {c.rating}): {c.summary} {finds}".strip())
        pseudo_entities = [{"name": c.title, "type": "Community",
                           "description": c.summary} for c in children]
        pseudo_relations = [{"src": "theme", "tgt": c.title, "predicate": "includes",
                            "description": line, "weight": c.rating, "negated": False}
                           for c, line in zip(children, lines)]
        return await self.summarise(pseudo_entities, pseudo_relations)

    async def summarise(
        self,
        entities: Sequence[Dict[str, object]],
        relations: Sequence[Dict[str, object]],
    ) -> CommunityReport:
        """Generate a report, raising when the model returns nothing usable."""
        body = render_community(entities, relations)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": body},
        ]

        result = await self.llm_service.chat_completion(messages, response_format=CommunityReport)
        if isinstance(result, str):
            result = self._parse(result)

        if not isinstance(result, CommunityReport) or not (result.title or result.summary):
            raise ExtractionError("LLM returned no usable community report.")

        # A report with no summary is not worth storing or embedding.
        if not result.summary.strip():
            raise ExtractionError("Community report had an empty summary.")
        return result

    @staticmethod
    def _parse(raw: str) -> Optional[CommunityReport]:
        if not raw or not raw.strip():
            return None
        cleaned = re.sub(r"^\s*```(?:json)?|```\s*$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            return CommunityReport(**json.loads(cleaned))
        except Exception as e:
            logger.debug("Could not parse community report JSON: %s", e)
            return None


def report_to_text(report: CommunityReport) -> str:
    """Flatten a report into the text that gets embedded and shown to the model."""
    parts = [report.title, report.summary]
    parts.extend(f"{f.summary}: {f.explanation}" for f in report.findings)
    return "\n".join(p for p in parts if p)
