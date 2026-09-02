"""Importance-scored archiving: bounded growth as demotion, never deletion.

Rusu et al. (arXiv:2608.28978) show ~10% of a conversational graph prunes at
no measured quality cost, scoring nodes by recency, access frequency, degree
centrality and age. We keep the score and reject the deletion. An archived
entity is withheld from retrieval and community builds exactly as a dormant
one is -- excluded, reversible, and fully preserved for audit -- because a
system whose pitch is "history is kept" cannot quietly erase it.

The score's four signals come from what 1.10.0 already records:
  recency    last retrieval hit on the entity (coverage telemetry)
  frequency  number of retrieval hits         (coverage telemetry)
  centrality relation degree                  (one SQL aggregate)
  age        belief-time age of the entity     (created_at)

Retention therefore requires record_retrieval_events; without it the recency
and frequency terms are undefined, and apply_retention refuses rather than
scoring on structure alone.
"""
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _parse(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass
class RetentionReport:
    scored: int = 0
    archived: int = 0
    threshold: float = 0.10
    dry_run: bool = True
    archived_ids: List[str] = field(default_factory=list)
    score_min: float = 0.0
    score_max: float = 0.0


class RetentionManager:
    # Defaults are Rusu et al.'s weights; half-lives likewise (recency 90d,
    # age 1000 turns -> here, days).
    W_RECENCY, W_FREQ, W_CENTRALITY, W_AGE = 0.35, 0.25, 0.20, 0.20
    RECENCY_HALFLIFE_DAYS = 90.0
    AGE_HALFLIFE_DAYS = 365.0

    def __init__(self, store):
        self.store = store
        self.client = store.client
        self.realm = store.realm

    def _score(self, now, hits, last_hit, degree, created) -> float:
        # recency: exponential decay from last retrieval; never hit -> 0
        last = _parse(last_hit)
        if last is None:
            recency = 0.0
        else:
            days = max(0.0, (now - last).total_seconds() / 86400.0)
            recency = 0.5 ** (days / self.RECENCY_HALFLIFE_DAYS)
        frequency = math.log1p(hits) / math.log1p(hits + 4)  # 0..~1, saturating
        centrality = math.log1p(degree) / math.log1p(degree + 4)
        c = _parse(created)
        if c is None:
            age = 1.0
        else:
            age_days = max(0.0, (now - c).total_seconds() / 86400.0)
            age = 0.5 ** (age_days / self.AGE_HALFLIFE_DAYS)
        return (self.W_RECENCY * recency + self.W_FREQ * frequency +
                self.W_CENTRALITY * centrality + self.W_AGE * age)

    async def apply(self, threshold: float = 0.10, dry_run: bool = True,
                    space: Optional[str] = None) -> RetentionReport:
        if not self.store.config.record_retrieval_events:
            raise ValueError(
                "apply_retention requires record_retrieval_events: without "
                "telemetry the recency and frequency terms are undefined, and "
                "scoring on structure alone would archive well-connected but "
                "never-asked-about entities. Enable telemetry first.")
        now = datetime.now(timezone.utc)
        stats = await self.store.entity_retention_stats(space=space)
        report = RetentionReport(threshold=threshold, dry_run=dry_run)
        scores = []
        to_archive = []
        for row in stats:
            if row.get("archived_at") or row.get("dormant_since"):
                continue                       # already withheld; leave as-is
            s = self._score(now, row["hits"], row["last_hit"],
                            row["degree"], row["created_at"])
            scores.append(s)
            report.scored += 1
            if s < threshold:
                to_archive.append(row["id"])
        if scores:
            report.score_min, report.score_max = min(scores), max(scores)
        report.archived_ids = to_archive
        report.archived = len(to_archive)
        if not dry_run and to_archive:
            await self.store.set_entities_archived(to_archive, now.isoformat(),
                                                   space=space)
        return report

    async def restore(self, entity_ids: List[str],
                      space: Optional[str] = None) -> int:
        return await self.store.clear_entities_archived(entity_ids, space=space)
