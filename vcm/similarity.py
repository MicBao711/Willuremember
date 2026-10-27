"""Multi-dimensional similarity and snapshot pointers.

A single similarity number cannot tell a true cognate from a false friend: Spanish
*salir* ("go out") and French *salir* ("make dirty") are identical in form and unrelated
in meaning. A profile keeps the dimensions apart so each use can ask for the one it needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Mapping


@dataclass(frozen=True)
class SimilarityProfile:
    scores: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        for dim, s in self.scores.items():
            if not -1.0 <= float(s) <= 1.0:
                raise ValueError(f"similarity on {dim!r} must be in [-1, 1], got {s}")
        object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))

    def __getitem__(self, dim: str) -> float:
        return self.scores[dim]

    def get(self, dim: str, default: float = 0.0) -> float:
        return self.scores.get(dim, default)

    @property
    def dims(self) -> tuple[str, ...]:
        return tuple(self.scores)

    def blended(self, weights: Mapping[str, float] | None = None) -> float:
        """Collapse to one number — what a single-similarity design would see."""
        weights = weights or {d: 1.0 for d in self.scores}
        total = sum(weights.values())
        return sum(w * self.get(d) for d, w in weights.items()) / total if total else 0.0

    def satisfies(self, requirements: Mapping[str, float]) -> bool:
        return all(self.get(d, 0.0) >= t for d, t in requirements.items())

    def merged(self, other: "SimilarityProfile") -> "SimilarityProfile":
        return SimilarityProfile({**self.scores, **other.scores})

    def to_dict(self) -> dict[str, float]:
        return dict(self.scores)


@dataclass(frozen=True)
class Pointer:
    """Downward dependency edge to a frozen snapshot of `target`, annotated per dimension."""

    target: str
    version: int
    role: str | None = None
    similarity: SimilarityProfile = field(default_factory=SimilarityProfile)


@dataclass(frozen=True)
class Relation:
    a: str
    b: str
    profile: SimilarityProfile


class SimilarityIndex:
    """Lateral similarity records between units, queried one dimension at a time."""

    def __init__(self):
        self._edges: dict[str, dict[str, SimilarityProfile]] = {}

    def relate(self, a: str, b: str, profile: SimilarityProfile | Mapping[str, float]) -> Relation:
        if a == b:
            raise ValueError("a unit is trivially similar to itself")
        if not isinstance(profile, SimilarityProfile):
            profile = SimilarityProfile(profile)
        self._edges.setdefault(a, {})[b] = profile
        self._edges.setdefault(b, {})[a] = profile
        return Relation(a, b, profile)

    def profile(self, a: str, b: str) -> SimilarityProfile:
        return self._edges.get(a, {}).get(b, SimilarityProfile())

    def neighbors(
        self,
        unit_id: str,
        requirements: Mapping[str, float] | None = None,
        rank_by: str | None = None,
        candidates: Iterable[str] | None = None,
    ) -> list[tuple[str, SimilarityProfile]]:
        """Neighbours satisfying per-dimension thresholds, ranked by one dimension.

        With `rank_by=None` the ranking uses the blended score, which is the
        single-number behaviour this index exists to avoid; it is kept for comparison.
        """
        allowed = set(candidates) if candidates is not None else None
        out = []
        for other, prof in self._edges.get(unit_id, {}).items():
            if allowed is not None and other not in allowed:
                continue
            if requirements and not prof.satisfies(requirements):
                continue
            out.append((other, prof))
        key = (lambda kv: kv[1].get(rank_by)) if rank_by else (lambda kv: kv[1].blended())
        return sorted(out, key=key, reverse=True)
