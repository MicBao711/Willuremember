"""Promotion: a cache entry that is used often and stays valid becomes a concept unit.

Two modes after promotion:

* KEEP_DEPS — the concept remembers what it is made of and is rebuilt (lazily) when a
  part changes. Always consistent with its parts; changes ripple further.
* CUT_DEPS  — the concept becomes independent, like *breakfast* no longer meaning
  "break" + "fast". Stable and cheap; may drift away from its parts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .cache import CacheEntry, CompositionCache
from .units import UnitKind, UnitStore


class PromotionMode(str, Enum):
    KEEP_DEPS = "keep_deps"
    CUT_DEPS = "cut_deps"


@dataclass
class PromotionPolicy:
    min_hits: int = 20
    min_stable_hits: int = 10
    max_errors: int = 0
    mode: PromotionMode = PromotionMode.CUT_DEPS

    def eligible(self, entry: CacheEntry) -> bool:
        return (
            entry.promoted_to is None
            and entry.hits >= self.min_hits
            and entry.stable_hits >= self.min_stable_hits
            and entry.errors <= self.max_errors
        )


def promote(
    store: UnitStore,
    entry: CacheEntry,
    unit_id: str,
    mode: PromotionMode | str = PromotionMode.CUT_DEPS,
    domain: str = "concepts",
) -> str:
    """Create a concept unit from a cache entry; returns the new unit id."""
    mode = PromotionMode(mode)
    if not entry.deps:
        raise ValueError("an entry without dependencies has nothing to promote")
    level = 1 + max(store.level(u) for u in entry.deps)
    meta = {"origin": dict(entry.deps), "mode": mode.value, "promoted_from": entry.key}
    store.add(
        unit_id,
        entry.value,
        domain=domain,
        level=level,
        kind=UnitKind.CONCEPT,
        deps=entry.deps if mode is PromotionMode.KEEP_DEPS else None,
        spec=entry.spec,
        meta=meta,
    )
    entry.promoted_to = unit_id
    return unit_id


def promote_eligible(
    store: UnitStore,
    cache: CompositionCache,
    policy: PromotionPolicy,
    name_fn=lambda entry: f"concept:{entry.key}",
    domain: str = "concepts",
) -> list[str]:
    """Promote every eligible entry; returns the new concept ids."""
    created = []
    for entry in cache:
        if policy.eligible(entry):
            created.append(promote(store, entry, name_fn(entry), policy.mode, domain))
    return created
