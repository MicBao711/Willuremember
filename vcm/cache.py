"""Composition cache with version-stamped lazy invalidation.

Each entry records the version of every unit it was built from. Nothing is pushed to
the cache when a unit changes; instead, every read compares the recorded versions with
the current ones ("check on use, repair when stale"). This avoids the cascade of eager
recomputation, and coalesces several updates into a single rebuild.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Hashable, Iterator

Builder = Callable[[Any], "tuple[Any, dict[str, int]]"]


class StalePolicy(str, Enum):
    RECOMPOSE = "recompose"  # rebuild in place, keep the entry's usage history
    EVICT = "evict"          # drop the entry (and its history), rebuild as a fresh miss


@dataclass
class CacheEntry:
    key: Hashable
    spec: Any
    value: Any
    deps: dict[str, int]
    hits: int = 0
    stable_hits: int = 0      # valid hits since the last invalidation
    invalidations: int = 0
    errors: int = 0           # negative external feedback
    confirmations: int = 0    # positive external feedback
    created_tick: int = 0
    last_tick: int = 0
    promoted_to: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    stale: int = 0
    composes: int = 0
    evictions: int = 0
    version_checks: int = 0
    feedback_errors: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


class CompositionCache:
    def __init__(
        self,
        builder: Builder,
        version_of: Callable[[str], int],
        policy: StalePolicy = StalePolicy.RECOMPOSE,
        capacity: int | None = None,
        on_insert: Callable[[CacheEntry], None] | None = None,
        on_remove: Callable[[CacheEntry], None] | None = None,
    ):
        self._build = builder
        self._version_of = version_of
        self.policy = StalePolicy(policy)
        self.capacity = capacity
        self._on_insert = on_insert or (lambda e: None)
        self._on_remove = on_remove or (lambda e: None)
        self._entries: "OrderedDict[Hashable, CacheEntry]" = OrderedDict()
        self.stats = CacheStats()
        self._tick = 0

    # -- container protocol -------------------------------------------------
    def __contains__(self, key: Hashable) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[CacheEntry]:
        return iter(list(self._entries.values()))

    def entry(self, key: Hashable) -> CacheEntry:
        return self._entries[key]

    # -- core ---------------------------------------------------------------
    def is_fresh(self, entry: CacheEntry) -> bool:
        for unit_id, version in entry.deps.items():
            self.stats.version_checks += 1
            if self._version_of(unit_id) != version:
                return False
        return True

    def stale_units(self, entry: CacheEntry) -> dict[str, tuple[int, int]]:
        """{unit: (recorded, current)} for every dependency that moved on."""
        out = {}
        for unit_id, version in entry.deps.items():
            current = self._version_of(unit_id)
            if current != version:
                out[unit_id] = (version, current)
        return out

    def get(self, key: Hashable, spec: Any = None) -> Any:
        """Return a value that is consistent with the current unit versions."""
        self._tick += 1
        entry = self._entries.get(key)
        if entry is None:
            if spec is None:
                raise KeyError(f"{key!r} is not cached and no spec was given")
            self.stats.misses += 1
            return self._insert(key, spec).value
        if self.is_fresh(entry):
            self.stats.hits += 1
            entry.hits += 1
            entry.stable_hits += 1
            entry.last_tick = self._tick
            self._entries.move_to_end(key)
            return entry.value
        self.stats.stale += 1
        if self.policy is StalePolicy.EVICT:
            spec = entry.spec if spec is None else spec
            self._remove(key)
            self.stats.evictions += 1
            self.stats.misses += 1
            return self._insert(key, spec).value
        self._rebuild(entry, entry.spec if spec is None else spec)
        entry.hits += 1
        return entry.value

    def peek(self, key: Hashable) -> CacheEntry | None:
        """Look without validating or counting a hit."""
        return self._entries.get(key)

    def refresh(self, key: Hashable) -> bool:
        """Rebuild `key` now if it is stale (used by eager maintenance). Returns rebuilt?"""
        entry = self._entries[key]
        if self.is_fresh(entry):
            return False
        self.stats.stale += 1
        self._rebuild(entry, entry.spec)
        return True

    def invalidate(self, key: Hashable) -> None:
        if key in self._entries:
            self._remove(key)
            self.stats.evictions += 1

    def feedback(self, key: Hashable, ok: bool) -> None:
        """External judgement (a person or another model) on a cached result."""
        entry = self._entries.get(key)
        if entry is None:
            return
        if ok:
            entry.confirmations += 1
            return
        entry.errors += 1
        entry.stable_hits = 0
        self.stats.feedback_errors += 1
        self._remove(key)
        self.stats.evictions += 1

    def dependents_of(self, unit_id: str) -> list[Hashable]:
        return [k for k, e in self._entries.items() if unit_id in e.deps]

    # -- internals ----------------------------------------------------------
    def _insert(self, key: Hashable, spec: Any) -> CacheEntry:
        value, deps = self._build(spec)
        self.stats.composes += 1
        entry = CacheEntry(
            key=key, spec=spec, value=value, deps=dict(deps),
            created_tick=self._tick, last_tick=self._tick,
        )
        self._entries[key] = entry
        self._on_insert(entry)
        self._enforce_capacity()
        return entry

    def _rebuild(self, entry: CacheEntry, spec: Any) -> None:
        self._on_remove(entry)
        value, deps = self._build(spec)
        self.stats.composes += 1
        entry.spec, entry.value, entry.deps = spec, value, dict(deps)
        entry.invalidations += 1
        entry.stable_hits = 0
        entry.last_tick = self._tick
        self._on_insert(entry)

    def _remove(self, key: Hashable) -> None:
        entry = self._entries.pop(key)
        self._on_remove(entry)

    def _enforce_capacity(self) -> None:
        if self.capacity is None:
            return
        while len(self._entries) > self.capacity:
            oldest = next(iter(self._entries))
            self._remove(oldest)
            self.stats.evictions += 1
