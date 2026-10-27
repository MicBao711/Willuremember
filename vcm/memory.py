"""The VCM facade: units + similarity pointers + role binding + versioned cache + promotion."""

from __future__ import annotations

from typing import Hashable, Iterable, Mapping

import numpy as np

from .cache import CacheEntry, CompositionCache, StalePolicy
from .compose import Composer, CompositionSpec
from .conditional import ConditionalKnowledge
from .promotion import PromotionMode, PromotionPolicy, promote
from .similarity import Pointer, SimilarityIndex, SimilarityProfile
from .units import Unit, UnitKind, UnitStore
from .vsa import VSA, ItemMemory, cosine


class VCM:
    def __init__(
        self,
        dim: int = 1024,
        seed: int = 0,
        stale_policy: StalePolicy | str = StalePolicy.RECOMPOSE,
        promotion: PromotionPolicy | None = None,
    ):
        self.vsa = VSA(dim, seed)
        self.store = UnitStore()
        self.similarity = SimilarityIndex()
        self.composer = Composer(self.vsa)
        self.knowledge = ConditionalKnowledge()
        self.promotion = promotion
        self.cache = CompositionCache(
            builder=self._build,
            version_of=self.fresh_version,
            policy=StalePolicy(stale_policy),
            on_insert=self._pin_entry,
            on_remove=self._unpin_entry,
        )
        self.concept_refreshes = 0

    # -- units --------------------------------------------------------------
    def add(self, name: str, vector: np.ndarray | None = None, domain: str = "default") -> Unit:
        vec = self.vsa.random(name) if vector is None else np.asarray(vector, float)
        return self.store.add(name, vec, domain=domain)

    def update(self, name: str, vector: np.ndarray) -> int:
        return self.store.update(name, np.asarray(vector, float))

    def vector(self, name: str, version: int | None = None) -> np.ndarray:
        return self.store.snapshot(name, version).payload

    # -- similarity ---------------------------------------------------------
    def relate(self, a: str, b: str, **scores: float) -> SimilarityProfile:
        return self.similarity.relate(a, b, scores).profile

    def similar(
        self, name: str, rank_by: str | None = None, **requirements: float
    ) -> list[tuple[str, SimilarityProfile]]:
        return self.similarity.neighbors(name, requirements or None, rank_by)

    # -- composition --------------------------------------------------------
    def compose(
        self,
        key: Hashable,
        bindings: Mapping[str, str],
        similarity: Mapping[str, Mapping[str, float]] | None = None,
    ) -> np.ndarray:
        spec = CompositionSpec.of(bindings, similarity)
        entry = self.cache.peek(key)
        if entry is not None and entry.spec != spec:
            self.cache.invalidate(key)
        value = self.cache.get(key, spec)
        if self.promotion is not None:
            self._maybe_promote(key)
        return value

    def lookup(self, key: Hashable) -> np.ndarray:
        """Cached composition by key, validated against current versions."""
        value = self.cache.get(key)
        if self.promotion is not None:
            self._maybe_promote(key)
        return value

    def pointers(self, key: Hashable) -> dict[str, Pointer]:
        entry = self.cache.entry(key)
        return {
            role: Pointer(unit, entry.deps[unit], role, entry.spec.profile(role))
            for role, unit in entry.spec.bindings
        }

    def _build(self, spec: CompositionSpec) -> tuple[np.ndarray, dict[str, int]]:
        fillers, deps = {}, {}
        for role, unit_id in spec.bindings:
            deps[unit_id] = self.fresh_version(unit_id)
            fillers[role] = self.store.snapshot(unit_id).payload
        return self.composer.compose(fillers), deps

    def fresh_version(self, unit_id: str) -> int:
        """Current version; a keep-deps concept is first rebuilt if a part moved on.

        Dependencies point strictly to lower levels, so this recursion terminates.
        """
        unit = self.store.get(unit_id)
        if unit.kind is UnitKind.CONCEPT and unit.deps:
            if any(self.fresh_version(d) != v for d, v in unit.deps.items()):
                value, deps = self._build(unit.spec)
                self.store.update(unit_id, value)
                self.store.set_deps(unit_id, deps)
                self.concept_refreshes += 1
        return unit.version

    def _pin_entry(self, entry: CacheEntry) -> None:
        for unit_id, version in entry.deps.items():
            self.store.pin(unit_id, version)

    def _unpin_entry(self, entry: CacheEntry) -> None:
        for unit_id, version in entry.deps.items():
            self.store.unpin(unit_id, version)

    # -- decoding -----------------------------------------------------------
    def item_memory(self, candidates: Iterable[str] | None = None) -> ItemMemory:
        """Current payloads, plus any older snapshot still pinned (labelled unit@vN)."""
        allowed = set(candidates) if candidates is not None else None
        mem = ItemMemory()
        for unit in self.store:
            if allowed is not None and unit.id not in allowed:
                continue
            mem.add(unit.id, unit.payload)
        for unit_id, version in self.store.retained_old_snapshots():
            if allowed is None or unit_id in allowed:
                mem.add(f"{unit_id}@v{version}", self.store.snapshot(unit_id, version).payload)
        return mem

    def decode(
        self,
        source: Hashable | np.ndarray,
        role: str,
        candidates: Iterable[str] | None = None,
    ) -> tuple[str, float]:
        vector = source if isinstance(source, np.ndarray) else self.cache.entry(source).value
        noisy = self.composer.extract(vector, role)
        return self.item_memory(candidates).cleanup(noisy)

    # -- promotion ----------------------------------------------------------
    def _maybe_promote(self, key: Hashable) -> str | None:
        entry = self.cache.peek(key)
        if entry is not None and self.promotion.eligible(entry):
            return promote(self.store, entry, f"concept:{key}", self.promotion.mode)
        return None

    def promote(
        self, key: Hashable, name: str | None = None,
        mode: PromotionMode | str = PromotionMode.CUT_DEPS,
    ) -> str:
        return promote(self.store, self.cache.entry(key), name or f"concept:{key}", mode)

    def drift(self, concept: str) -> float:
        """1 - cos(concept, what its recipe would give today). 0 means no drift."""
        unit = self.store.get(concept)
        if unit.spec is None:
            raise ValueError(f"{concept!r} has no recipe")
        fillers = {
            role: self.store.snapshot(u, self.fresh_version(u)).payload
            for role, u in unit.spec.bindings
        }
        return 1.0 - cosine(unit.payload, self.composer.compose(fillers))
