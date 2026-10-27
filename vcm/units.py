"""Versioned knowledge units with copy-on-write snapshots.

* Every unit lives in exactly one *domain* partition; an update touches one unit only.
* Every update bumps the unit's version by one.
* A snapshot is an immutable copy of one (unit, version). Pointers pin snapshots, so a
  referrer keeps seeing the value it was built from even after the original changes.
* Snapshots are reference-counted: an old version is dropped once nobody pins it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator, Mapping

import numpy as np


class DAGViolation(ValueError):
    """Raised when a dependency would not point strictly to a lower level."""


class UnitKind(str, Enum):
    BASE = "base"
    CONCEPT = "concept"


def freeze(payload: Any) -> Any:
    """Default copy-on-write: deep copy, and make numpy arrays read-only."""
    if isinstance(payload, np.ndarray):
        out = payload.copy()
        out.setflags(write=False)
        return out
    return copy.deepcopy(payload)


@dataclass(frozen=True)
class Snapshot:
    unit_id: str
    version: int
    payload: Any


@dataclass
class Unit:
    id: str
    domain: str
    payload: Any
    version: int = 1
    level: int = 0
    kind: UnitKind = UnitKind.BASE
    meta: dict = field(default_factory=dict)
    # For concepts that keep their dependencies: what they are composed of.
    deps: dict[str, int] = field(default_factory=dict)
    spec: Any = None


class UnitStore:
    """Holds units partitioned by domain, with versioning and snapshot pinning."""

    def __init__(self, freeze_fn: Callable[[Any], Any] = freeze):
        self._freeze = freeze_fn
        self._units: dict[str, Unit] = {}
        self._domains: dict[str, set[str]] = {}
        self._snapshots: dict[tuple[str, int], Snapshot] = {}
        self._refs: dict[tuple[str, int], int] = {}
        self._listeners: list[Callable[[str, int, int], None]] = []
        self.update_count = 0

    # -- basic access -------------------------------------------------------
    def __contains__(self, unit_id: str) -> bool:
        return unit_id in self._units

    def __iter__(self) -> Iterator[Unit]:
        return iter(self._units.values())

    def __len__(self) -> int:
        return len(self._units)

    def get(self, unit_id: str) -> Unit:
        try:
            return self._units[unit_id]
        except KeyError:
            raise KeyError(f"unknown unit {unit_id!r}") from None

    def version(self, unit_id: str) -> int:
        return self.get(unit_id).version

    def level(self, unit_id: str) -> int:
        return self.get(unit_id).level

    def domain(self, name: str) -> set[str]:
        return set(self._domains.get(name, set()))

    @property
    def domains(self) -> list[str]:
        return sorted(self._domains)

    # -- mutation -----------------------------------------------------------
    def add(
        self,
        unit_id: str,
        payload: Any,
        domain: str = "default",
        *,
        level: int = 0,
        kind: UnitKind = UnitKind.BASE,
        deps: Mapping[str, int] | None = None,
        spec: Any = None,
        meta: Mapping | None = None,
    ) -> Unit:
        if unit_id in self._units:
            raise KeyError(f"unit {unit_id!r} already exists")
        deps = dict(deps or {})
        for dep in deps:
            if self.level(dep) >= level:
                raise DAGViolation(
                    f"{unit_id!r} (level {level}) cannot depend on {dep!r} "
                    f"(level {self.level(dep)})"
                )
        unit = Unit(
            id=unit_id,
            domain=domain,
            payload=self._freeze(payload),
            level=level,
            kind=kind,
            deps=deps,
            spec=spec,
            meta=dict(meta or {}),
        )
        self._units[unit_id] = unit
        self._domains.setdefault(domain, set()).add(unit_id)
        self._record_snapshot(unit)
        return unit

    def update(self, unit_id: str, payload: Any, **meta) -> int:
        """Replace a unit's payload; returns the new version."""
        unit = self.get(unit_id)
        old = unit.version
        self._release_if_unpinned(unit_id, old)
        unit.payload = self._freeze(payload)
        unit.version = old + 1
        unit.meta.update(meta)
        self._record_snapshot(unit)
        self.update_count += 1
        for listener in self._listeners:
            listener(unit_id, old, unit.version)
        return unit.version

    def set_deps(self, unit_id: str, deps: Mapping[str, int]) -> None:
        unit = self.get(unit_id)
        for dep in deps:
            if self.level(dep) >= unit.level:
                raise DAGViolation(f"{unit_id!r} cannot depend on {dep!r}")
        unit.deps = dict(deps)

    def subscribe(self, listener: Callable[[str, int, int], None]) -> None:
        """Call listener(unit_id, old_version, new_version) after each update."""
        self._listeners.append(listener)

    # -- snapshots ----------------------------------------------------------
    def _record_snapshot(self, unit: Unit) -> None:
        key = (unit.id, unit.version)
        self._snapshots[key] = Snapshot(unit.id, unit.version, unit.payload)

    def _release_if_unpinned(self, unit_id: str, version: int) -> None:
        key = (unit_id, version)
        if self._refs.get(key, 0) == 0:
            self._snapshots.pop(key, None)

    def snapshot(self, unit_id: str, version: int | None = None) -> Snapshot:
        unit = self.get(unit_id)
        version = unit.version if version is None else version
        try:
            return self._snapshots[(unit_id, version)]
        except KeyError:
            raise KeyError(
                f"snapshot {unit_id}@v{version} was not pinned and has been released"
            ) from None

    def pin(self, unit_id: str, version: int | None = None) -> Snapshot:
        snap = self.snapshot(unit_id, version)
        key = (snap.unit_id, snap.version)
        self._refs[key] = self._refs.get(key, 0) + 1
        return snap

    def unpin(self, unit_id: str, version: int) -> None:
        key = (unit_id, version)
        count = self._refs.get(key, 0)
        if count <= 0:
            raise KeyError(f"{unit_id}@v{version} is not pinned")
        if count == 1:
            del self._refs[key]
            if self.get(unit_id).version != version:
                self._snapshots.pop(key, None)
        else:
            self._refs[key] = count - 1

    def pinned_versions(self, unit_id: str) -> list[int]:
        return sorted(v for (u, v) in self._refs if u == unit_id)

    @property
    def snapshot_count(self) -> int:
        return len(self._snapshots)

    def retained_old_snapshots(self) -> list[tuple[str, int]]:
        """Snapshots kept alive only because something still pins them."""
        return sorted(k for k in self._snapshots if self._units[k[0]].version != k[1])
