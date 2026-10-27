"""Role-bound composition: v = Σ_role  r_role ⊛ f_role.

Each source of information sits in its own role slot, so "where the form came from"
and "where the meaning came from" can be pulled apart again by unbinding the role.
A weighted average of the fillers, by contrast, cannot be taken apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .similarity import SimilarityProfile
from .vsa import VSA, bind, normalize, unbind


class Composer:
    def __init__(self, vsa: VSA):
        self.vsa = vsa
        self._roles: dict[str, np.ndarray] = {}
        self.bind_ops = 0

    def role(self, name: str) -> np.ndarray:
        if name not in self._roles:
            self._roles[name] = self.vsa.unitary(name)
        return self._roles[name]

    @property
    def roles(self) -> list[str]:
        return sorted(self._roles)

    def compose(self, fillers: Mapping[str, np.ndarray], normalized: bool = True) -> np.ndarray:
        if not fillers:
            raise ValueError("a composition needs at least one role")
        total = np.zeros(self.vsa.dim)
        for role, filler in fillers.items():
            total += bind(self.role(role), np.asarray(filler, float))
            self.bind_ops += 1
        return normalize(total) if normalized else total

    def extract(self, vector: np.ndarray, role: str) -> np.ndarray:
        """Noisy filler of `role`; pass it through an ItemMemory to clean it up."""
        return unbind(np.asarray(vector, float), self.role(role))


@dataclass(frozen=True)
class CompositionSpec:
    """What a composition is made of: role -> unit id, plus per-role similarity notes."""

    bindings: tuple[tuple[str, str], ...]
    similarity: tuple[tuple[str, SimilarityProfile], ...] = field(default=())

    @classmethod
    def of(
        cls,
        bindings: Mapping[str, str],
        similarity: Mapping[str, SimilarityProfile | Mapping[str, float]] | None = None,
    ) -> "CompositionSpec":
        sims = []
        for role, prof in (similarity or {}).items():
            if role not in bindings:
                raise KeyError(f"similarity given for unbound role {role!r}")
            sims.append((role, prof if isinstance(prof, SimilarityProfile) else SimilarityProfile(prof)))
        return cls(tuple(sorted(bindings.items())), tuple(sorted(sims, key=lambda kv: kv[0])))

    @property
    def roles(self) -> dict[str, str]:
        return dict(self.bindings)

    def profile(self, role: str) -> SimilarityProfile:
        return dict(self.similarity).get(role, SimilarityProfile())
