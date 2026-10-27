"""Conditional knowledge: what to do when two pieces of knowledge collide.

* If the attribute is a *fixed* objective fact under the stated conditions, two
  different values cannot both be true — at least one claim is wrong. The conflict is
  recorded and left for external feedback to settle.
* If the attribute can *vary*, the collision means a hidden variable was missed. Both
  claims are kept, each conditioned on its own value of that variable: A | T1, B | T2.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class ConditionType(str, Enum):
    FIXED = "fixed"
    VARIABLE = "variable"


class Outcome(str, Enum):
    STORED = "stored"
    REINFORCED = "reinforced"
    CONDITIONED = "conditioned"
    CONTRADICTION = "contradiction"


@dataclass(frozen=True)
class Claim:
    subject: str
    attribute: str
    value: Any
    conditions: Mapping[str, Any] = field(default_factory=dict)
    # Observed metadata (time, source, task, ...) that is not yet a condition.
    context: Mapping[str, Any] = field(default_factory=dict)
    source: str | None = None
    weight: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "conditions", MappingProxyType(dict(self.conditions)))
        object.__setattr__(self, "context", MappingProxyType(dict(self.context)))

    def given(self, **conditions) -> "Claim":
        return replace(self, conditions={**self.conditions, **conditions})


@dataclass
class Resolution:
    outcome: Outcome
    claims: list[Claim]
    variable: str | None = None
    note: str = ""


@dataclass
class Contradiction:
    subject: str
    attribute: str
    claims: list[Claim]
    resolved: Claim | None = None


@dataclass
class Answer:
    value: Any
    branches: list[Claim]

    @property
    def ambiguous(self) -> bool:
        return self.value is None and len(self.branches) > 1


def compatible(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Two condition sets can hold at once (no shared variable with different values)."""
    return all(a[k] == b[k] for k in a.keys() & b.keys())


class ConditionalKnowledge:
    def __init__(self, default: ConditionType = ConditionType.VARIABLE):
        self.default = ConditionType(default)
        self._kinds: dict[str, ConditionType] = {}
        self._claims: dict[tuple[str, str], list[Claim]] = {}
        self.contradictions: list[Contradiction] = []
        self._hidden = itertools.count(1)

    def declare(self, attribute: str, kind: ConditionType | str) -> None:
        self._kinds[attribute] = ConditionType(kind)

    def kind(self, attribute: str) -> ConditionType:
        return self._kinds.get(attribute, self.default)

    def claims(self, subject: str, attribute: str) -> list[Claim]:
        return list(self._claims.get((subject, attribute), []))

    def assert_claim(self, claim: Claim) -> Resolution:
        slot = self._claims.setdefault((claim.subject, claim.attribute), [])
        for i, old in enumerate(slot):
            if old.value == claim.value and dict(old.conditions) == dict(claim.conditions):
                slot[i] = replace(old, weight=old.weight + claim.weight)
                return Resolution(Outcome.REINFORCED, [slot[i]])
        conflicts = [
            c for c in slot
            if c.value != claim.value and compatible(c.conditions, claim.conditions)
        ]
        if not conflicts:
            slot.append(claim)
            return Resolution(Outcome.STORED, [claim])

        if self.kind(claim.attribute) is ConditionType.FIXED:
            slot.append(claim)
            record = Contradiction(claim.subject, claim.attribute, conflicts + [claim])
            self.contradictions.append(record)
            return Resolution(
                Outcome.CONTRADICTION, record.claims,
                note="fixed attribute: at least one of these claims is wrong",
            )

        variable, values = self._distinguishing_variable(conflicts + [claim])
        rewritten = {id(c): c.given(**{variable: values[i]}) for i, c in enumerate(conflicts)}
        slot[:] = [rewritten.get(id(c), c) for c in slot]
        new = claim.given(**{variable: values[-1]})
        slot.append(new)
        return Resolution(
            Outcome.CONDITIONED, list(rewritten.values()) + [new], variable=variable,
            note="variable attribute: conflict explained by a hidden variable",
        )

    def _distinguishing_variable(self, claims: list[Claim]) -> tuple[str, list[Any]]:
        """Prefer an observed context key whose values tell the claims apart."""
        shared = set.intersection(*(set(c.context) for c in claims))
        used = set().union(*(set(c.conditions) for c in claims))
        for key in sorted(shared - used):
            values = [c.context[key] for c in claims]
            if len(set(map(repr, values))) == len(values):
                return key, values
        name = f"_hidden{next(self._hidden)}"
        return name, [f"t{i + 1}" for i in range(len(claims))]

    def resolve(self, contradiction: Contradiction, keep: Claim) -> None:
        """External feedback picks the correct claim; the others are dropped."""
        slot = self._claims[(contradiction.subject, contradiction.attribute)]
        losers = {id(c) for c in contradiction.claims if c is not keep}
        slot[:] = [c for c in slot if id(c) not in losers]
        contradiction.resolved = keep

    def query(self, subject: str, attribute: str, **given) -> Answer:
        """Answer under the given conditions; ambiguous if a needed variable is unknown."""
        slot = self._claims.get((subject, attribute), [])
        matching = [c for c in slot if compatible(c.conditions, given)]
        determined = [c for c in matching if set(c.conditions) <= set(given)]
        pool = determined or matching
        if not pool:
            return Answer(None, [])
        values = {repr(c.value) for c in pool}
        if len(values) == 1:
            return Answer(pool[0].value, pool)
        if determined:
            best = max(determined, key=lambda c: (len(c.conditions), c.weight))
            return Answer(best.value, pool)
        return Answer(None, pool)
