"""Vector-symbolic primitives: Holographic Reduced Representations (Plate, 1995).

Binding is circular convolution, computed in the Fourier domain. Role vectors are
*unitary* (every Fourier coefficient has magnitude 1), so unbinding a role is exact
for the term it bound; what remains is crosstalk from the other terms, which a
clean-up memory removes by nearest-neighbour lookup.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Mapping

import numpy as np


def _seed_from(name: str, salt: int) -> int:
    digest = hashlib.blake2b(f"{salt}:{name}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular convolution a ⊛ b."""
    return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=a.shape[-1])


def involution(a: np.ndarray) -> np.ndarray:
    """Approximate inverse a* with a*[i] = a[-i mod d]; exact inverse for unitary a."""
    return np.concatenate([a[:1], a[1:][::-1]])


def unbind(c: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Recover the filler bound to `a` inside `c` (plus crosstalk noise)."""
    return bind(c, involution(a))


class VSA:
    """Deterministic factory for random filler vectors and unitary role vectors."""

    def __init__(self, dim: int = 1024, seed: int = 0):
        if dim < 8:
            raise ValueError("dim must be >= 8")
        self.dim = dim
        self.seed = seed

    def random(self, name: str) -> np.ndarray:
        """Unit-norm Gaussian vector, reproducible from `name`."""
        rng = np.random.default_rng(_seed_from(name, self.seed))
        return normalize(rng.standard_normal(self.dim))

    def unitary(self, name: str) -> np.ndarray:
        """Vector whose spectrum has unit magnitude and random phase."""
        rng = np.random.default_rng(_seed_from("role:" + name, self.seed))
        n_freq = self.dim // 2 + 1
        phase = rng.uniform(-np.pi, np.pi, n_freq)
        phase[0] = 0.0
        if self.dim % 2 == 0:
            phase[-1] = 0.0
        return np.fft.irfft(np.exp(1j * phase), n=self.dim)


class ItemMemory:
    """Clean-up memory: maps noisy vectors back to the nearest stored symbol."""

    def __init__(self):
        self._names: list[str] = []
        self._rows: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None

    def __contains__(self, name: str) -> bool:
        return name in self._names

    def __len__(self) -> int:
        return len(self._names)

    def add(self, name: str, vector: np.ndarray) -> None:
        if name in self._names:
            self._rows[self._names.index(name)] = normalize(np.asarray(vector, float))
        else:
            self._names.append(name)
            self._rows.append(normalize(np.asarray(vector, float)))
        self._matrix = None

    def remove(self, name: str) -> None:
        i = self._names.index(name)
        del self._names[i]
        del self._rows[i]
        self._matrix = None

    def cleanup(
        self, vector: np.ndarray, candidates: Iterable[str] | None = None
    ) -> tuple[str, float]:
        """Return (name, cosine) of the best-matching stored symbol."""
        if not self._names:
            raise LookupError("item memory is empty")
        if self._matrix is None:
            self._matrix = np.stack(self._rows)
        sims = self._matrix @ normalize(np.asarray(vector, float))
        if candidates is not None:
            allowed = set(candidates)
            mask = np.array([n in allowed for n in self._names])
            if not mask.any():
                raise LookupError("no candidate is stored in item memory")
            sims = np.where(mask, sims, -np.inf)
        i = int(np.argmax(sims))
        return self._names[i], float(sims[i])


def superpose(terms: Iterable[np.ndarray]) -> np.ndarray:
    terms = list(terms)
    if not terms:
        raise ValueError("nothing to superpose")
    return np.sum(terms, axis=0)


def weighted_average(vectors: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> np.ndarray:
    """The blending baseline the README argues against: sources become inseparable."""
    total = sum(weights.values())
    return sum(weights[k] * vectors[k] for k in vectors) / total
