"""VCM: Versioned Compositional Memory."""

from .cache import CacheEntry, CompositionCache, StalePolicy
from .compose import Composer, CompositionSpec
from .conditional import Claim, ConditionalKnowledge, ConditionType, Outcome
from .memory import VCM
from .promotion import PromotionMode, PromotionPolicy, promote
from .similarity import Pointer, SimilarityIndex, SimilarityProfile
from .units import DAGViolation, Snapshot, Unit, UnitKind, UnitStore
from .vsa import VSA, ItemMemory, bind, cosine, unbind

__all__ = [
    "VCM", "VSA", "ItemMemory", "bind", "unbind", "cosine",
    "UnitStore", "Unit", "UnitKind", "Snapshot", "DAGViolation",
    "SimilarityProfile", "SimilarityIndex", "Pointer",
    "Composer", "CompositionSpec",
    "CompositionCache", "CacheEntry", "StalePolicy",
    "PromotionMode", "PromotionPolicy", "promote",
    "ConditionalKnowledge", "Claim", "ConditionType", "Outcome",
]
__version__ = "0.1.0"
