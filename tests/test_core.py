import numpy as np
import pytest

from vcm import (
    VCM, VSA, Claim, CompositionCache, ConditionalKnowledge, ConditionType, DAGViolation,
    ItemMemory, Outcome, PromotionMode, PromotionPolicy, SimilarityIndex, SimilarityProfile,
    StalePolicy, UnitStore, bind, cosine, unbind,
)
from vcm.graph import dependents, find_cycle, topological_order
from vcm.vsa import weighted_average


# -- VSA ---------------------------------------------------------------------
def test_unitary_unbinding_is_exact():
    vsa = VSA(512, seed=1)
    r, f = vsa.unitary("role"), vsa.random("filler")
    assert np.allclose(unbind(bind(r, f), r), f, atol=1e-10)
    assert np.allclose(np.abs(np.fft.rfft(r)), 1.0)


def test_role_binding_separates_sources_but_average_does_not():
    vm = VCM(dim=1024, seed=0)
    for w in ["salir_es", "go_out", "dirty", "exit", "sortir_fr"]:
        vm.add(w)
    vm.compose("salir_es_concept", {"form": "salir_es", "meaning": "go_out"})
    assert vm.decode("salir_es_concept", "form")[0] == "salir_es"
    assert vm.decode("salir_es_concept", "meaning")[0] == "go_out"
    avg = weighted_average(
        {"a": vm.vector("salir_es"), "b": vm.vector("go_out")}, {"a": 1.0, "b": 1.0}
    )
    # The blend is equally close to both sources: nothing tells which was the form.
    assert abs(cosine(avg, vm.vector("salir_es")) - cosine(avg, vm.vector("go_out"))) < 1e-9


def test_item_memory_candidates():
    mem = ItemMemory()
    vsa = VSA(256)
    mem.add("a", vsa.random("a"))
    mem.add("b", vsa.random("b"))
    assert mem.cleanup(vsa.random("a"))[0] == "a"
    assert mem.cleanup(vsa.random("a"), candidates=["b"])[0] == "b"


# -- units / snapshots -----------------------------------------------------------
def test_versions_and_copy_on_write_snapshots():
    store = UnitStore()
    arr = np.ones(4)
    store.add("u", arr, domain="es")
    arr[:] = 5  # the caller's array is not the stored one
    assert store.snapshot("u").payload[0] == 1
    snap = store.pin("u")
    assert store.update("u", np.zeros(4)) == 2
    assert snap.payload[0] == 1 and store.snapshot("u", 1).payload[0] == 1
    assert store.snapshot("u").payload[0] == 0
    with pytest.raises(ValueError):
        store.snapshot("u").payload[0] = 3  # snapshots are read-only
    store.unpin("u", 1)
    with pytest.raises(KeyError):
        store.snapshot("u", 1)


def test_unpinned_old_versions_are_released():
    store = UnitStore()
    store.add("u", np.zeros(2))
    store.update("u", np.ones(2))
    with pytest.raises(KeyError):
        store.snapshot("u", 1)
    assert store.snapshot_count == 1


def test_domains_are_isolated():
    store = UnitStore()
    store.add("salir_es", np.ones(3), domain="es")
    store.add("salir_fr", np.ones(3), domain="fr")
    store.update("salir_es", np.zeros(3))
    assert store.domain("es") == {"salir_es"} and store.domain("fr") == {"salir_fr"}
    assert store.version("salir_fr") == 1
    assert np.all(store.snapshot("salir_fr").payload == 1)


def test_dependencies_must_point_down():
    store = UnitStore()
    store.add("a", 1)
    store.add("c", 2, level=1, deps={"a": 1})
    with pytest.raises(DAGViolation):
        store.add("d", 3, level=1, deps={"c": 1})
    store.add("e", 3, level=2, deps={"c": 1, "a": 1})


def test_graph_helpers():
    edges = {"c": ["a", "b"], "d": ["c"], "a": [], "b": []}
    assert find_cycle(edges) is None
    order = topological_order(edges)
    assert order.index("a") < order.index("c") < order.index("d")
    assert dependents(edges, "a") == {"c", "d"}
    cyclic = {"x": ["y"], "y": ["z"], "z": ["x"]}
    cycle = find_cycle(cyclic)
    assert cycle[0] == cycle[-1] and set(cycle) == {"x", "y", "z"}
    with pytest.raises(ValueError):
        topological_order(cyclic)


# -- similarity ------------------------------------------------------------------
def test_multidimensional_similarity_avoids_false_friends():
    idx = SimilarityIndex()
    idx.relate("salir_es", "salir_fr", {"form": 1.0, "meaning": 0.0, "syntax": 0.9})
    idx.relate("salir_es", "sortir_fr", {"form": 0.33, "meaning": 0.95, "syntax": 0.9})
    by_form = idx.neighbors("salir_es", rank_by="form")[0][0]
    by_meaning = idx.neighbors("salir_es", rank_by="meaning")[0][0]
    assert (by_form, by_meaning) == ("salir_fr", "sortir_fr")
    # One blended number yields one ranking, so it is wrong for at least one purpose.
    blended = idx.neighbors("salir_es")[0][0]
    assert blended in (by_form, by_meaning) and by_form != by_meaning
    assert [n for n, _ in idx.neighbors("salir_es", {"meaning": 0.5})] == ["sortir_fr"]
    with pytest.raises(ValueError):
        SimilarityProfile({"form": 1.5})


# -- cache -----------------------------------------------------------------------
def _counting_cache(store, policy=StalePolicy.RECOMPOSE):
    calls = []

    def build(spec):
        calls.append(spec)
        return sum(store.snapshot(u).payload for u in spec), {u: store.version(u) for u in spec}

    return CompositionCache(build, store.version, policy), calls


def test_cache_hits_and_lazy_invalidation():
    store = UnitStore()
    store.add("a", 1)
    store.add("b", 2)
    cache, calls = _counting_cache(store)
    assert cache.get("ab", ("a", "b")) == 3
    assert cache.get("ab") == 3 and len(calls) == 1
    store.update("a", 10)
    store.update("a", 20)  # nothing happens yet: lazy
    assert len(calls) == 1
    assert cache.get("ab") == 22  # two updates coalesced into one rebuild
    assert len(calls) == 2
    e = cache.entry("ab")
    assert e.invalidations == 1 and e.stable_hits == 0 and e.deps == {"a": 3, "b": 1}
    assert cache.stats.hits == 1 and cache.stats.stale == 1 and cache.stats.misses == 1


def test_cache_evict_policy_and_feedback():
    store = UnitStore()
    store.add("a", 1)
    cache, calls = _counting_cache(store, StalePolicy.EVICT)
    cache.get("k", ("a",))
    cache.get("k")
    store.update("a", 5)
    assert cache.get("k") == 5
    assert cache.entry("k").hits == 0  # history was dropped with the evicted entry
    cache.feedback("k", ok=False)
    assert "k" not in cache and cache.stats.feedback_errors == 1
    with pytest.raises(KeyError):
        cache.get("k")


def test_cache_capacity_lru():
    store = UnitStore()
    for u in "abc":
        store.add(u, 1)
    cache, _ = _counting_cache(store)
    cache.capacity = 2
    cache.get("a", ("a",))
    cache.get("b", ("b",))
    cache.get("a")
    cache.get("c", ("c",))
    assert "b" not in cache and "a" in cache and "c" in cache


def test_facade_pins_snapshots_used_by_cached_compositions():
    vm = VCM(dim=512)
    vm.add("x")
    vm.add("y")
    vm.compose("xy", {"r1": "x", "r2": "y"})
    vm.update("x", vm.vsa.random("x2"))
    assert vm.store.pinned_versions("x") == [1]
    # The stale cached value still decodes against the snapshot it was built from.
    stale = vm.cache.peek("xy").value
    assert vm.decode(stale, "r1")[0] == "x@v1"
    vm.lookup("xy")  # lazy check -> rebuild -> repin to v2
    assert vm.store.pinned_versions("x") == [2]
    assert vm.store.retained_old_snapshots() == []
    assert vm.decode("xy", "r1")[0] == "x"


# -- promotion ------------------------------------------------------------------
def test_promotion_after_frequent_stable_use():
    vm = VCM(dim=512, promotion=PromotionPolicy(min_hits=3, min_stable_hits=3))
    vm.add("break")
    vm.add("fast")
    for _ in range(3):
        vm.compose("breakfast", {"a": "break", "b": "fast"})
    assert "concept:breakfast" not in vm.store
    vm.lookup("breakfast")
    concept = vm.store.get("concept:breakfast")
    assert concept.level == 1 and concept.kind.value == "concept"


def test_invalidation_blocks_promotion():
    vm = VCM(dim=512, promotion=PromotionPolicy(min_hits=4, min_stable_hits=3))
    vm.add("u")
    vm.compose("k", {"r": "u"})
    vm.lookup("k")
    vm.lookup("k")
    vm.update("u", vm.vsa.random("u2"))
    vm.lookup("k")  # rebuilt: stable streak restarts
    vm.lookup("k")
    assert "concept:k" not in vm.store
    vm.lookup("k")
    vm.lookup("k")
    assert "concept:k" in vm.store


def test_keep_vs_cut_dependencies():
    vm = VCM(dim=1024)
    vm.add("break")
    vm.add("fast")
    vm.compose("bf", {"a": "break", "b": "fast"})
    keep = vm.promote("bf", "bf_keep", PromotionMode.KEEP_DEPS)
    vm.compose("bf2", {"a": "break", "b": "fast"})
    cut = vm.promote("bf2", "bf_cut", PromotionMode.CUT_DEPS)
    vm.compose("meal", {"what": keep})
    vm.compose("meal_cut", {"what": cut})
    vm.update("fast", vm.vsa.random("fast_new_meaning"))
    assert vm.store.version(keep) == 1  # nothing is pushed eagerly
    vm.lookup("meal")  # lazy: refresh concept, then the composite that uses it
    assert vm.store.version(keep) == 2 and vm.concept_refreshes == 1
    assert vm.drift(keep) < 1e-9
    assert vm.drift(cut) > 0.2  # independent concept drifted away from its origin
    assert vm.store.version(cut) == 1
    stats = vm.cache.stats.stale
    vm.lookup("meal_cut")  # unaffected by the change in its origin
    assert vm.cache.stats.stale == stats


# -- conditional knowledge -------------------------------------------------------
def test_fixed_attribute_conflict_is_a_contradiction():
    kb = ConditionalKnowledge()
    kb.declare("boiling_point_c", ConditionType.FIXED)
    kb.assert_claim(Claim("water", "boiling_point_c", 100, {"pressure_atm": 1}))
    r = kb.assert_claim(Claim("water", "boiling_point_c", 90, {"pressure_atm": 1}))
    assert r.outcome is Outcome.CONTRADICTION and len(kb.contradictions) == 1
    # A different stated condition is not a conflict at all.
    r = kb.assert_claim(Claim("water", "boiling_point_c", 70, {"pressure_atm": 0.3}))
    assert r.outcome is Outcome.STORED
    kb.resolve(kb.contradictions[0], keep=kb.contradictions[0].claims[0])
    assert kb.query("water", "boiling_point_c", pressure_atm=1).value == 100


def test_variable_attribute_conflict_becomes_conditional():
    kb = ConditionalKnowledge()
    kb.assert_claim(Claim("team", "captain", "Ann", context={"season": 2023}))
    r = kb.assert_claim(Claim("team", "captain", "Bo", context={"season": 2024}))
    assert r.outcome is Outcome.CONDITIONED and r.variable == "season"
    assert kb.query("team", "captain", season=2023).value == "Ann"
    assert kb.query("team", "captain", season=2024).value == "Bo"
    assert kb.query("team", "captain").ambiguous


def test_hidden_variable_is_introduced_when_context_cannot_explain():
    kb = ConditionalKnowledge()
    kb.assert_claim(Claim("door", "state", "open"))
    r = kb.assert_claim(Claim("door", "state", "closed"))
    assert r.outcome is Outcome.CONDITIONED and r.variable.startswith("_hidden")
    assert kb.query("door", "state", **{r.variable: "t1"}).value == "open"
    assert kb.query("door", "state", **{r.variable: "t2"}).value == "closed"
    assert kb.assert_claim(Claim("door", "state", "open", {r.variable: "t1"})).outcome is Outcome.REINFORCED


def test_judge_can_keep_a_stale_value_until_the_next_change():
    from vcm import Decision
    store = UnitStore()
    store.add("a", 1)
    store.add("b", 2)
    verdicts = iter([Decision.KEEP, Decision.RECOMPOSE])
    seen = []

    def judge(entry, stale):
        seen.append(stale)
        return next(verdicts)

    cache, calls = _counting_cache(store)
    cache.judge = judge
    cache.get("ab", ("a", "b"))
    store.update("a", 10)
    assert cache.get("ab") == 3          # judge said the old value is still right
    assert seen == [{"a": (1, 2)}] and cache.stats.kept == 1
    assert cache.get("ab") == 3 and len(seen) == 1  # acknowledged: no re-check
    store.update("b", 5)
    assert cache.get("ab") == 15         # next change: judge asked again, rebuilds
    assert seen[1] == {"a": (1, 2), "b": (1, 2)}
    assert cache.entry("ab").deps == {"a": 2, "b": 2}
