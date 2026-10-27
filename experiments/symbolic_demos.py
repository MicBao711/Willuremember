"""Three small, inspectable demos of the README's mechanisms.

1. salir: multi-dimensional similarity + role binding vs. a single score / averaging.
2. breakfast: promotion, and keep-deps vs cut-deps when a part changes.
3. conditional knowledge: fixed facts vs. a hidden variable.

Form similarity is computed (character trigram Jaccard). Meaning similarity comes
from a tiny hand-written concept table, so it is illustrative, not measured.

    python experiments/symbolic_demos.py [--out results/symbolic_demos.md]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vcm import VCM, Claim, ConditionType, PromotionMode, PromotionPolicy, cosine
from vcm.vsa import weighted_average


def trigrams(word: str) -> set[str]:
    w = f"^{word}$"
    return {w[i:i + 3] for i in range(len(w) - 2)}


def form_similarity(a: str, b: str) -> float:
    ta, tb = trigrams(a), trigrams(b)
    return len(ta & tb) / len(ta | tb)


LEXICON = {
    # unit id: (spelling, concept, part of speech)
    "es:salir": ("salir", "GO_OUT", "verb"),
    "fr:salir": ("salir", "MAKE_DIRTY", "verb"),
    "fr:sortir": ("sortir", "GO_OUT", "verb"),
    "en:exit": ("exit", "GO_OUT", "verb"),
    "es:salida": ("salida", "EXIT_PLACE", "noun"),
    "fr:sale": ("sale", "DIRTY", "adj"),
}
RELATED_CONCEPTS = {frozenset({"GO_OUT", "EXIT_PLACE"}): 0.6, frozenset({"MAKE_DIRTY", "DIRTY"}): 0.8}


def meaning_similarity(c1: str, c2: str) -> float:
    return 1.0 if c1 == c2 else RELATED_CONCEPTS.get(frozenset({c1, c2}), 0.0)


def demo_salir(lines: list[str]) -> None:
    vm = VCM(dim=2048, seed=7)
    for uid, (spelling, concept, pos) in LEXICON.items():
        vm.add(uid, domain=uid.split(":")[0])
    for concept in {c for _, c, _ in LEXICON.values()}:
        vm.add(f"meaning:{concept}", domain="meaning")
    ids = list(LEXICON)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            sa, ca, pa = LEXICON[a]
            sb, cb, pb = LEXICON[b]
            vm.relate(a, b, form=form_similarity(sa, sb), meaning=meaning_similarity(ca, cb),
                      syntax=1.0 if pa == pb else 0.0)

    lines.append("## 1. salir：多维相似度与角色绑定\n")
    lines.append("与 `es:salir` 的相似度（form = 字符三元组 Jaccard，实测；meaning = 手写概念表；syntax = 词性是否相同）：\n")
    lines.append("| 单元 | form | meaning | syntax | 单一混合分 |")
    lines.append("|---|---|---|---|---|")
    for other, prof in vm.similar("es:salir"):
        lines.append(f"| `{other}` | {prof['form']:.2f} | {prof['meaning']:.2f} | "
                     f"{prof['syntax']:.2f} | {prof.blended():.2f} |")
    by_form = vm.similar("es:salir", rank_by="form")[0][0]
    by_meaning = vm.similar("es:salir", rank_by="meaning", meaning=0.5)[0][0]
    form_only = vm.similar("es:salir", rank_by="form")
    lines.append("")
    lines.append(f"- 只看形式（单一相似度的典型情形）时最近邻是 `{form_only[0][0]}`——假朋友。")
    lines.append(f"- 按 meaning 维度检索（要求 meaning ≥ 0.5）时最近邻是 `{by_meaning}`；按 form 检索是 `{by_form}`。"
                 "同一对单元，不同用途取不同维度。\n")

    vm.compose("es:salir+", {"form": "es:salir", "meaning": "meaning:GO_OUT"},
               similarity={"form": {"form": 1.0}, "meaning": {"meaning": 1.0}})
    f_dec, f_sim = vm.decode("es:salir+", "form")
    m_dec, m_sim = vm.decode("es:salir+", "meaning")
    avg = weighted_average({"f": vm.vector("es:salir"), "m": vm.vector("meaning:GO_OUT")},
                           {"f": 1.0, "m": 1.0})
    lines.append("角色绑定 $v = r_{形}\\otimes f_{salir} + r_{义}\\otimes f_{出去}$ 之后解绑：\n")
    lines.append(f"- 解绑 `form` 角色 → `{f_dec}`（cos = {f_sim:.2f}）")
    lines.append(f"- 解绑 `meaning` 角色 → `{m_dec}`（cos = {m_sim:.2f}）")
    lines.append(f"- 对照：加权平均向量与两个来源的 cos 分别为 "
                 f"{cosine(avg, vm.vector('es:salir')):.2f} / {cosine(avg, vm.vector('meaning:GO_OUT')):.2f}，"
                 "无法说出哪一个是“形”、哪一个是“义”。\n")


def demo_breakfast(lines: list[str]) -> None:
    vm = VCM(dim=2048, seed=3, promotion=PromotionPolicy(min_hits=5, min_stable_hits=5,
                                                          mode=PromotionMode.CUT_DEPS))
    for w in ["break", "fast", "morning", "meal"]:
        vm.add(w)
    for _ in range(6):
        vm.compose("breakfast", {"action": "break", "object": "fast"})
    cut = "concept:breakfast"
    lines.append("## 2. breakfast：晋升，以及切断 / 保持依赖\n")
    entry = vm.cache.entry("breakfast")
    lines.append(f"- `breakfast` 组合被调用 {entry.hits + 1} 次、期间未失效 → 自动晋升为 `{cut}`"
                 f"（level {vm.store.level(cut)}，模式 cut_deps）。")
    vm.compose("breakfast_keep", {"action": "break", "object": "fast"})
    keep = vm.promote("breakfast_keep", "concept:breakfast_keep", PromotionMode.KEEP_DEPS)
    vm.compose("my_morning", {"time": "morning", "event": keep})
    vm.compose("my_morning_cut", {"time": "morning", "event": cut})
    before = vm.cache.stats.composes

    # "fast" shifts its everyday meaning (abstain-from-food -> quick).
    vm.update("fast", vm.vsa.random("fast:quick"))
    lines.append(f"- 修改底层单元 `fast`（v1 → v{vm.store.version('fast')}）后，没有任何缓存被立即重算"
                 f"（compose 次数仍为 {vm.cache.stats.composes - before}）。")
    vm.lookup("my_morning")
    vm.lookup("my_morning_cut")
    lines.append(f"- 调用 `my_morning`（指向 keep_deps 概念）：版本比对发现过期 → 先惰性重建概念 "
                 f"（v{vm.store.version(keep)}），再重建组合；漂移 = {vm.drift(keep):.3f}。")
    lines.append(f"- 调用 `my_morning_cut`（指向 cut_deps 概念）：版本一致，直接命中；概念保持 "
                 f"v{vm.store.version(cut)}，与原料的漂移 = {vm.drift(cut):.3f}——"
                 "就像 *breakfast* 早已不再是 break + fast。")
    lines.append(f"- 本轮共新增 compose {vm.cache.stats.composes - before} 次，概念刷新 {vm.concept_refreshes} 次。\n")


def demo_conditional(lines: list[str]) -> None:
    vm = VCM()
    kb = vm.knowledge
    kb.declare("boiling_point_c", ConditionType.FIXED)
    lines.append("## 3. 条件变量\n")
    kb.assert_claim(Claim("water", "boiling_point_c", 100, {"pressure_atm": 1}, source="textbook"))
    r = kb.assert_claim(Claim("water", "boiling_point_c", 90, {"pressure_atm": 1}, source="noisy_sensor"))
    lines.append(f"- 固定事实：水在 1 atm 下沸点 100 vs 90 → `{r.outcome.value}`（{r.note}）。"
                 "两条都暂存，等待外部反馈裁决。")
    kb.resolve(kb.contradictions[0], keep=kb.contradictions[0].claims[0])
    lines.append(f"- 外部反馈保留 textbook 后，查询 → {kb.query('water', 'boiling_point_c', pressure_atm=1).value}")
    kb.assert_claim(Claim("library", "opening_hours", "9-21", context={"season": "term"}))
    r = kb.assert_claim(Claim("library", "opening_hours", "10-17", context={"season": "vacation"}))
    lines.append(f"- 可变条件：图书馆开放时间 9-21 vs 10-17 → `{r.outcome.value}`，"
                 f"引入条件变量 `{r.variable}`：")
    for c in kb.claims("library", "opening_hours"):
        lines.append(f"  - {c.value} | {dict(c.conditions)}")
    amb = kb.query("library", "opening_hours")
    lines.append(f"- 不给条件查询 → 有歧义（{len(amb.branches)} 个分支）；给 season=vacation → "
                 f"{kb.query('library', 'opening_hours', season='vacation').value}")
    kb.assert_claim(Claim("door", "state", "open"))
    r = kb.assert_claim(Claim("door", "state", "closed"))
    lines.append(f"- 没有可用上下文时（门开 vs 门关），系统引入占位隐藏变量 `{r.variable}`，"
                 "两条知识以 A | t1、B | t2 共存，等待日后识别这个变量是什么。\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/symbolic_demos.md")
    args = ap.parse_args()
    lines = ["# 符号层演示（自动生成）\n",
             "由 `python experiments/symbolic_demos.py` 生成。\n"]
    demo_salir(lines)
    demo_breakfast(lines)
    demo_conditional(lines)
    text = "\n".join(lines)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
