# 符号层演示（自动生成）

由 `python experiments/symbolic_demos.py` 生成。

## 1. salir：多维相似度与角色绑定

与 `es:salir` 的相似度（form = 字符三元组 Jaccard，实测；meaning = 手写概念表；syntax = 词性是否相同）：

| 单元 | form | meaning | syntax | 单一混合分 |
|---|---|---|---|---|
| `fr:sortir` | 0.10 | 1.00 | 1.00 | 0.70 |
| `fr:salir` | 1.00 | 0.00 | 1.00 | 0.67 |
| `en:exit` | 0.00 | 1.00 | 1.00 | 0.67 |
| `es:salida` | 0.38 | 0.60 | 0.00 | 0.33 |
| `fr:sale` | 0.29 | 0.00 | 0.00 | 0.10 |

- 只看形式（单一相似度的典型情形）时最近邻是 `fr:salir`——假朋友。
- 按 meaning 维度检索（要求 meaning ≥ 0.5）时最近邻是 `fr:sortir`；按 form 检索是 `fr:salir`。同一对单元，不同用途取不同维度。

角色绑定 $v = r_{形}\otimes f_{salir} + r_{义}\otimes f_{出去}$ 之后解绑：

- 解绑 `form` 角色 → `es:salir`（cos = 0.69）
- 解绑 `meaning` 角色 → `meaning:GO_OUT`（cos = 0.69）
- 对照：加权平均向量与两个来源的 cos 分别为 0.71 / 0.71，无法说出哪一个是“形”、哪一个是“义”。

## 2. breakfast：晋升，以及切断 / 保持依赖

- `breakfast` 组合被调用 6 次、期间未失效 → 自动晋升为 `concept:breakfast`（level 1，模式 cut_deps）。
- 修改底层单元 `fast`（v1 → v2）后，没有任何缓存被立即重算（compose 次数仍为 0）。
- 调用 `my_morning`（指向 keep_deps 概念）：版本比对发现过期 → 先惰性重建概念 （v2），再重建组合；漂移 = 0.000。
- 调用 `my_morning_cut`（指向 cut_deps 概念）：版本一致，直接命中；概念保持 v1，与原料的漂移 = 0.500——就像 *breakfast* 早已不再是 break + fast。
- 本轮共新增 compose 1 次，概念刷新 1 次。

## 3. 条件变量

- 固定事实：水在 1 atm 下沸点 100 vs 90 → `contradiction`（fixed attribute: at least one of these claims is wrong）。两条都暂存，等待外部反馈裁决。
- 外部反馈保留 textbook 后，查询 → 100
- 可变条件：图书馆开放时间 9-21 vs 10-17 → `conditioned`，引入条件变量 `season`：
  - 9-21 | {'season': 'term'}
  - 10-17 | {'season': 'vacation'}
- 不给条件查询 → 有歧义（2 个分支）；给 season=vacation → 10-17
- 没有可用上下文时（门开 vs 门关），系统引入占位隐藏变量 `_hidden1`，两条知识以 A | t1、B | t2 共存，等待日后识别这个变量是什么。
