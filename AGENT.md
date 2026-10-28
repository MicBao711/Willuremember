# AGENT.md — 给接手这个项目的人（或智能体）

开工前请先读完本文件和 `STATUS.md`。前者讲“这是什么、规矩是什么”，后者讲“现在走到哪儿了”。
完成一项工作后，更新 `STATUS.md`；需求或约定变了，更新本文件。

## 项目是什么

**VCM（Versioned Compositional Memory，带版本的组合式记忆）**：把知识拆成独立存储的基础单元，
用带角色、多维相似度的指针组合出新概念；常用组合结果缓存起来，通过版本号实现惰性失效；
稳定的缓存可以晋升为新的概念节点。目标是在持续学习中减少遗忘、保留迁移。构想见 `README.md`。

仓库有两层实现：

| 层 | 位置 | 依赖 | 作用 |
|---|---|---|---|
| 符号层 | `vcm/*.py` | numpy | 把 README 的六个组件原样实现成数据结构：单元、快照、指针、角色绑定、缓存、晋升、条件知识 |
| 神经层 | `vcm/cl/` | torch | 把同一套机制套到神经网络持续学习上，并与基线比较；网络按“模板 + 角色槽”组合 |

## 目录速览

```
vcm/                  符号层（核心库）
  vsa.py              HRR：循环卷积绑定、幺正角色向量、清理记忆
  units.py            UnitStore：按领域分区、版本号、写时复制快照、引用计数、DAG 层级约束
  similarity.py       SimilarityProfile（多维相似度）、Pointer（指向快照）、SimilarityIndex
  compose.py          Composer（角色绑定组合 / 解绑）、CompositionSpec
  cache.py            CompositionCache：记录依赖版本、用时检查、外部裁决（judge）
  promotion.py        晋升策略与 keep_deps / cut_deps 两种模式
  conditional.py      条件变量：固定事实冲突 → 矛盾；可变条件冲突 → A|T1, B|T2
  graph.py            DAG 工具：找环、拓扑序、依赖者
  memory.py           VCM 门面类，把以上组件串起来
vcm/cl/               神经层
  data.py             Permuted MNIST、Split CIFAR-10/100、组合式 MNIST（下载 + 缓存到 data/）
  models.py           模板（Template）与按“角色槽”切分的 MLP / CNN / compose_mlp、乘加次数统计
  runner.py           实验协议：学习 → 服务查询流 → 测量
  learners/           finetune / ewc / er / independent / grace（改编）/ vcm_learner
experiments/          可复现实验脚本（每个脚本顶部有用法）
tests/                pytest（tests/test_cl.py 用合成数据，不需要下载）
results/              实验产出：*.md 摘要、*.png 图；results/raw/ 是原始 JSON（可据此重画图）
docs/design.md        概念 → 代码的对应关系、神经层实例化细节、已知局限
```

## 常用命令

```bash
pip install -e ".[cl,dev]"                       # numpy + torch + matplotlib + pytest
python -m pytest -q                              # 全部测试（约 1 分钟）
python experiments/symbolic_demos.py             # salir / breakfast / 条件变量演示
python experiments/cache_cost.py                 # 惰性 vs 立即失效的计算开销
python experiments/promotion_depth.py            # 晋升与层级深度
python experiments/run_cl.py --benchmark pmnist --methods finetune ewc vcm_lazy --seeds 0 1 2
python experiments/plot_cl.py --raw results/raw/pmnist --out results/pmnist
bash experiments/run_suite.sh pmnist          # 结果背后的确切命令；另有 pmnist_stress / pmnist_stress_val /
                                              # pmnist_few / cifar10 / cifar100 / compose_val / compose / selection
STAGE=3 bash experiments/queue_rest.sh        # 超参数验证集选择 + CIFAR（可按阶段续跑）
python experiments/compose_readers.py         # 组合式 MNIST：强制指定读图器（概念作为积木本身值多少）
python experiments/compose_report.py          # 组合式 MNIST 学习器套件的表格与图
python experiments/selection_curves.py --raw results/raw/selection --out results/selection   # 候选选择：曲线重放
python experiments/selection_report.py        # 候选选择：学习器里 probe / halving / full 的对比
```

CPU 上的耗时参考（4 核）：Permuted MNIST 全套约 25 分钟；CIFAR-10 全套约 1 小时；CIFAR-100 约 1.5 小时；
组合式 MNIST（4 个数据量 × 11 种方法 × 5 个种子）约 45 分钟，强制读图器对照约 1 小时；候选选择（`run_suite.sh selection`）约 1 小时。
并行跑多个进程时注意内存（每个进程只在需要时生成任务数据，学完即释放训练集）。

数据集会自动下载到 `data/`（已在 `.gitignore` 中）。MNIST 来自 PyTorch 的 S3 镜像并校验 MD5；
CIFAR 来自 fast.ai 的 S3 镜像（多伦多大学原站在本环境的网络策略下不可达）。

## 约定

- **诚实第一。** 实验结论只写测得的东西；负面结果照实写进 `STATUS.md` 和结果摘要。
  超参数在验证集上选（`--eval-split val`），不在测试集上调。
- **基线要公平。** ER 的回放缓冲区与 VCM 的反馈缓冲区每任务样本数相同（`--buffer`）。
  计算量统一按乘加次数估计（训练样本 ≈ 3 次前向）。
- **VCM 默认设置。** `feedback_source="val"`、`repair_steps=0`（v0.1 实验里更好的一组）。v0.1 结果用的是旧默认
  （`--feedback-source train --repair-steps 100`），`run_suite.sh` 里用 `$V01` 显式固定；改默认值时同步更新脚本，保证旧结果可复现。
- **相关工作。** 与神经层最接近的是 MNTDP（冻结模块 + 任务驱动先验选组合）；对外描述 VCM 的新意时要区分开。
- **模板与旧结果。** 固定架构只有一个 `default` 模板，走的代码路径必须与 v0.1 逐位一致（改 `vcm/cl/` 时，
  用旧代码在同一命令下对拍 R 矩阵、计算量、裁决次数）。新功能用配置开关加（如 `promote_compositions`，默认关）。
- **组合式 MNIST 的超参数**在验证集上选（`run_suite.sh compose_val` → `results/sweeps/compose.md`）：
  `--min-steps 500`、`--probe-steps 100`、EWC λ = 1e5；主实验 5 个种子。
- **候选选择**（v0.3）：默认 `--selection probe`（v0.1–v0.2 的结果靠它复现）；`halving` / `full` 是可选项。
  选择方式的研究在验证集上做（曲线重放：选择集与打分集分开），测试集只用来做事先定好的最终比较；
  测试集上看到的小效应（如“探测后换优化器”）不在测试集上继续调。
- **许可证：MIT**（根目录 `LICENSE`）。
- **GRACE 是改编版。** GRACE 原本用于模型编辑，这里的 `GraceCL` 是我们为分类持续学习做的改编，
  改编点写在 `vcm/cl/learners/grace.py` 的文件头里，引用时要说明。
- **代码风格。** 英文 docstring / 注释，简洁；文档用中文。新增机制要配测试。
- **不要把模型标识写进提交信息或代码。**
- 提交到功能分支（v0.1：`claude/vcm-prototype`；v0.2 组合式实验：`claude/ecstatic-hawking-cc95sc`），不要直接推 `main`。
