# STATUS.md — 项目进度

_最后更新：2026-10-27_

## 当前阶段

原型 v0.1：符号层核心库完成并有测试；神经层持续学习框架、五个基线、VCM 学习器完成；
Permuted MNIST 主实验进行中。

## 已完成

- [x] 符号层核心库 `vcm/`（单元/版本/快照、多维相似度指针、角色绑定、版本化缓存 + 外部裁决、晋升、条件变量、DAG）
- [x] 符号层实验：`results/symbolic_demos.md`、`results/cache_cost/`、`results/promotion_depth/`
- [x] 神经层：数据（Permuted MNIST、Split CIFAR-10/100）、角色切分模型、实验协议、指标
- [x] 基线：Fine-tune、EWC（online）、ER、Independent、GRACE-CL（改编）
- [x] VCM 学习器（freeze / update × lazy / eager、修复、晋升 cut/keep）
- [x] 测试：`tests/test_core.py`（符号层）、`tests/test_cl.py`（神经层，合成数据）
- [x] 超参数验证集选择：`results/sweeps/pmnist_val.md`

## 进行中 / 待办

- [ ] Permuted MNIST 主实验（10 任务 × 3 种子）
- [ ] 低数据前向迁移实验（`pmnist_few`）
- [ ] Split CIFAR-10 / CIFAR-100
- [ ] 把结论写回 README

## 已知问题与局限

见 `docs/design.md` 第 3 节。
