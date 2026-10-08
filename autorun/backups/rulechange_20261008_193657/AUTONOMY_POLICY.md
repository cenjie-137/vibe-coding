# AUTONOMY_POLICY — 全自动分割 Agent 规则契约（v1.5）

> **本文件是人机之间的唯一接口。** 人写规则，Agent 遵守。
> **Agent 不得修改本文件及其配套的 `autorun/policy.json`、`autorun/judge.py`、`autorun/redlines.py`、`autorun/ledger.py`、`ROUTE_POOL.yaml`（红线 R7）。**
> 要改规则，只能由人改；改完把版本号 +1，并 git commit。
>
> 数值的机器可读版本在 `autorun/policy.json`，两者必须一致。**以 policy.json 为准**（judge.py 只读它）。

---

## 0. 数据集目录（可扩展到十几个）

**唯一清单**：`autorun/datasets.yaml`。加一个数据集 = 加一条 + 跑 `python autorun/build_policy.py --write`。**不需要改任何代码。**

目录共 15 个数据集，按就绪程度分三档：

| 档位 | 含义 | 数量 | 是否进判定 |
|------|------|------|-----------|
| `active` | 已就绪：协议已核实 + 保底线已实测 + 有官方评估脚本 | 2（GLaS、DRIVE） | ✅ 进 `policy.json` 的 targets |
| `preparing` | 基线已实测，但缺官方评估脚本（R5 挡着） | 1（PUMA） | ❌ 补脚本后转 active |
| `planned` | 已列入，协议与基线待 Step 0 核实 | 12 | ❌ |

planned 名单（2D）：ISIC2018、KvasirSEG、MoNuSeg、CHASEDB1、BUSI、BEETLE
planned 名单（3D）：ISLES2022、BraTS2023、LiTS、KiTS2023、ACDC、SynapseBTCV

> **为什么 planned 不进 targets**：判定器需要保底线才能工作。没有实测基线就进 targets，
> 判定器只会崩（`UNKNOWN_TASK`）。所以"想做的"和"能判的"必须分开 —— 这是诚实的边界。

---

## 1. 目标（两级，越靠前越好）

| 级别 | 定义 | GLaS | DRIVE |
|------|------|------|-------|
| **L1 保底** | 超过自家 nnU-Net v2 官方口径基线 | testB ObjDice > 0.8184；testA ObjDice > 0.8723 | CV Dice > 0.7990 |
| **L2 冲刺** | 超过已发表 SOTA / 榜前 50% | testB ObjDice > 0.842 | 排行榜前 50% |

- 达到 L1 **不停止**，继续冲 L2。
- 达到 L2 → 成功终止。
- 所有比较必须**同指标 + 同子集 + 同口径**（R1）。
- 新增数据集的 L1 / L2 一律先留 `null`，Step 0 核实来源后再填（R2）。
- **复现值不计入 `best`（R10）**：锚点路线复现出来的分数，只用来**校验我们的评估口径对不对**，
  必须存进 `state.anchor`，**不得**写进 `state.best`。`state.best.source` 只能是 `ours`，
  即 `best` 只记**我们自研 / 改进模型**的分数。
  理由：复现的是**别人的方法**；把它记成自己的成绩，等于拿他人论文的数字冒充自己达标，
  直接摧毁整篇论文的可信度根基。这条是本项目区别于"刷榜式"自主 Agent 的关键。

---

## 2. 阈值（触发换路线）

| 项 | 定义 | 阈值 |
|----|------|------|
| 小轮 | 1 折 + 30 epoch | 连续 **3** 轮相对提升 < **0.3%** → 小轮判负 |
| 大轮 | 全量 5 折 | 连续 **2** 轮相对提升 < **0.5%** → 弃该路线 |

- "相对提升" = `(v_new − v_best_prev) / max(v_best_prev, ε)`。
- 小轮判负不直接弃路线，先回 Step 2 再试；大轮判负才弃路线、回 Step 1 取下一条。

---

## 3. 重试与失败

| 场景 | 处理 |
|------|------|
| 训练崩溃 / OOM / 环境小问题 | 重试 **3** 次（换 seed / 降 batch / 重连），每次留痕 |
| 仍失败且属方法本身 | 标记该路线失败，取池中下一条 |
| 仍失败且属基础设施 | **硬停** + 写现场报告 |
| 连续失败路线数 | 达 **3** 条 → 硬停 |

---

## 4. 路线池准入（半强制复现）

- **有开源代码** → 必须复现成功（指标落在论文报告值合理区间）才能进入改进阶段；成功后升级为**锚点路线**。
- **无开源代码** → 允许直接自研，但必须标 `confidence: low`，且**不参与锚点判定**。
- 复现失败（重试 3 次后）→ 记录原因，取池中下一条。

---

## 5. 红线（Agent 不可违反）

| 编号 | 红线 |
|------|------|
| **R1** | **口径锁定**：任何比较必须同指标 + 同子集 + 同口径；跨口径数字禁止并列、禁止用于判定 |
| **R2** | **来源强制**：任何基线 / SOTA 数字必须带来源 URL；无来源标 `UNVERIFIED`，不得用于判定 |
| **R3** | **停止阈值**：小轮 3×0.3% / 大轮 2×0.5%（见第 2 节） |
| **R4** | **复现准入**：代码链接须**亲自核实**（`code_verified: true`）才可标 `high`；无代码路线标 `low` 且不做锚点；未复现成功不得进 `improving` / `anchor` |
| **R5** | **评估锁定**：只用官方协议脚本评估，禁止自造指标或临时改口径 |
| **R6** | **破坏性操作**：删权重 / 覆盖 / 清目录前必须先备份 + 留痕；禁止未备份的删除 |
| **R7** | **自我修改禁止**：不得修改 `AUTONOMY_POLICY.md` / `policy.json` / `judge.py` / `redlines.py` / `ledger.py` / `ROUTE_POOL.yaml` 的判定字段 |
| **R8** | **留痕强制**：每个动作写 `run_log.jsonl`；每个结论写 `RESEARCH_LOG.md` |
| **R9** | **资源边界**：不设 GPU 小时上限；但受「最多试 N 条路线」+「连续失败上限」约束 |
| **R10** | **锚点隔离**：复现所得分数只作口径校验，存 `state.anchor`；`state.best.source` 必须为 `ours`，禁止把他人方法的成绩计入自身 `best` |

---

## 6. 终止条件（三出口，缺一不可）

1. **达标**：best ≥ L2 → `SUCCESS_L2`（若池耗尽时仅达 L1 → `SUCCESS_L1`）
2. **候选池耗尽**：全部路线走完仍未达 L1 → `POOL_EXHAUSTED`
3. **连续失败**：连续 3 条路线失败 → `CONSECUTIVE_FAILURE`

---

## 7. 修改本文件的规则

- 只有**人**能改本文件。
- 每次修改：版本号 +1 → 同步 `autorun/policy.json` → git commit（message 写明改了什么、为什么）。
- Agent 只能**读取**，不得写入。

---

**版本**：v1.5
**配套文件**：
- `autorun/datasets.yaml` —— 数据集目录（15 个，可扩展）
- `autorun/policy.json` —— 机器可读契约（判定器唯一数据源）
- `autorun/judge.py` —— 确定性判定器
- `autorun/redlines.py` —— 十条红线的可执行断言
- `autorun/ledger.py` —— 溯源账本 + 可信记分卡
- `autorun/replay.py` —— 重放验证器（可信性的技术证据）
- `autorun/build_policy.py` —— 目录 → policy.json 编译器（只许人跑）
- `autorun/acceptance.py` —— 一键验收跑批
- `ROUTE_POOL.yaml` —— 候选路线池
