# nnU-Net 基线 Agent — 使用说明

> 一个用**标准 nnU-Net v2** 以固定协议自动跑分割竞赛的基线 agent，作为你「LLM 语义
> 自动化」方法的**规则化自动化对照组**（head-to-head，同官方协议可对比可复现）。

## 技能位置
- 技能本体：`.trae/skills/nnunet-baseline/`
  - `SKILL.md` — 触发规则 + 红线 + 调用说明
  - `reference/nnunet_baseline_prompt.md` — 配方式启动提示词
  - `scripts/convert_to_nnunet.py` — 无损数据转换脚手架
  - `scripts/run_baseline.sh` — 一键流程编排（服务器）
  - `templates/log_block.md` — RESEARCH_LOG 分区记录模板
- 模板本源：`NNUNET_BASELINE_AGENT_PROMPT.md`

## 如何触发（任选其一）
1. 在 TRAE 里直接说：`用 nnunet-baseline 技能给 <数据集> 跑一个 nnU-Net 基线`，并给出数据集路径。
2. 新开对话时，把 `reference/nnunet_baseline_prompt.md` 中「复制线以下」内容作为**第一条消息**完整粘贴，再补一句任务变量（哪个数据集、路径）。

## 标准工作流（技能强制按序执行）
```
Step 0  SOTA/工具调研（只调研，禁止先写代码）→ 核对 nnU-Net v2 用法 + 比赛数据是否已验证
Step 1  无损数据转换（convert_to_nnunet.py，改配置后运行），打印每类样本数自检
Step 2  nnUNetv2_plan_and_preprocess
Step 3  5 折训练 + 预测 + nnU-Net 自带 post-processing
Step 4  ensemble（5 折融合）
Step 5  官方评估协议算指标（不自造）
Step 6  结果 + 配置 + 命令写进 RESEARCH_LOG.md（nnU-Net 分区，与 LLM 方法并列）
```

## 红线（坏了会失去对照价值）
- 只用**标准 nnU-Net v2**，禁止手写方法污染；**seed=42**；**官方协议**评估；**全程留痕**；
  **RESEARCH_LOG 分区**、指标与 LLM 方法不混写。

## 服务器命令速查（nnuUNet 已装，env 三件套已 export 的前提下）
```bash
python .trae/skills/nnunet-baseline/scripts/convert_to_nnunet.py --config <cfg> --smoke 5
bash .trae/skills/nnunet-baseline/scripts/run_baseline.sh plan
DATASET_ID=<ID> TASK_NAME=<TASK> bash .trae/skills/nnunet-baseline/scripts/run_baseline.sh train
DATASET_ID=<ID> TASK_NAME=<TASK> bash .trae/skills/nnunet-baseline/scripts/run_baseline.sh predict
```

## 首个数据集建议
从 GLaS 或 DRIVE 开始（数据已就绪），先 `--smoke 5` 跑通链路拿「流程可复现」票，
再真实 5 折出一个基线数字。