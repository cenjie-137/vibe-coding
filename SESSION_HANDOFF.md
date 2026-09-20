# SESSION HANDOFF — 会话交接文件

> **用途**：当你把任务从当前对话迁移到新对话时，把本文件内容（或本文件的路径）
> 作为第一条消息发给 AI，即可无缝恢复完整工作流。
>
> **使用方法**：新对话的第一条消息写：
> 「请先读取 c:\Users\32201\Desktop\T\vibe coding\DRIVE\SESSION_HANDOFF.md，
> 然后按其中的标准工作流继续任务：<你的新需求>」

---

## 1. 项目一句话定位

AI 辅助的医学 2D 分割 benchmark：用固定的标准化 prompt 工作流 + 病理基础模型 UNI，
在多个竞赛上对比「病理域预训练 vs ImageNet 预训练」的差距，产出可复现、可发表的基准论文。

---

## 2. 固定配置（不可变，新对话必须沿用）

- **AI 平台**：TRAE（不是 WorkBuddy）
- **AI 模型**：DeepSeek-V4-Pro（禁用 Auto Mode）
- **Prompt 模板**：`PROMPT_STANDARD.md` v1.1（系统提示固定 + Step 0 SOTA 调研 + 任务变量）
- **随机种子**：42
- **训练范式**：5-fold 分层交叉验证
- **复现工具**：`prompt_logger.py`（每次实验记录元数据）
- **实验日志**：`RESEARCH_LOG.md`（三层记忆：结论/配置/代码）

---

## 3. 已完成进度（关键数据点）

| 竞赛 | 状态 | 成绩 | 编码器 |
|------|------|------|--------|
| DRIVE | 已提交 | 前 2%（Dice 0.828） | ResNet34 |
| PUMA | 训练完，提交入口关闭 | TTA Micro Dice 0.6762（5折） | UNI |
| SIIM-ACR | 代码写好，暂缓 | — | 待定 |

**核心发现**：
- UNI 病理编码器在 PUMA 上比 ResNet34 高 +15.24%，比 nnU-Net baseline 高 +12.14%
- necrosis 从 0 提升到 0.90（UNI 的价值）
- blood_vessel 仍是短板（0.38-0.43）

---

## 4. 标准工作流（每个新竞赛重复执行）

```
Step 0  SOTA 调研（1 个 prompt，研究优先，禁止先写代码）
        → 调研数据集、SOTA 方法、冠军策略、失败模式
        → 输出 SOTA_RESEARCH YAML 存到 RESEARCH_LOG.md
        prompt: PROMPT_STANDARD.md 的 Step 0 固定模板

Step 1  写数据转换脚本 convert_<task>.py（1 个 prompt）
        → 原始数据 → 统一格式（images/ + labels/）
        验证：脚本跑通，打印类别分布

Step 2  写模型 + 训练代码（1 个 prompt，用固定系统提示）
        → models.py / dataset.py / train.py / inference.py
        编码器选择：病理→UNI，非病理→ResNet34/SwinUNETR
        必须引用 Step 0 的 SOTA 结论（loss/增强/后处理）

Step 3  Smoke test（1 epoch，验证能跑通）
        → 命令：python train.py --smoke_test

Step 4  全量训练（5 折）
        → UNI fold + ImageNet fold 两组对比

Step 5  评估 + 记录
        → prompt_logger.log_experiment() 记录所有元数据
        → 更新 RESEARCH_LOG.md（三层记忆）

Step 6  打包提交（如竞赛开放）
        → 生成 Docker + 提交
```

---

## 5. 当前决策状态

- **下一个竞赛**：BEETLE 已排除（587 张 WSI，100+GB 数据量，超出 2D 小图框架）
- **GLaS 腺体分割**（165 张 H&E 小图，~775×522，几百 MB，UNI 适用）← 推荐
  - 注意：GLaS 是 MICCAI 2015 比赛，**已关闭提交**，作为 benchmark 数据集用（可对标 SOTA，适合论文）
- 备选：MoNuSeg（30 张）、TNBC（50 张）——同为病理小图，不能提交

---

## 6. 文件清单

| 文件 | 作用 | 位置 |
|------|------|------|
| PROMPT_STANDARD.md | 标准 prompt 模板 | 本地 DRIVE 目录 |
| prompt_logger.py | 实验元数据记录 | 本地 DRIVE 目录 |
| RESEARCH_LOG.md | 实验日志 | 本地 DRIVE 目录 |
| SESSION_HANDOFF.md | 本文件 | 本地 DRIVE 目录 |
| models.py 等训练代码 | 训练代码 | 服务器 /root/autodl-tmp/medseg_project/code/ |

---

## 7. 服务器信息

- SSH：`ssh -p 44263 root@connect.nmb2.seetacloud.com`
- 代码目录：`/root/autodl-tmp/medseg_project/code/`
- 数据目录：`/root/autodl-tmp/medseg_project/data/`
- HuggingFace 镜像：`export HF_ENDPOINT=https://hf-mirror.com`
- UNI 权重：`MahmoodLab/UNI`（已授权，已下载）

---

## 8. 论文投稿目标

- 首选：MICCAI 2027 / Medical Image Analysis
- 核心创新点（4 个）：
  1. Prompt 元数据标准（可复现性）
  2. 人机协作 vs 纯 Agent 对比（ReX-MLE 数据为基线）
  3. 病理基础模型跨竞赛迁移验证
  4. MedAI-Repro Checklist（可复现性清单）