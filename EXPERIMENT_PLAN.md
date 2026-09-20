# AI 辅助科研范式 — 实验方案（修订版）

> 创建时间：2026-09-01
> 基于 CamylaNet Baseline-First 思路 + 病理基础模型双线并行

---

## 一、核心纠正

### PUMA Baseline 实际方法

| 项目 | 之前认知（错误） | 实际情况 |
|------|----------------|---------|
| Baseline 编码器 | ConvNeXt V2 Base | **nnU-Net v2 (2D)** |
| Baseline Micro Dice | 0.5548 | 0.5548（正确） |
| Top 1 方法 | 不清楚 | Virchow2 病理基础模型 + Efficient-UNet (0.7823) |
| Top 2 方法 | 不清楚 | 多阶段 auto-context (0.7237) |
| Top 3 方法 | 不清楚 | rictoo (0.6326) |

**关键差距来源**：前 3 名全部使用病理基础模型（Virchow2, ViT-H/14, 310 万张 WSI 预训练），而非 ImageNet 预训练编码器。

### 对 Exp11 的影响

原计划 Exp11 用 ConvNeXt V2 Base 编码器，但：
1. Baseline 不是 ConvNeXt V2 Base，而是 nnU-Net v2 — 用 ConvNeXt V2 Base 不一定对标 baseline
2. 真正拉开差距的是病理基础模型，不是通用架构
3. → 需要两条线并行

---

## 二、双线并行策略

### Line A：CamylaNet 架构对比（5 个架构）

从 CamylaNet 的 9 个 2D 架构中选 5 个：

| # | 架构 | 家族 | Trainer 类 | 选择理由 |
|---|------|------|-----------|---------|
| 1 | nnU-Net | 卷积 | `nnUNetTrainer` | 就是 PUMA baseline，必须作为对照组 |
| 2 | SwinUNETR | Transformer | `SwinUNETRTrainer` | 医学影像验证最充分的 Transformer |
| 3 | SegResNet | 卷积 | `SegResNetTrainer` | nnU-Net 系变体，稳定且高效 |
| 4 | U-Mamba | State-space | `UMambaTrainer` | 2024-2026 新趋势，Mamba 在医学分割有潜力 |
| 5 | U-Net++ | 卷积 | `UNetPlusPlusTrainer` | 经典嵌套 skip 改进，广泛验证 |

砍掉的 4 个：TransUNet（被 SwinUNETR 覆盖）、UTNet（学术影响小）、SwinUMamba（和 U-Mamba 重叠）、U-KAN（KAN 理论存争议）。

### Line B：病理基础模型接入

| 模型 | 架构 | 可用性 | 用途 |
|------|------|--------|------|
| UNI | ViT-L/16 | HuggingFace `mahmoodlab/UNI`（需申请，通常 1-2 天） | 主力编码器 |
| UNI2-h | ViT-h/14 | HuggingFace `mahmoodlab/UNI2-h`（需申请） | 备选（更大更强） |
| CONCH | ViT-B-16 | HuggingFace `mahmoodlab/conch`（需申请） | 备选（视觉-语言模型） |

接入方式：UNI 作为 U-Net 编码器 + 标准 U-Net 解码器，作为 CamylaNet 自定义 Trainer。

---

## 三、自动化实验流水线设计

```
┌─────────────────────────────────────────────────────────┐
│              AI 辅助科研流水线                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Step 1: 数据转换                                        │
│  convert_puma_to_nnunet.py                               │
│  .tif + GeoJSON → .nii.gz + dataset.json               │
│  （含 5-fold 分层划分文件）                                │
│                                                         │
│  Step 2: 架构对比 (Line A)                               │
│  experiment_runner.py                                   │
│  ┌───────────────────────────────────┐                  │
│  │ for arch in [nnUNet, SwinUNETR,   │                  │
│  │   SegResNet, U-Mamba, UNet++]:    │                  │
│  │   1. 1-epoch smoke test           │                  │
│  │   2. if pass → queue full run     │                  │
│  │   3. record results to JSON       │                  │
│  └───────────────────────────────────┘                  │
│                                                         │
│  Step 3: 基础模型对比 (Line B, 并行)                      │
│  foundation_model_trainer.py                             │
│  ┌───────────────────────────────────┐                  │
│  │ 1. Load UNI encoder (frozen)       │                  │
│  │ 2. Build UNet decoder on top       │                  │
│  │ 3. Smoke test → Full training      │                  │
│  └───────────────────────────────────┘                  │
│                                                         │
│  Step 4: 结果汇总                                        │
│  ┌───────────────────────────────────┐                  │
│  │ Compare all Dice scores            │                  │
│  │ Select best architecture           │                  │
│  │ Fine-tune: loss/TTA/ensemble       │                  │
│  └───────────────────────────────────┘                  │
│                                                         │
│  Step 5: Docker 提交                                     │
│  Rebuild with best model → test → submit                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 四、5 天执行计划（修订）

### Day 1：环境搭建 + 数据转换

**服务器端：**
```bash
# 1. 安装 CamylaNet
cd /root/autodl-tmp/medseg_project
git clone https://github.com/yifangao112/CamylaNet.git
cd CamylaNet && pip install -e .
export camylanet_raw="/root/autodl-tmp/camylanet_raw"
export camylanet_preprocessed="/root/autodl-tmp/camylanet_preprocessed"
export camylanet_results="/root/autodl-tmp/camylanet_results"

# 2. 转换 PUMA 数据
python convert_puma_to_nnunet.py \
    --data_dir /root/autodl-tmp/medseg_project/data/PUMA \
    --output_dir /root/autodl-tmp/camylanet_raw \
    --dataset_id 101

# 3. 预处理
python -c "
import camylanet
plans = camylanet.plan_and_preprocess(dataset_id=101, configurations=['2d'])
print(f'Plans: {plans}')
"
```

### Day 2：5 架构 Smoke Test

```bash
python experiment_runner.py \
    --dataset_id 101 \
    --architectures nnUNet SwinUNETR SegResNet UMamba UNetPlusPlus \
    --mode smoke_test
```

### Day 3：Top 3 全量训练 + 基础模型 Smoke Test

```bash
# Line A: 全量训练 top 3
python experiment_runner.py \
    --dataset_id 101 \
    --architectures <top3_from_smoke> \
    --mode full \
    --epochs 100

# Line B: UNI 基础模型
python experiment_runner.py \
    --dataset_id 101 \
    --architectures FoundationModel_UNI \
    --mode smoke_test
```

### Day 4：微调最佳模型

根据 Day 3 结果选最佳架构，进行：
- 损失函数消融（Focal vs CE+Dice vs Focal+Lovász）
- TTA 对比（无 TTA vs 水平+垂直翻转 vs 8 种增强）
- 集成策略（单折 vs 5 折平均 vs 概率平均）

### Day 5：Docker 重新打包 + 提交

```bash
python generate_puma_submit.py --create_docker --model <best_model> --target_size 512
```

---

## 五、可推广性

此流水线设计为通用框架，可推广到其他 2D 医学分割任务：

| 任务 | 数据集 | 转换脚本 | 架构 | 预期 |
|------|--------|---------|------|------|
| PUMA | H&E 黑色素瘤 | convert_puma_to_nnunet.py | 5 架构 + UNI | ≥ 0.55 |
| DRIVE | 视网膜眼底 | convert_drive_to_nnunet.py | 同 5 架构 | ≥ 0.83 |
| SIIM | 气胸 X-ray | convert_siim_to_nnunet.py | 同 5 架构 | ≥ 0.85 |
| BEETLE | 乳腺 H&E | convert_beetle_to_nnunet.py | 同 5 架构 | ≥ 0.88 |

每个新任务只需：
1. 写一个数据转换脚本（复用 mask 生成逻辑）
2. 跑 `experiment_runner.py --dataset_id <new_id>`
3. 选最佳架构微调

---

## 六、Divergent Feedback 机制

借鉴 Camyla 的失败反馈策略：

| 触发条件 | 动作 |
|---------|------|
| 同一方向失败 3 次 | 强制换策略 |
| 架构 smoke test 失败 | 记录错误，跳过，不阻塞其他架构 |
| 基础模型权重无法获取 | 回退到 ImageNet 预训练 |
| 全部架构不如 baseline | 优先考虑基础模型方向 |
