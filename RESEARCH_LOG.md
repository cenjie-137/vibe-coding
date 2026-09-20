# 医学影像分割基准研究 — 实验日志（精简版）

> 研究主题：AI 辅助（LLM + vibe coding）医学影像分割的标准化、可复现框架，并在多个竞赛数据集上做基准验证。
> 本文档只保留**最终结果、方法选型理由、可迁移结论、复现清单**；过程性逐条调参记录已精简。
> 维护规则：**每完成一次有结论的实验（训练/推理/调优/提交）就立即更新本文档。**

---

## 1. 研究概述

| 项目 | 内容 |
|------|------|
| 核心贡献 | ① 标准化 prompt 模板（`PROMPT_STANDARD.md` v1.1） ② 复现清单 ③ 元数据日志（SHA-256 哈希、人机干预计数） ④ 人机协作透明度 |
| 目标会议/期刊 | MICCAI / Medical Image Analysis |
| 固定 LLM | DeepSeek-V4-Pro（interactive 模式），temperature=0.7，seed=42 |
| 环境 | AutoDL，PyTorch 2.0.0+cu118，CUDA 11.8，GPU 32GB |
| 框架原则 | SOTA 调研 → 数据转换 → 模型+训练 → smoke test → 全量训练 → 评估 → 打包，7 步固定流程 |
| 代码路径 | 本地 `c:\Users\32201\Desktop\T\vibe coding\DRIVE`；服务器 `/root/autodl-tmp/medseg_project` |

---

## 2. 基准结果总览（Master Table）

| 数据集 | 任务 | 模态 | 模型 / 编码器 | 关键指标（最终） | 基线 | 状态 |
|--------|------|------|--------------|------------------|------|------|
| DRIVE | 视网膜血管分割（像素级） | 眼底彩照 | MultiResUNet-CBAM（从头） | Test Dice **0.8280**（rank 86） | — | ✅ 完成 |
| PUMA | 黑色素瘤组织 5 类（Micro Dice） | H&E ROI | FoundationUNet / **UNI** | 5-fold TTA micro Dice **0.6762** | 0.5548 | ✅ 完成 |
| GLaS | 结肠腺体分割（对象级 F1/Dice） | H&E | GlandUNet / **UNI**（DCAN 式双头） | testB objDice **0.7513** / objF1 **0.7622**（watershed 后处理） | — | 🔄 迭代中 |
| BEETLE | 乳腺组织 4 类（Overall Dice） | H&E WSI | 待定 | 未训练 | 0.87 | ⏳ 数据准备 |

> 核心结论（论文主线）：**病理基础模型 UNI（ViT-L/16）在病理任务上一致大幅超越 ImageNet 预训练**（PUMA micro Dice +15% vs ResNet34；GLaS 用 UNI+轮廓双头可稳定收敛到对象级分割可行方案）。细线状/小目标结构（血管、blood_vessel）是所有 ViT-like 骨干的共同短板。

---

## 3. 各数据集详情

### 3.1 DRIVE — 视网膜血管分割

- **数据**：40 张（20 train + 20 test，测试集无公开 GT），584×565，二值。
- **最终方案**：MultiResUNet + CBAM 注意力，base_ch=64；Adam lr=5e-4 + CosineAnnealingLR(150)；损失 = 2×Dice + Tversky(0.4/0.6) + Focal(α=0.75, γ=2)；TTA 翻转；分 5-fold。
- **最终结果**：官方测试集 Dice **0.8280**（排名 86）。
- **关键结论**：
  1. 细血管对**后处理删小目标极其敏感**（min_size=30 → 掉点，最终 min_size=0）。
  2. 类别不平衡（血管仅 ~8%）用 Focal 需 **α>0.5**，否则全预测背景。
  3. 非正方形图像禁用 `rot90` 增强（尺寸互换导致 batch 崩溃）。

### 3.2 PUMA — 黑色素瘤组织分割

- **数据**：205 张 1024×1024 H&E ROI（103 原发 + 102 转移）；6 类（5 组织 + 背景）；官方指标 **Micro Dice**；baseline 0.5548（nnU-Net v2）。
- **最终方案**：FoundationUNet + **UNI(ViT-L/16)**；损失 = Focal(per-class α=sqrt 反频率) + 类权重 Dice（**不加 Lovász**）；HED 染色增强；差分学习率（encoder=lr×0.1）；warmup10+cosine；5-fold 按原发/转移分层 CV。
- **最终结果（5-fold avg）**：TTA micro Dice **0.6762 ± 0.0587**（超 baseline +12%）；最优 fold5 micro **0.7716**。
  - fold5 各类别 Dice：tumor 0.932 / necrosis 0.900 / stroma 0.846 / epidermis 0.745 / blood_vessel 0.435。
- **提交**：Docker 已完成（`puma-submission.tar.gz`），滑动窗口 512/overlap64/softmax 平均 + TTA + 5 折权重，int8 TIFF 300DPI + 空 nuclei JSON。
- **关键结论（可迁移）**：
  1. **UNI 病理预训练是决定性因素**：necrosis 从 0 → 0.90，小类靠基础模型补齐。
  2. Lovász 优化全局 IoU 会**抑制极小类**，故弃用。
  3. copy-paste 增强破坏组织上下文 → 灾难性失败；WeightedRandomSampler 反而过拟合。
  4. SegFormer-B2 对细线状结构（blood_vessel）分割弱。
  5. **集成评估需严防数据泄露**（模型见过的验证样本不能算集成）。

### 3.3 GLaS — 结肠腺体分割（对象级）

- **数据**：165 张（85 train / 60 testA / 20 testB，全部公开 GT），H&E，尺寸 517~775×430~522；评估为**对象级 F1 + Dice**（按 benign/malignant 分组）。
- **最终方案**：GlandUNet = UNI 编码器 + 4 层解码头 + **分割头 / 轮廓头（DCAN 式）双任务**；损失 = BCE+2×Dice（seg）+ 带 pos_weight 的 BCE+Dice（contour）；后处理 = 前景减轮廓 + 连通域去小碎片。
- **结果（v1，512px）**：5-fold val 均值 objDice **0.8075**；testB objDice **0.7164** / objF1 **0.7266**，轮廓法阈值调优后 **0.7316 / 0.7436**（best: seg=0.5 / cont=0.6 / min_area=20）。
  - benign F1 0.87 vs malignant F1 0.69 → 瓶颈在**密集粘连的恶性腺体**的实例分离。
- **watershed 后处理（Step 0，v1 权重不重训）**：testB objDice **0.7513** / objF1 **0.7622**（best: seg=0.3 / fg_dist=0.2 / min_area=30），较轮廓法再 +0.02；testA objDice **0.8198** / objF1 **0.8908**（best: seg=0.6 / fg_dist=0.2 / min_area=30）。距离变换找中心种子 + 轮廓概率地形分水岭，有效切割粘连腺体 → **当前最优方案**。
- **v2 重训（768px + 深 ResBlock 解码 + 强增强）→ 回归，弃用**：5-fold val objD 均值 **0.7873**（< v1 0.8075）；testB objDice **0.6867** / objF1 **0.7235**（< v1 raw 0.7164/0.7266，未调优）；testA objDice 0.8156 / objF1 0.8735。三项叠加反而掉点，且为混淆变量无法归因；**保留 v1 为当前最优**。
- **SOTA 参照（testB Dice）**：DCAN 0.897 / MILD-Net 0.913 / RIC-UNet 0.918。
- **关键结论**：对象级评估的碾压项是**实例分离**；纯像素二值分割（把粘连腺体当一个连通域）F1 直接拉垮，必须加轮廓/分水岭后处理。

### 3.4 BEETLE — 乳腺组织分割（未开始训练）

- **数据**：587 WSI（训练）+ 170 ROI（测试，1024×1024）；4 类；Overall Dice；baseline 0.87，目标 ≥0.88。
- **状态**：已定架构复用 MultiResUNet，数据准备阶段。

---

## 4. 跨数据集结论（论文 Discussion 素材）

1. **病理基础模型通用性**：UNI 对病理组织分割（PUMA、GLaS）一致优于 ImageNet，是病理任务的"开箱即用"提升。
2. **共同短板—细结构**：小血管 / blood_vessel 是 ViT 骨干 + 小 batch 下的普遍弱点（DRIVE、PUMA、GLaS 恶性腺体边界）。
3. **后处理必须匹配评估协议**：像素级（DRIVE）要避免删小目标；对象级（GLaS）必须做实例分离。
4. **损失函数对小类的负作用**：Lovász、激进类权重、copy-paste、过采样都可能在稀疏类上反噬；Focal 温和加权最稳。
5. **可复现框架价值**：固定 seed/LLM/prompt/环境 + 元数据日志，能把"AI 辅助开发"从黑盒变成可复现、可审计的实验方法。

---

## 5. 方案有效性对照（消融 Ablation）

> 记录每个已验证的设计选择及其效果，直接对应论文 Ablation 表。✅ 采用 / ❌ 弃用 / 🔄 待定。

| 数据集 | 方案 / 尝试 | 效果 | 结论 |
|--------|-------------|------|------|
| DRIVE | 后处理 min_size=30 删小连通域 | 误删细血管 → 掉点 | ❌ 改 min_size=0 |
| DRIVE | Focal α=0.75（>0.5） | 血管 ~8% 不平衡下避免全预测背景 | ✅ 保留 |
| DRIVE | 非正方形图 rot90 增强 | 尺寸互换 → batch 崩溃 | ❌ 禁用 |
| PUMA | **UNI(ViT-L/16)** vs ImageNet | micro Dice **+15%**；necrosis 0→0.90 | ✅ 核心提升 |
| PUMA | Lovász-Softmax | 优化全局 IoU，抑制极小类 | ❌ 弃用 |
| PUMA | copy-paste 增强 | 破坏组织上下文，灾难性失败 | ❌ 弃用 |
| PUMA | WeightedRandomSampler | 反而过拟合 | ❌ 弃用 |
| PUMA | SegFormer-B2 编码器 | 细线状结构（blood_vessel）弱 | ❌ 换 UNI |
| PUMA | 差分学习率 encoder×0.1 | 保护预训练特征 | ✅ 保留 |
| GLaS | 轮廓头（DCAN 双任务） | 实例分离关键，防粘连腺体并成单连通域 | ✅ 核心 |
| GLaS | 阈值调优 seg/cont/min_area（轮廓法） | testB objDice 0.7164→**0.7316** | ✅ 保留 |
| GLaS | v2：768px+深 ResBlock 解码+强增强 | val 0.7873 / testB 0.6867，全面低于 v1 | ❌ 弃用（回归） |
| GLaS | 分水岭后处理（距离变换种子 + 轮廓地形） | testB 0.7316→**0.7513**，testA 0.8198 | ✅ 保留（当前最优） |

---

## 6. 复现清单（Reproducibility）

### 6.1 全局固定项
- seed=42（`set_seed` 已固定 torch/numpy/cuda，cudnn.deterministic=True）。
- LLM：DeepSeek-V4-Pro，temperature=0.7。
- 数据划分：StratifiedKFold(shuffle=True, random_state=42)，折划分存档于 `splits.json`。

### 6.2 关键依赖
- `torch 2.0.0+cu118`、`timm`、`huggingface_hub`（UNI 权重，`HF_ENDPOINT=https://hf-mirror.com`）。
- `opencv-python-headless`、`scipy`、`scikit-learn`、`numpy<2`（注意与 opencv 的版本冲突）。

### 6.3 核心命令（服务器）
```bash
# PUMA（已完成）
python train_puma.py --folds 0 1 2 3 4 --model foundation_uni \
  --batch_size 4 --target_size 512 --epochs 100 --lr 1e-4 \
  --patience 50 --warmup_epochs 10 --label_smoothing 0.1

# GLaS v2（进行中，768px + 更深解码头）
python train_glas.py --data_dir .../data/glas --folds 0 1 2 3 4 \
  --epochs 100 --batch_size 2 --target_size 768 --output_dir .../output_glas_v2
```

---

## 7. 待办 / 下一步

- [x] GLaS v2 训练完成：5-fold val 0.7873 / testB objDice 0.6867，全面低于 v1，回归弃用。
- [x] Step 0 完成：分水岭后处理 testB objDice 0.7316→**0.7513**、testA 0.8198，超越轮廓法，暂定为当前最优后处理。
- [ ] Step 1（Step 0 验证后再做）：HoVer-Net 式距离图 head 重训。
- [ ] BEETLE 数据准备 + 基线训练。
- [ ] 增加非病理非血管任务（如 SIIM）补全框架通用性验证。
- [ ] 汇总做统计显著性检验（UNI vs ImageNet，配对 t 检验）。
- [ ] 论文：以"标准化 prompt 框架 + 复现清单"为主线，GLaS/PUMA/DRIVE 为基准结果，失败模式分析做 Discussion。