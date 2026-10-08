# 医学影像分割基准研究 — 实验日志（精简版）

> 研究主题：**用 AI（LLM + vibe coding）自主调研、自主确定技术路线、自主选择甚至革新方法、尽量做到人不参与地完成医学分割任务，以结果（指标）为上，验证 vibe coding 的能力上限与痛点**；过程中沉淀标准化、可复现的 prompt 工作流与复现清单。
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

> **对照锚点（非 LLM 自主轨道）**：标准 **nnU-Net v2 默认配置**基线的结果与官方协议评估见 **第 6 节** —— DRIVE CV Dice **0.7990**；GLaS CV Dice **0.9276**、官方对象级 testA **0.8723 / 0.7739 / 67.31**、testB **0.8184 / 0.6113 / 118.85**。

> 核心结论（论文主线）：**用 vibe coding 做"AI 自主"分割——AI 调研、定路线、自由选/改方法，人尽量不参与，只认最终指标**。已有一正一反两个证据：① PUMA 用 UNI 冲到超 baseline +12% 的 SOTA 级效果（micro Dice 0.6762）；② GLaS 则暴露 vibe coding 的痛点——AI 的调研结论与实现决策"失联"，误锁"病理基础模型"，而更适配小样本的是 CNN 轮廓方法（DCAN/MILD-Net/RIC-UNet）。

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
- **方向纠偏（vibe coding 痛点 + 二次反转）**：早前调研已记录 SOTA 是 CNN（DCAN 0.897 / MILD-Net 0.913 / RIC-UNet 0.918），但实现时误用 UNI、testB 0.7513 落后 U-Net 基线 0.768，于是转向 CNN DCAN。**结果 CNN 全配置 val 仅 0.63~0.70，均低于 UNI v1 val 0.8075**：深监督开/关、encoder_lr 0.1→1.0 三组消融都不及 UNI。**结论回到 UNI**——testB 落后的真正瓶颈是跨集泛化 + 实例分离，而非编码器；"误用 UNI"的首次诊断是过度反应（仅凭 testB 单点否定，未用 val 交叉验证）。这是 vibe coding 第二个痛点：AI 的"纠偏"同样会跑偏。
- **官方评估协议核对结论**：GLaS 官方（`/Evaluation/`）为**对象级**三项指标——**Object F1（检测）**、**Object Dice（分割）**、**Object Hausdorff（形状相似度）**。实现要点：① 三者共用"最大重叠/最大 Dice"的 1:1 失配（非全图连通域平均）；② Object Dice 官方为**双向对称平均** `0.5·(Σ_s/Σ_s 的 n + Σ_g/Σ_g 的 n)`，社区存在"面积加权 vs 简单平均"的分歧，本项目采用简单平均（与官方描述一致）；③ 连通域合并用 `scipy.ndimage.label`，未在官方页明确 4/8 连通，本项目统一 4 连通并在 watershed 后处理做相邻实例挖除保证一致。已在 `eval_glas.py` 补齐 Object Hausdorff（此前缺失，用对称 Hausdorff 距离、边界点集计算）。
- **resize→letterbox 修复（preprocessing 硬伤）**：原实现把 GLaS 非正方形图像（517~775×430~522）**直接 resize 到 512×512，拉伸变形腺体**；已改为 `letterbox_resize`（长边缩放 + 置中 pad 到正方形，保持纵横比），训练 `dataset_glas.py`、推理 `inference_glas.py`、tuning `tune_glas.py` 三处同步，推理端 `unletterbox` 裁回原图后评估。代价：pad 区占显存，有效分辨率略降；收益：形状不失真，对象级边界更准。**此前所有 val/test 数字均基于被拉伸的旧预处理，letterbox 修复后需重训+重评才有可比性。**
- **letterbox 重训（GlandUNet/UNI，512px）val 汇总**：5-fold objDice = 0.7543 / **0.8237** / 0.7684 / 0.7864 / 0.7836，**mean 0.7833**。对比旧拉伸预处理 v1 val 均值 0.8075，**降 0.024**。解读：旧 val 虚高（train/val 同用拉伸分布，模型学到变形几何、val 又用同样变形评估）；letterbox 修正后 val 更"诚实"。**真正的结论要等 testB/testA 跨集泛化**。
- **letterbox 重训的 testB/testA 泛化（watershed 后处理）**：testB objDice **0.7258** / objF1 **0.7752**（best seg=0.6 / fg_dist=0.2 / min_area=5~30）；testA objDice **0.7939** / objF1 **0.8659**（best seg=0.6 / fg_dist=0.2 / min_area=30）。对照旧拉伸 v1（testB **0.7513** / testA **0.8198**），**全面微降 ~0.026**，未兑现"修复几何失真 → 提升泛化"的预期。结论：① 几何失真**不是** testB 落后的主因，坐实"瓶颈在实例分离 / 跨集泛化"；② 0.025 落在 fold 波动（0.07）内，单次训练无法因果判定 letterbox 属有害还是噪声——这也是 vibe coding 的又一个痛点：AI 认定 resize 是"硬伤"并投入修复，实测却无收益。
- **HoVer-Net 式距离图头（已训练）**：给 GlandUNet 加第 3 头 `dist_head`（2 通道，输出每前景像素到其所属腺体质心的方向距离向量 dy/dx）。三处实现要点（也是修正初版硬伤）：① GT 用**全局统一归一化**（整图最大距离归一，非逐实例归一）——保证大小腺体梯度能量尺度一致；② `dist_head` 输出**不加 tanh**（腺体远大于细胞核、逐像素距离可达上百，tanh 会饱和）；③ 损失 = 前景 MSE + λ·MSGE（梯度一致性，HoVer-Net 关键项，用 `torch.gradient` 实现）。另修复 `inference_glas.py` 中 `--method hover` 传了不存在的 `seed_quantile` 关键字（应为 `fg_dist_thresh`）会导致 TypeError 的 bug。后处理 `postprocess_hover`：距离变换中心种子 + 距离图方向场梯度能量作分水岭地形。冒烟测试：① 全局归一化使大/小腺体内部梯度能量比 = **1.00**（逐实例归一化会差 6 倍）；② 薄颈相触双腺体成功切成 2 实例。
- **HoVer 距离图头训练结果（letterbox，5 折 val objDice）**：fold 0..4 = 0.7324 / 0.7862 / 0.7751 / 0.7414 / 0.7526，**mean 0.7575**；极差 0.054、std≈0.02。对比同预处理 letterbox 基线 mean 0.7833，**-0.026（约 1σ，落在 fold 波动内）**——不能判定有害，但**至少未带来 val 提升**。val 非终极判据（HoVer 目标是 testB 实例分离 + 跨集泛化），**待 testB/testA 评估后定论**。

### 3.4 BEETLE — 乳腺组织分割（未开始训练）

- **数据**：587 WSI（训练）+ 170 ROI（测试，1024×1024）；4 类；Overall Dice；baseline 0.87，目标 ≥0.88。
- **状态**：已定架构复用 MultiResUNet，数据准备阶段。

---

## 4. 跨数据集结论（论文 Discussion 素材）

1. **病理基础模型并非万能（任务依赖）**：UNI 在 PUMA（多类/类别极不平衡/病灶小）上大幅领先（+12~15%），但在 GLaS（二值/小样本/大腺体）上与 U-Net 基线持平，不能默认所有病理任务都套 ViT 基础模型。
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
| GLaS | 误用 UNI(ViT-L) 编码器 + 无 skip 解码头 | testB 0.7513，落后 U-Net 基线 0.768 | ❌ 方向带偏（vibe coding 痛点） |
| GLaS | 转向 CNN DCAN：ResNet34+skip+双头+深监督 | Fold0 val objD **0.6489**（< UNI v1 0.8075）| ❌ 翻车 |
| GLaS | CNN DCAN + 关深监督（`--deep_sup 0`） | Fold0 val objD **0.6342** | ❌ 深监督非元凶，CNN 主干弱于 UNI |
| GLaS | CNN + 关深监督 + encoder_lr×1.0 | Fold0 val objD **0.7049**（> lr0.1 的 0.6342）| ❌ lr 是因子(+0.07)，仍远低于 UNI 0.8075 → 放弃 CNN |
| GLaS | 图像直接 resize 到 512×512 | 非正方形图被拉伸，腺体形状/边界变形 | ⚠️ 理论修正；实测 letterbox 无收益（见下） |
| GLaS | letterbox 保持纵横比（vs 直接 resize） | val 0.8075→0.7833、testB 0.7513→0.7258、testA 0.8198→0.7939，全面微降 ~0.025 | ❌ 无收益：几何失真非主因，或在 fold 噪声(0.07)内 |
| GLaS | 缺 Object Hausdorff 指标 | 评估协议不完整，无法对照 SOTA 的形状相似度 | ❌ 已补齐（对称 Hausdorff + 边界点集） |
| GLaS | HoVer 距离图头（全局归一化 + MSE + MSGE + 无 tanh + hover 分水岭） | 5 折 val objDice **0.7575**（< letterbox 基线 0.7833，-0.026）；testB/testA 待测 | 🔄 待定（val 未兑现，等跨集评估；推理用 `--method hover`） |

---

## 6. nnU-Net v2 标准基线轨道（对照，非 LLM 自主）

> **定位与分区**：本节与第 2~5 节的 LLM/vibe coding 轨道**严格分离**。本轨道全程使用**官方 nnU-Net v2 默认配置**（2d 配置、默认损失/增强/调度，无任何自定义编码器或后处理），seed=42，按各数据集官方协议评估，目的是给出"标准强基线"，作为 LLM 自主方案（第 3 节）的对照锚点。
> 产物：`nnunet_baseline_results.json`（机器可读）、`NNUNET_BASELINE_AGENT_PROMPT.md`（流程）、服务器 `/root/autodl-tmp/nnUNet`。

### 6.1 结果总览

| 数据集 | 配置 | 内部 5 折 CV 像素 Dice（逐折） | 测试集像素 Dice | 状态 |
|--------|------|-------------------------------|----------------|------|
| DRIVE (Dataset071) | nnU-Net v2 2d，官方默认 | **0.7990**（0.7763 / 0.8142 / 0.8172 / 0.8082 / 0.7792） | —（测试集无公开 GT，已产出提交包） | ✅ 完成 |
| GLaS (Dataset070) | nnU-Net v2 2d，官方默认 | **0.9276**（0.9193 / 0.9253 / 0.9336 / 0.9211 / 0.9388） | **0.9199**（testA 0.9276 / testB 0.8966，n=80） | ✅ 完成 |

- GLaS 5 折均跑满 1000 epoch（无提前停止），5 折 ensemble（`checkpoint_final.pth`）推理 testA 60 + testB 20 = 80 张。
- 测试集像素 Dice（0.9199）与 CV（0.9276）仅差 0.008 → 划分正常，无泄漏或分布断裂迹象。
- 划分与官方 Table 1 一致（train 85 / testA 60 / testB 20）；本地 `data/glas/masks` 与官方 `*_anno.bmp` 二值化逐像素一致，实例标签数 == 4 连通域数。

### 6.2 GLaS 官方对象级评估（5 折 ensemble，官方协议）

口径：官方 `/Evaluation/` —— 最大重叠 1:1 对应、**面积加权 + 数据集级池化**、F1 判定 `|S∩G| ≥ 0.5·|G|`。

| 子集 | n | Object Dice | Object F1 | Object HD |
|------|---|-------------|-----------|-----------|
| testA | 60 | **0.8723** | **0.7739** | **67.31** |
| testB | 20 | **0.8184** | **0.6113** | **118.85** |
| 全部 | 80 | 0.8567 | 0.7458 | 82.28 |
| benign | — | 0.8955 | 0.8253 | 48.88 |
| malignant | — | 0.8219 | 0.6228 | 112.25 |

**外部对照（已发表方法，同协议）**：DRU-Net（IET IPR 2020）testA F1 0.828 / ObjDice 0.8695 / ObjHD 59.39，testB F1 0.5998 / ObjDice 0.7800 / ObjHD 150.65；TA-Net 汇总表中 SegNet F1 83.1 / ObjD 84.9 / ObjH 76.6，DCAN F1 81.4 / ObjD 83.9 / ObjH 102.9。
→ 本基线落在已发表方法的正常区间；**testB 的高 ObjHD（~119）是该协议下的普遍现象，不是实现错误**。

### 6.3 假阳性碎片消融（后处理）

nnU-Net 默认不做实例分离、也不剔除小目标，输出含大量 1px 级假阳性碎片。

| 后处理 | testA ObjF1 | testB ObjF1 | testA ObjDice | testB ObjDice |
|--------|-------------|-------------|---------------|---------------|
| 无（raw） | 0.7739 | 0.6113 | 0.8723 | 0.8184 |
| 3×3 开运算 ×1 | 0.8261 | 0.6791 | 0.8732 | 0.8185 |
| 3×3 开运算 ×2 | 0.8457 | 0.7054 | 0.8737 | 0.8183 |
| 3×3 开运算 ×3 | 0.8567 | 0.7309 | 0.8738 | 0.8184 |

**结论**：ObjDice 几乎不动（面积加权对碎片不敏感），但 ObjF1 可白涨 **+0.08（testA）/ +0.12（testB）**，代价仅删掉约 100~700 个像素。Object HD 同样主要被这些碎片拉高——未加权中位数仅 **12.04 px**（testA），而面积加权均值 67.31。

### 6.4 跨轨道对比（含口径警示）⚠️

**口径差异是首要混淆项**：LLM 轨道（第 3.3 节）的 Object Dice 用**逐图简单平均**，官方协议为**面积加权 + 数据集级池化**。用同一份 nnU-Net 预测做口径对照：

| 口径 | testA ObjDice | testB ObjDice |
|------|---------------|---------------|
| 官方（面积加权 + 数据集池化） | 0.8723 | 0.8184 |
| 逐图简单平均（LLM 轨道口径） | 0.7507 | 0.6544 |

**仅口径就造成 +0.12（testA）/ +0.16（testB）的差异**。因此跨轨道比较必须在同口径下进行：

| 方案 | 口径 | testA ObjDice | testB ObjDice |
|------|------|---------------|---------------|
| LLM：GlandUNet/UNI + watershed（v1 拉伸） | 逐图简单平均 | 0.8198 | 0.7513 |
| LLM：GlandUNet/UNI + watershed（letterbox） | 逐图简单平均 | 0.7939 | 0.7258 |
| **nnU-Net v2 基线（raw）** | 逐图简单平均 | 0.7507 | 0.6544 |
| **nnU-Net v2 基线（raw）** | 官方 | 0.8723 | 0.8184 |

**同口径（逐图简单平均）下，LLM 的 GlandUNet/UNI + watershed 仍优于 nnU-Net raw**（testB 0.7513 vs 0.6544；testA 0.8198 vs 0.7507），差值主要来自**分水岭实例分离后处理**（nnU-Net raw 无此步骤，粘连腺体并成单连通域）。但若给 nnU-Net 加上同等后处理、或改用官方口径，差距会大幅收窄。
⚠️ **这提示第 3.3 节"testB 落后 U-Net 基线 0.768"的结论依赖口径，需在官方口径下重新推导。**

> ⚠️ **阻塞项**：LLM 轨道的 GLaS 预测已不在服务器上（`output_glas*` 已清理），无法就地重评。需重跑一次 LLM 轨道推理，再用 `eval_glas_official.py` 统一口径评估，才能给出可信的跨轨道结论。

### 6.5 复现清单（nnU-Net 轨道）

```bash
# 训练（每折 setsid nohup 后台启动，断网不影响）
export nnUNet_raw=.../nnUNet/raw nnUNet_preprocessed=.../nnUNet/preprocessed nnUNet_results=.../nnUNet/results
nnUNetv2_train 070 2d $FOLD          # FOLD=0..4（GLaS）
# 5 折 ensemble 推理
nnUNetv2_predict -i .../raw/Dataset070_GLAS/imagesTs -o .../glas_test_pred \
  -d 070 -c 2d -f 0 1 2 3 4 -tr nnUNetTrainer -p nnUNetPlans -chk checkpoint_final.pth
# 官方口径对象级评估
python eval_glas_official.py --gt_dir data/glas/masks --pred_dir glas_test_pred
```

- 评估脚本：`eval_glas_official.py`（面积加权 + 数据集级池化 + 无重叠时双向 min-HD 退化）；诊断脚本：`.ssh_keys/diag_*.py`。
- ⚠️ 官方 MATLAB 评估代码需 Warwick 登录，下载失败（`evaluation_v6.zip` 损坏），三项对象级指标为**按官方 Evaluation 页公式的忠实复现**，非官方代码输出；若后续拿到官方代码应复核。

### 6.6 外部基线核实与「U-Net 0.768」溯源（2026-09-29）

**结论：此前 LLM 轨道使用的「U-Net 基线 0.768」是口径错配的伪基线，不成立。**

| 候选来源 | 数值 | 真实口径 | 是否为 testB 对象级 Dice |
|---------|------|---------|------------------------|
| HEA-Net (MDPI Sensors 2022) / TAC-UNet (QIMS 2024) | IoU **76.81 / 76.84%**（Dice 86.3%） | **像素级**语义分割 | ❌ 指标类型 + 子集双错 |
| J Integr Bioinform 2025 (PMC12569585) | F1 **0.766**（testA） | 对象级 **F1**，作者自训 | ❌ 指标类型 + 子集双错 |

→ 文献中不存在「U-Net testB 对象级 Dice = 0.768」。LLM 把像素级 IoU 或 testA 的 F1 当成了 testB 对象级 Dice 基线。**这是本研究记录的 vibe coding 痛点实例：口径混淆 → 错误基线 → 错误结论（"落后基线"）。**

**已发表方法的 GLaS 官方对象级数值（供对照，多源交叉一致）：**

| 方法 | testA F1 | testA ObjDice | testA ObjHD | testB F1 | testB ObjDice | testB ObjHD | 来源 |
|------|---------|--------------|-------------|---------|--------------|-------------|------|
| U-Net (2015) | 0.766 | — | 76.51 | — | — | — | JIB 2025 |
| CUMedVision1 (2015) | 0.868 | 0.867 | 74.60 | 0.769 | 0.800 | 153.65 | GLaS'15 / Micro-Net |
| **DCAN = CUMedVision2 (2015)** | **0.912** | **0.897** | 45.42 | 0.716 | 0.781 | 160.35 | GLaS'15 / EUSIPCO'22 |
| **MILD-Net (2019)** | **0.914** | **0.913** | 41.54 | **0.844** | **0.836** | 105.89 | EUSIPCO'22 / DoubleU-Net |
| **nnU-Net v2 agent（本项目）** | 0.774 | 0.872 | 67.31 | 0.611 | **0.818** | 118.85 | 本项目实测 |

**关键判读**：
1. **无公开 nnU-Net GLaS 基线**——nnU-Net 原论文未收录 GLaS，文献中亦无常用 nnU-Net-GLaS 对象级数字。本项目此次运行的 nnU-Net v2 默认配置基线**即该任务的 nnU-Net 参照**，可作为论文补充贡献。
2. 我们 testA ObjDice 0.872 接近 DCAN(0.897)，ObjF1 0.774 略高于 U-Net(0.766)，但与 MILD-Net 仍有差距。
3. testB **ObjDice 0.818 反超 DCAN(0.781)**，但 **ObjF1 0.611 为全表最低**——"分割质量好、实例分离差"的典型特征，瓶颈确认在实例分离。
4. testB ObjHD 偏大（118.85）由假阳性碎片导致，与该协议下已发表方法同量级（DCAN 160.35 / CUMedVision1 153.65），非实现错误。

**组会交付物**：`glas_baseline_comparison.html`（含主对比表 + testA/testB 分组柱状图 + 口径警示 + 讲稿要点）。

### 6.7 待办（nnU-Net 轨道）

- [x] GLaS 5 折训练完成、5 折 ensemble 推理完成、官方口径对象级评估完成。
- [ ] 把小目标剔除（≥30px）+ 1~2 次开运算加进推理后处理：ObjF1 testA 0.774→0.857、testB 0.611→0.731。
- [ ] 复核 testA_14 / testB_14（像素 Dice ~0.617）是否为标注/染色异常样本。
- [ ] 重跑 LLM 轨道 GLaS 推理并在官方口径下重评，替换第 3.3 节的跨轨道结论。
- [ ] （可选）nnU-Net 3d 配置或更强增强，验证 testB 能否提升。

### 6.8 路线池开源实现核实（Step 0，2026-10-08）

**目的**：进入自主循环前，先确认"打算抄的那几本参考书真的能借到"。核实用的是可复现手段：GitHub 仓库搜索 → 打开仓库确认文件 → 回论文原文取数。

**结论 1｜三条高知名度路线，都没有官方开源代码**

| 路线 | 搜索方式 | 命中情况 | 判定 |
|------|---------|---------|------|
| MILD-Net (MIA 2019) | `MILD-Net in:name` / `user:simon-graham` | 仅 Vinnni002/MILD-Net（1★，第三方）；作者账号 0 仓库 | 无官方代码 |
| DCAN (CVPR 2016) | `DCAN gland segmentation contour aware` | 仅 ShamilKaleel/DCAN-Gland-Segmentation（2026-10-08 新建、**0 字节**、0★） | 空壳，不可用 |
| RIC-Unet (2019) | `RIC-Unet gland segmentation` | 0 条 | 无代码；且原文任务域疑为**细胞核**而非腺体 |

→ 按红线 R4，三条一律 `confidence: low`，**不作复现锚点**；其数值仍可作参考线。

**结论 2｜找到两条真实可用、已打开确认含代码的实现**

| 路线 | 仓库 | 核实到的文件 | 报告口径 |
|------|------|-------------|---------|
| **R05 U-Node**（Pinckaers & Litjens, 2019） | `DIAGNijmegen/neural-odes-segmentation`（111★） | models.py / dataloader.py / metrics.py / train_models.ipynb / inference_utils.py，README 附 Colab | **官方对象级** ✅ |
| R06 GLaS U-Net | `twpkevin06222/Gland-Segmentation`（28★） | train.py / model.py / metric.py / weights/ | **像素级** ❌ 口径不对口 |

**结论 3｜口径陷阱实录（红线 R1 活教材）**：R06 的 README 明确写结果按 *pixel-wise* 评估、"instead of gland-wise"，其 testB Dice 0.91 与我们的对象级 0.8184 **不是一把尺子**，并列即错。

**结论 4｜修正一处来源错误**：原 R02 的 paper 指向 `arXiv:1603.00275`，核实后那是 **GLaS 挑战赛综述论文**（Sirinukunwattana et al.），非 DCAN 原文；DCAN 原文为 `arXiv:1604.02677`，已更正（claims.jsonl 同步修正）。

**新增可核对的官方对象级数值（来源：NODE 论文 arXiv:1910.10470 Table 1）**

| 方法 | Object Dice (A, B) | F1 (A, B) | 备注 |
|------|-------------------|-----------|------|
| U-Node | 0.881 (0.893, **0.842**) | 0.861 (0.882, 0.801) | 2m 参数；**有代码，可作锚点** |
| U-Net | 0.868 (0.884, 0.819) | 0.841 (0.865, 0.768) | 30m 参数 |
| Chen et al. = DCAN | 0.868 (0.897, 0.781) | 0.863 (0.912, 0.716) | 与 §6.6 表一致 ✅ 交叉核对通过 |
| Graham et al. = Rota-Net | 0.902 (0.919, 0.849) | 0.896 (0.920, 0.824) | ⚠ 注意：这是 **Rota-Net**，勿与 MILD-Net 混 |

**落地配置**：`ROUTE_POOL.yaml` 重写（R01–R03 标 `code:"<无>"`+low；新增 R05/R06）；`autorun/seed_claims.py` 账本扩到 12 条、来源覆盖率 87.5% → **91.7%**。

**遗留待人工确认（不阻塞）**：
- MILD-Net 的 0.836/0.913 沿用 §6.6 的二次来源（EUSIPCO'22 / DoubleU-Net 复现表），本次**未从原文表格重取**；MIA 版 DOI 有两个候选，建议人工确认。
- U-Node 的 testB ObjDice **0.842 > 现用 L2 0.836**。是否上移 L2 属**判定字段**，按红线 R7 由人决定，本文件不擅自改动。

**复现清单（可逐条重跑）**：
```text
# 1) 查官方代码是否存在（GitHub 搜索）
query: "MILD-Net in:name"                          → 仅第三方 1★
query: "DCAN gland segmentation contour aware"     → 仅空壳仓库
query: "RIC-Unet gland segmentation"               → 0 条
query: "user:simon-graham"                         → 0 仓库
# 2) 确认可用仓库含代码
GET  github.com/DIAGNijmegen/neural-odes-segmentation  → 见上表文件清单
GET  github.com/twpkevin06222/Gland-Segmentation       → 见上表文件清单
# 3) 回原文取数（对象级口径）
GET  arxiv.org/abs/1910.10470        → Table 1
GET  arxiv.org/abs/1603.00275        → 确认为 GLaS 挑战赛综述（用于纠错）
GET  arxiv.org/abs/1604.02677        → DCAN 原文
# 4) 落库并自检
python autorun/seed_claims.py --reset
python autorun/redlines.py --audit    → 13 项检查，0 违规
python autorun/acceptance.py          → 8/8 通过
```

---

### 6.9 判定字段变更记录（policy v1.3 → v1.4，2026-10-08）

**变更内容（仅两处，均由人决定）**

| 项 | 旧值 | 新值 | 理由 |
|---|---|---|---|
| GLaS **L2 冲刺线** | 0.836（MILD-Net） | **0.842（U-Node）** | U-Node 有可复现开源代码，能同时充当"口径对齐锚点"；MILD-Net 无官方代码，撑不起锚点角色 |
| 路线池 **R03_RICUnet** | 在池内 | **已移除** | 已核实 4 个独立来源：RIC-Unet 原文做的是**细胞核**分割（Zeng et al., IEEE Access 2019），非腺体；且无实现、原值 0.918 无来源（违反 R2） |

**连带改动（L2 变了，旧用例必然失效 —— 这正是"判定字段不可随意改"的原因）**

| 文件 | 改动 | 为什么必须改 |
|------|------|-------------|
| `AUTONOMY_POLICY.md` | v1.3 → v1.4；目标表 L2 0.836 → 0.842 | 人机唯一接口，须与 policy.json 一致 |
| `autorun/datasets.yaml` | GLaS `l2` / `l2_source`；meta 版本 1.0 → 1.1；修正"14 个"→"15 个" | 目录是 policy.json 的唯一来源 |
| `autorun/policy.json` | 由 `build_policy.py --write` 编译生成 v1.4（已自动备份） | 判定器唯一数据源 |
| `autorun/judge.py` | 自检用例「达标L2」的 best 0.84 → **0.845** | 0.84 < 0.842，旧用例会假失败 |
| `autorun/replay.py` | demo 轨迹末值 0.840 → **0.845**；`route_total` 4 → 5 | 同上；路线池少了一条 |
| `ROUTE_POOL.yaml` | L2 上移；删除 R03（保留"为什么删"的注释） | 路线池是 Step 1 的输入 |
| `autorun/episodes/EP01_GLaS_demo/` | **整包重新生成**（policy_snapshot 变为 v1.4） | 见下方 ⚠ |

**⚠ 本次暴露的一个真实缺口（下一步要修）**

`replay.py` 重放时用的是**当前** `policy.json`，而不是 episode 自带的 `policy_snapshot.json`。
后果：**只要改一次判定字段，所有历史证据包立刻"不可重放"** —— 本次改 L2 后 EP01 就必须整包重生成，
否则 5 条决策里会有 1 条对不上（0.84 ≥ 0.842 不成立）。

这不是 bug 而是设计没做完：可信性要求"历史证据永远可复核"，那就必须
**把重放钉死在当时的规则版本上**。修法见 §6.10（已修）。

**验收**：红线审计 12 项 0 违规；一键验收 8/8 通过；来源覆盖率 91.7%；EP01 重放 5/5 一致
（policy v1.4，链哈希 `sha256:ac2864699e42499bf0a117ee59fbe793`）。

---

### 6.10 治理收口（policy v1.4 → v1.5，2026-10-08）：锚点隔离 R10 + 重放钉死规则版本

**背景**：§6.9 留下两个待办 ——（a）复现别人的方法所得分数有被当成"自身成绩"的风险；（b）重放用的是"当前 policy"，改一次阈值历史证据就失效。本节把两者一次收口，并把 GLaS 起跑态固定下来。

**变更 1｜新增红线 R10「锚点隔离」**
- 复现（别人的方法）所得分数**只作口径校验**，写进 `state.anchor`；**不得**计入 `state.best`。
- `state.best.source` 必须为 `ours`（`best` 只记自研 / 改进模型的结果）。
- **双重把关**：`judge.decide()` 遇到 `best.source != "ours"` 直接 `stop / ANCHOR_IN_BEST`；`redlines.check_r10` 审计 `state.json` 与所有 episode 的 state。
- 理由：把复现值写成 `best` = 拿他人论文的数字冒充达标，直接摧毁可信度根基。这是本项目区别于"刷榜式"自主 Agent 的关键。

**变更 2｜重放钉死 episode 自带规则版本**
- `replay.py` 新增 `resolve_policy()`：优先用 trace 同目录的 `policy_snapshot.json`，缺失才回退当前 policy；报告新增 `policy_source` 字段（`snapshot` / `current` / `explicit`）。
- 新增对抗自检：同一份历史，用快照重放 **5/5 一致**；强行用"当前 policy"重放 → **必对不上**（1 条不一致）。把"为什么必须钉死"变成可跑的用例。

**变更 3｜留痕强制带 state_snapshot**
- `judge.py` 新增 `--log`：判定后直接写 `run_log.jsonl`，**强制带 state_snapshot**（缺失则账本拒收 + `replay` 判不合格）。

**变更 4｜起跑态与证据包**
- `autorun/state.json` 重写为 GLaS **起跑态**：`current_route=R05_UNODE`、`route_index=0`、`route_total=5`、`best={value:0.0, source:"ours"}`、新增 `anchor`（U-Node 0.842，待复现回填）、`policy_version=1.5`。
- 新建 `autorun/episodes/EP02_GLaS_real/`：真实 GLaS episode 的**起跑包**（state + `policy_snapshot.json` v1.5 + 开跑留痕）—— 规则版本在起跑那一刻即被钉死。
- `autorun/episodes/EP01_GLaS_demo/` 整包重新生成（快照 v1.5）。

**验收**：一键验收 **8/8**；红线审计 **15 项 0 违规**；来源覆盖率 91.7%；EP01 重放 5/5 一致（policy v1.5，链哈希 `sha256:ac2864699e42499bf0a117ee59fbe793`）。
各组件对抗用例：judge **11/11**、ledger 10/10、redlines 27/27、replay **7/7**。

**复现清单（可逐条重跑）**
```text
python autorun/build_policy.py --check     → targets/catalog/eval 无变化，version 1.4→1.5
python autorun/build_policy.py --write     → 编译 v1.5（自动备份 + 留痕）
python autorun/judge.py --selftest         → 11/11（含 R10 对抗用例）
python autorun/replay.py --selftest        → 7/7（含"钉死规则版本"对抗用例）
python autorun/replay.py --demo            → 重生成 EP01 证据包
python autorun/judge.py --log              → 起跑判定 IN_PROGRESS，写 run_log（带快照）
python autorun/redlines.py --audit         → 15 项，0 违规
python autorun/acceptance.py --html        → 8/8
```

---

### 6.11 U-Node 复现：数据 / 评估脚本就位 + 口径校验锚点（2026-10-08）

<a id="glas-unode-repro"></a>

**目的**：阶段 1 的第一件事不是训练，而是先把「尺子」钉死 —— 数据与评估脚本就位，
并用一条**有官方代码、报告官方对象级指标**的路线（R05 U-Node）跑通，
验证我们的评估管线口径是否对齐。复现所得分数按**红线 R10** 只作口径校验，
写 `state.anchor`，**不计入 `state.best`**。

**做了三件事**

1. **数据与评估脚本就位（本地）**：`data/glas/`（images / masks / metadata.csv / splits.json）；
   `eval_glas_official.py` 在既有 nnU-Net 预测上自检 → testB ObjDice = **0.8184**，与第 6 节归档值一致。
2. **取回官方仓库并跑通推理**：`DIAGNijmegen/neural-odes-segmentation`
   （含 models.py / dataloader.py / metrics.py / train_models.ipynb / inference_utils.py）
   + 官方权重 `best_border_unode_paper.pt`（state_dict 零缺失加载）。按官方
   `inference_utils.inference_image`（三向 TTA + postprocess：阈值 0.7 / 去小目标 500 / 生长 / 填洞）
   在 testA(60) + testB(20) 上推理；依赖补装 `torchdiffeq==0.2.5`。
   产出：`glas_unode_pred/`（二值，80 张）、`glas_unode_pred_label/`（实例标签，80 张）、
   `_unode_dl/unode_metrics.json`。
3. **用官方 metrics.py 评估** → 复现值：

| 子集 | 复现值（U-Node 官方口径） | 论文报告值 | 差 |
|---|---|---|---|
| testA ObjDice | **0.8872** | 0.893 | −0.006 |
| testB ObjDice | **0.8428** | 0.842 | +0.001 |
| testB ObjF1 | 0.6754 | 0.801 | −0.126 |

testB ObjDice 0.8428 与论文 0.842 吻合 → **复现本身可信，锚点成立**。
（F1 偏低，因官方权重只复现了 Dice 分支的报告水平；Dice 才是我们的判定指标。）

**意外收获：抓到一个 ≈0.02 的口径陷阱**

同一批预测，改成二值输出后用本仓库 `eval_glas_official.py` 评估，testB 只剩 **0.8215**。
为定位差异，新增 `audit_glas_metric_protocol.py`，把评估协议拆成
**连通性(4/8) × 匹配(argmax/mode) × 加权(weighted/unweighted) × 聚合(pooled/per_image)**
共 16 组做对照（testB 结果）：

| 预测形态 | 我们的口径<br>4/argmax/weighted/pooled | U-Node 口径<br>8/mode/weighted/per_image |
|---|---|---|
| 实例标签输入 | **0.8440** | **0.8433** |
| 二值（重做连通域） | **0.8215** | **0.8221** |

**根因结论**：那 0.02 的差距**不是**连通性 / 匹配规则 / 聚合方式造成的
（在加权聚合下，这几项彼此相差 <0.002），而是 **「实例标签 vs 二值重标注」**：
U-Node 的 `grow_to_fill_borders` 让相邻腺体在二值图上粘连、但在标签图里仍是独立对象；
二值重标注会把它们合并（抽样 5 图：89 → 50 个对象），从而系统性低估对象级 Dice。
**这是本项目记录的又一次「口径混淆 → 错误结论」实例（红线 R1）。**

**复现清单（可逐条重跑）**
```text
# 本地：评估脚本自检（用既有 nnU-Net 预测）
python eval_glas_official.py --gt_dir data/glas/masks --pred_dir glas_test_pred --split testB

# 服务器：U-Node 推理（官方权重 + 官方后处理）
python unode_repro.py            # 产出 glas_unode_pred/ 与 glas_unode_pred_label/
# 官方口径评估 → testB 0.8428；二值口径评估 → testB 0.8215

# 本地：16 组协议对照（定位口径差来源）
python audit_glas_metric_protocol.py --gt_dir data/glas/masks --pred_dir glas_unode_pred_label --labeled_pred
python audit_glas_metric_protocol.py --gt_dir data/glas/masks --pred_dir glas_unode_pred
```

**遗留（需人工裁定，见 §8.2）**：L2 线 0.842 取自 U-Node 的**实例标签**口径，
而本仓库 `eval_glas_official.py` 目前只吃**二值**。两者不同口径，
现在拿二值口径去够 0.842，等于"苹果比橘子"（红线 R1）。
已登记为 `state.pending_human_decisions[GLaS_EVAL_CONVENTION]`，阻塞 Step 2。

**顺带触发一条红线（系统自己抓到的设计缺口）**：把上述校准数字写进 `claims.jsonl` 后，
红线 **R1** 报 2 项违规 —— 它与原有官方口径数字同 `metric + subset`，被判为"跨口径并列"。
但这两条数字的用途**恰恰是记录口径差异**、不是拿来比较的，且 R10 已保证它们进不了 `best`。
说明 R1 的实现比规则文本更严（缺"校准类数字"豁免）。因 `redlines.py` 属 R7 保护文件，
**Agent 不擅自修改**，已登记为 `state.pending_human_decisions[R1_CALIBRATION_EXEMPTION]`；
在此之前一键验收为 **7/8**（其余 7 项全过）。

---

## 7. 复现清单（Reproducibility）

### 7.1 全局固定项
- seed=42（`set_seed` 已固定 torch/numpy/cuda，cudnn.deterministic=True）。
- LLM：DeepSeek-V4-Pro，temperature=0.7。
- 数据划分：StratifiedKFold(shuffle=True, random_state=42)，折划分存档于 `splits.json`。

### 7.2 关键依赖
- `torch 2.0.0+cu118`、`timm`、`huggingface_hub`（UNI 权重，`HF_ENDPOINT=https://hf-mirror.com`）。
- `opencv-python-headless`、`scipy`、`scikit-learn`、`numpy<2`（注意与 opencv 的版本冲突）。

### 7.3 核心命令（服务器）
```bash
# PUMA（已完成）
python train_puma.py --folds 0 1 2 3 4 --model foundation_uni \
  --batch_size 4 --target_size 512 --epochs 100 --lr 1e-4 \
  --patience 50 --warmup_epochs 10 --label_smoothing 0.1

# GLaS Step 1（进行中）：CNN DCAN 转向，对标冠军 DCAN 0.897
python train_glas.py --data_dir .../data/glas --folds 0 1 2 3 4 \
  --model cnndcan --batch_size 8 --target_size 512 --epochs 100 \
  --output_dir .../output_glas_cnndcan
# 训练后推理 + 阈值扫描：
python tune_glas.py --data_dir .../data/glas --weights_dir .../output_glas_cnndcan \
  --output_dir .../output_glas_cnndcan/preds_tuned --folds 5 --split testB \
  --method watershed --model cnndcan
```

---

## 8. 待办 / 下一步

> 治理轨道（autoseg-agent）自身的待办见 **§8.2**；nnU-Net 对照轨道见 **§6.7**。

### 8.1 LLM / vibe coding 轨道

- [x] **nnU-Net v2 对照基线（第 6 节）**：DRIVE 与 GLaS 训练/推理/官方口径评估全部完成并归档；GLaS 官方对象级 testA 0.8723/0.7739/67.31、testB 0.8184/0.6113/118.85，落在已发表方法正常区间。遗留：LLM 轨道 GLaS 预测已从服务器清理，跨轨道同口径对比需重跑推理。
- [x] GLaS v2 训练完成：5-fold val 0.7873 / testB objDice 0.6867，全面低于 v1，回归弃用。
- [x] Step 0 完成：分水岭后处理 testB objDice 0.7316→**0.7513**、testA 0.8198，超越轮廓法，暂定为当前最优后处理。
- [x] Step 1：GLaS 转向 CNN DCAN 三组消融（深监督开/关、encoder_lr 0.1/1.0）全部 Fold0 val ≤0.7049，低于 UNI v1 0.8075 → CNN 路线判定失败。
- [x] 核对 GLaS 官方评估协议：对象级 Object F1 / Object Dice / Object Hausdorff 三指标；`eval_glas.py` 补齐 Object Hausdorff。
- [x] 修复 resize 硬伤：`dataset_glas.py`/`inference_glas.py`/`tune_glas.py` 改为 letterbox 保持纵横比 + unletterbox 回裁（已过语法 + 冒烟测试）。
- [x] letterbox 重训 UNI 版（GlandUNet）完成：5-fold val objDice mean 0.7833（< 旧 v1 0.8075，属挤水分，见 3.3）。
- [x] letterbox 重训的 testB/testA 已评：testB 0.7258（旧 0.7513）、testA 0.7939（旧 0.8198），全面微降 ~0.026 → letterbox 无收益，几何失真非瓶颈。
- [ ] 下一步：HoVer 距离图头 5 折已训完（val mean 0.7575 < letterbox 0.7833，约 1σ 回落）；**立即跑 testB/testA 跨集评估定论** → 若 testB 提升则继续调 hover 阈值；否则转 RIC-UNet 残差+Inception（冲 0.91）。并做多 run 配对对比排除单次训练噪声。
- [ ] Step 2（Step 1 后，冲 0.91 可选）：RIC-UNet 残差+Inception 或 HoVer-Net 式距离图 head。
- [ ] BEETLE 数据准备 + 基线训练。
- [ ] 增加非病理非血管任务（如 SIIM）补全框架通用性验证。
- [ ] 汇总做统计显著性检验（UNI vs ImageNet，配对 t 检验）。
- [ ] 论文：以"标准化 prompt 框架 + 复现清单"为主线，GLaS/PUMA/DRIVE 为基准结果，失败模式分析做 Discussion。

---

### 8.2 治理轨道（autoseg-agent）

- [x] **Step 0 路线池核实**：三条"名头很响"的路线（MILD-Net / DCAN / RIC-Unet）均无可用官方代码，一律 `low` 不作锚点；锁定 **R05 U-Node** 为复现锚点（§6.8）。
- [x] **判定字段变更 v1.3→v1.4**：GLaS L2 由 MILD-Net 0.836 上移到 U-Node 0.842；移除不适用的 R03_RICUnet（§6.9）。
- [x] **治理收口 v1.4→v1.5**：新增红线 **R10 锚点隔离**、重放**钉死 episode 自带规则版本**、`judge --log` 留痕**强制带 state_snapshot**、GLaS **起跑态** + `EP02_GLaS_real` 起跑包（§6.10）。
- [x] **Step 1（阶段 1）：数据 / 评估脚本就位 + 复现 U-Node**（§6.11）：本地 `eval_glas_official.py` 自检 testB 0.8184；服务器跑通 U-Node 官方权重推理，官方口径复现 testB ObjDice **0.8428**（论文 0.842）→ 锚点成立，写入 `state.anchor`（**不进 best**，红线 R10）；并顺带量化出 ≈0.02 的「实例标签 vs 二值重标注」口径差。
- [ ] **⛔ 待人工裁定（阻塞 Step 2）**：GLaS 评估口径二选一 —— 见 `state.pending_human_decisions[GLaS_EVAL_CONVENTION]` / `AUTONOMY_POLICY` R7。
  - **A**：`eval_glas_official.py` 支持**实例标签**输入（8 连通 + 众数匹配 + 面积加权），L2 维持 0.842；优点是口径与 U-Node / 官方提交形态一致，缺点是脚本要改且后续路线都得输出实例标签。
  - **B**：维持**二值 + 池化**口径，把 L2 线按同口径重算（U-Node 同口径 0.8215、nnU-Net 0.8184）；优点是脚本不动、口径简单，缺点是 L2 线要下调、与文献报告值不可直接比。
  - 备注：审计显示两套协议在**同一预测形态**下仅差 <0.002，真正的分水岭是"喂实例标签还是二值"，故裁定本质是"我们对外报告用哪种预测形态"。
- [ ] **⛔ 待人工裁定（不阻塞 Step 2，阻塞验收 8/8）**：红线 R1 是否给「校准类数字」加豁免 —— 见 `state.pending_human_decisions[R1_CALIBRATION_EXEMPTION]`。
  - 现状：`redlines.py --audit` 15 项 2 违规、`acceptance.py` 7/8，原因是锚点校准数字与官方口径数字同 metric+subset 被判"跨口径并列"。
  - 推荐 A：R1 比较组排除 `role` 以 `anchor_calibration` 开头的数字（它们为记录口径差异而存在，且 R10 已保证进不了 best）。
- [ ] **Step 2（口径裁定后，先不训练）**：用 GLaS 真跑一遍完整循环 —— 复现 R05 U-Node（已完成）→ 在锚点上改进（优先 R04 后处理，零成本）→ `judge.py --log` 判定 → 按决策换路线。**目标：把"一轮 prompt 跑完"的标准化流程跑通并固化。**
- [ ] 补 PUMA / DRIVE 官方评估脚本并进 `eval.allowlist`，把二者转为可判定的 `active`。
- [ ] 扩展到第 2 个数据集（DRIVE 或 PUMA），验证 `datasets.yaml` 的"加一行即可扩展"确实成立。