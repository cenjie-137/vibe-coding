# GLaS 一看就会：零代码上手手册

> 目标：不懂代码也能在 AutoDL 服务器跑通 GLaS 腺体分割，拿到对象级指标。

---

## 一、数据集在哪，怎么解压

你上传的 zip 在：

```
/root/autodl-tmp/medseg_project/data/GIaS/archive (1).zip
```

解压命令（SSH 终端里执行）：

```bash
cd /root/autodl-tmp/medseg_project/data/GIaS
unzip -q "archive (1).zip" -d glas_raw
```

解压后会出现 `glas_raw/Warwick_QU_Dataset/`（里面有 `train_*.bmp`、`testA_*.bmp`、`testB_*.bmp` 和 `Grade.csv`）。

> 提示：`convert_glas.py` 支持直接吃 zip，所以**解压其实可以跳过**，一键脚本会自动处理。

---

## 二、要上传哪些文件到 code

把下面这些文件都上传到同一个目录 `/root/autodl-tmp/medseg_project/code/`：

| 文件 | 作用 | 是否必须 |
|------|------|----------|
| `convert_glas.py` | 原始数据 → 统一格式 | ✅ 必须 |
| `eval_glas.py` | 对象级 Dice/F1 评估 | ✅ 必须 |
| `models_glas.py` | GlandUNet 模型（UNI 编码器 + 双头） | ✅ 必须 |
| `dataset_glas.py` | 数据加载 + 轮廓 GT + H&E 增强 | ✅ 必须 |
| `train_glas.py` | 5-fold 训练 | ✅ 必须 |
| `inference_glas.py` | 5-fold 集成推理 + 后处理 | ✅ 必须 |
| `models.py` | 依赖：UNI 编码器 `FoundationViTEncoder` | ✅ 必须 |
| `datasets_puma.py` | 依赖：`apply_stain_augmentation` 染色增强 | ✅ 必须 |
| `run_glas.sh` | 一键流水线（可选，推荐） | 推荐 |
| `README_GLAS.md` | 本说明 | 可选 |

> 重要：`models.py` 和 `datasets_puma.py` 是**依赖文件**，缺了会报 `ModuleNotFoundError`。如果你服务器 `code/` 里已经有（之前 PUMA 用过），用本地这份**最新版覆盖**，保证版本一致。

---

## 三、环境准备（只需一次）

```bash
cd /root/autodl-tmp/medseg_project/code
pip install -U timm huggingface_hub
```

验证：

```bash
python -c "import torch, timm, huggingface_hub; print(torch.__version__, timm.__version__, torch.cuda.is_available())"
```

`cuda ready: True` 就绪。UNI 权重会走 `hf-mirror.com` 镜像自动下载（或直接用之前 PUMA 缓存的）。

---

## 四、正式跑（两种方式，二选一）

### 方式 A：一键脚本（推荐，最省事）

```bash
cd /root/autodl-tmp/medseg_project/code
bash run_glas.sh
```

脚本会依次完成：环境检查 → 数据转换 → Smoke test → 全量 5-fold 训练 → 推理 → 评估。

### 方式 B：手动一步步跑（便于排查）

```bash
cd /root/autodl-tmp/medseg_project/code

# 1. 数据转换
python convert_glas.py \
  --src "/root/autodl-tmp/medseg_project/data/GIaS/archive (1).zip" \
  --dst /root/autodl-tmp/medseg_project/data/glas

# 2. 训练（5-fold，每 fold 100 epochs，batch 4，32GB 显存可跑）
python train_glas.py \
  --data_dir /root/autodl-tmp/medseg_project/data/glas \
  --folds 0 1 2 3 4 --epochs 100 --batch_size 4 \
  --output_dir /root/autodl-tmp/medseg_project/code/output_glas

# 3. 推理（testA + testB，5-fold 集成 + 4 方向 TTA）
python inference_glas.py \
  --data_dir /root/autodl-tmp/medseg_project/data/glas \
  --weights_dir /root/autodl-tmp/medseg_project/code/output_glas \
  --output_dir /root/autodl-tmp/medseg_project/code/output_glas/preds --folds 5

# 4. 评估（官方 Object Dice / F1）
python eval_glas.py \
  --gt_dir /root/autodl-tmp/medseg_project/data/glas/masks \
  --pred_dir /root/autodl-tmp/medseg_project/code/output_glas/preds \
  --metadata /root/autodl-tmp/medseg_project/data/glas/metadata.csv --split testB
```

---

## 五、方案说明（为什么这么选）

- **编码器**：病理基础模型 **UNI（ViT-L/16）**，病理预训练，比 ImageNet 初始化更强。
- **网络**：**DCAN 式双头**——分割头（腺体前景）+ 轮廓头（边界），用边界把粘连腺体拆开，这是对象级评估拿高分的关键。
- **训练**：5-fold 分层交叉验证（按良性/恶性分层）；差分学习率（编码器 = 解码器 × 0.1），保护预训练特征；warmup + cosine 调度；AMP 混合精度。
- **增强**：翻转/旋转 + H&E 染色扰动（HED 颜色空间）。
- **推理**：5 个模型概率平均 + 4 方向 TTA + 轮廓拆分 + 去小碎片。
- **显存**：512×512 输入、batch 4、AMP，32GB 显存绰绰有余。

---

## 六、常见问题

| 现象 | 原因 | 解决 |
|------|------|------|
| `ModuleNotFoundError: models / datasets_puma` | 依赖文件没传或不在同一目录 | 确认 8 个文件都在 `code/` 同一级 |
| `Failed to load UNI` 且回退 ImageNet | 权重没下好 | 检查网络，重跑 `pip install -U timm huggingface_hub` |
| 训练 Loss NaN | 学习率过高 | 默认 lr=1e-4 通常稳定，无需改 |
| 显存不足 OOM | batch 太大 | 把 `--batch_size 4` 改成 `2` |