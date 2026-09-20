#!/usr/bin/env bash
# ============================================================
# GLaS 一键流水线（零代码：上传代码文件后，执行本脚本即可）
# 用法：bash run_glas.sh
# ============================================================
set -e

PROJ=/root/autodl-tmp/medseg_project
CODE=${PROJ}/code
DATA=${PROJ}/data
ZIP="${DATA}/GIaS/archive (1).zip"     # 你上传的 zip 路径
GLAS_DIR=${DATA}/glas                  # 统一格式输出目录
OUT=${CODE}/output_glas_v2             # 5-fold 模型权重 + 预测结果（v2 重训：768 + 更深解码头）

echo "============================================================"
echo "[0/5] 环境检查"
echo "============================================================"
python -c "import torch, timm, huggingface_hub; print('torch', torch.__version__, '| timm', timm.__version__, '| cuda ready:', torch.cuda.is_available())"

echo ""
echo "============================================================"
echo "[1/5] 数据转换 (165 张 -> images/ + masks/ + metadata.csv)"
echo "============================================================"
python ${CODE}/convert_glas.py --src "${ZIP}" --dst ${GLAS_DIR}

echo ""
echo "============================================================"
echo "[2/5] Smoke test (fold 0, 1 epoch, batch 2) —— 验证 UNI 能加载、能跑通"
echo "============================================================"
python ${CODE}/train_glas.py --data_dir ${GLAS_DIR} --folds 0 --epochs 1 \
    --batch_size 2 --output_dir ${OUT}_smoke

echo ""
echo "============================================================"
echo "[3/5] 全量 5-fold 训练 (每 fold 100 epochs, batch 2, 768px)"
echo "============================================================"
python ${CODE}/train_glas.py --data_dir ${GLAS_DIR} --folds 0 1 2 3 4 \
    --epochs 100 --batch_size 2 --target_size 768 --output_dir ${OUT}

echo ""
echo "============================================================"
echo "[4/5] 推理 (testA 60 + testB 20, 5-fold 集成 + TTA + 轮廓拆分)"
echo "============================================================"
python ${CODE}/inference_glas.py --data_dir ${GLAS_DIR} \
    --weights_dir ${OUT} --output_dir ${OUT}/preds --folds 5 --target_size 768

echo ""
echo "============================================================"
echo "[5/5] 对象级评估 (官方 Object Dice / F1)"
echo "============================================================"
echo "--- testA ---"
python ${CODE}/eval_glas.py --gt_dir ${GLAS_DIR}/masks \
    --pred_dir ${OUT}/preds --metadata ${GLAS_DIR}/metadata.csv --split testA
echo "--- testB (官方榜单用) ---"
python ${CODE}/eval_glas.py --gt_dir ${GLAS_DIR}/masks \
    --pred_dir ${OUT}/preds --metadata ${GLAS_DIR}/metadata.csv --split testB

echo ""
echo "============================================================"
echo "全部完成！结果在 ${OUT}/preds，评估汇总见上方输出。"
echo "============================================================"