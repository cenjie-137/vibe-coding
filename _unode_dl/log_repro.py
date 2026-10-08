# -*- coding: utf-8 -*-
"""把阶段1（U-Node 复现）的留痕写进溯源账本：run_log.jsonl + claims.jsonl。"""
import sys
from pathlib import Path

AUTORUN = Path(r"c:\Users\32201\Desktop\T\vibe coding\DRIVE\autorun")
sys.path.insert(0, str(AUTORUN))
from ledger import append_claim, append_run_log  # noqa: E402

LOG_REF = "RESEARCH_LOG.md#glas-unode-repro"

run = append_run_log(dict(
    run_id="reproduce-20261008_unode",
    task="GLaS",
    phase="reproduce",
    action=(
        "复现 R05 U-Node：取官方仓库 DIAGNijmegen/neural-odes-segmentation 与官方预训练权重 "
        "best_border_unode_paper.pt（state_dict 零缺失加载），在 testA(60)/testB(20) 上跑官方 "
        "train_models.ipynb 的推理与后处理（inference_utils.inference_image 含三向 TTA + "
        "postprocess 阈值0.7/去小目标500/生长/填洞），用官方 metrics.py 评估。"
        "testB ObjDice=0.8428（论文报告 0.842，吻合）。"
        "同一批预测改二值后用本仓库 eval_glas_official.py 评估得 testB=0.8215 —— 暴露两套口径差约 0.02。"
    ),
    actor="agent",
    artifact=("glas_unode_pred/, glas_unode_pred_label/, _unode_dl/unode_metrics.json, "
              "audit_glas_metric_protocol.py"),
    destructive=False,
))

claims = [
    dict(task="GLaS", metric="testB_objDice", subset="testB",
         convention="unode_official_object_level", value=0.8428,
         source_url=LOG_REF, source_type="reproduced_external", verified=True,
         role="anchor_calibration",
         note="U-Node 官方口径复现（实例标签+8连通+众数匹配+面积加权逐图平均）；论文报告 0.842。仅作口径校验，按 R10 不计入 best。"),
    dict(task="GLaS", metric="testA_objDice", subset="testA",
         convention="unode_official_object_level", value=0.8872,
         source_url=LOG_REF, source_type="reproduced_external", verified=True,
         role="anchor_calibration",
         note="同上，testA 复现（论文报告 0.893）。"),
    dict(task="GLaS", metric="testB_objDice", subset="testB",
         convention="official_object_level", value=0.8215,
         source_url=LOG_REF, source_type="reproduced_external", verified=True,
         role="anchor_calibration_convention_mapping",
         note="U-Node 同一批预测改为二值输出、用本仓库 eval_glas_official.py（4连通+argmax+数据集级池化）评估。用于量化两套口径的差距（≈0.02）。"),
    dict(task="GLaS", metric="testA_objDice", subset="testA",
         convention="official_object_level", value=0.8200,
         source_url=LOG_REF, source_type="reproduced_external", verified=True,
         role="anchor_calibration_convention_mapping",
         note="同上，testA 口径映射值。"),
]
for c in claims:
    append_claim(c)

print("run_log appended:", run["run_id"])
print("claims appended:", len(claims))
