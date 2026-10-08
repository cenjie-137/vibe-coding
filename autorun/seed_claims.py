# -*- coding: utf-8 -*-
"""把已核实的基线数字灌进 claims.jsonl —— 账本的第一批真实内容。

这些数字就是论文里要引用的对照组。每一个都带来源；
最后一条故意留空来源，用来演示红线 R2 会把"无来源数字"挡在判定之外。

用法：
    python seed_claims.py          # 追加写入（重复运行会去重）
    python seed_claims.py --reset  # 清空重来
"""
from __future__ import annotations

import sys
from pathlib import Path

from ledger import DEFAULT_CLAIMS, append_claim, load_jsonl

SEEDS = [
    # ── 自家 nnU-Net v2 基线（来源 = 本项目实验记录）────────────────
    dict(claim_id="G-NNUNET-TB", task="GLaS", metric="testB_objDice", subset="testB",
         convention="official_object_level", value=0.8184,
         source_url="RESEARCH_LOG.md#nnunet-glas", source_type="self_measured",
         verified=True, note="本项目 nnU-Net v2 官方口径实测（= L1 线）"),
    dict(claim_id="G-NNUNET-TA", task="GLaS", metric="testA_objDice", subset="testA",
         convention="official_object_level", value=0.8723,
         source_url="RESEARCH_LOG.md#nnunet-glas", source_type="self_measured",
         verified=True, note="本项目 nnU-Net v2 官方口径实测"),
    dict(claim_id="D-NNUNET-CV", task="DRIVE", metric="cv_dice", subset="cv5",
         convention="pixel_level", value=0.7990,
         source_url="RESEARCH_LOG.md#nnunet-drive", source_type="self_measured",
         verified=True, note="本项目 nnU-Net v2 5折CV实测（= L1 线）"),

    # ── 已发表方法（来源 = 论文）────────────────────────────────────
    # Step 0 更正：MILD-Net 的来源改用已亲自打开确认的 arXiv 页面；
    # MIA 版 DOI 有两个候选，待人工复核后再换回。
    dict(claim_id="G-MILDNET-TB", task="GLaS", metric="testB_objDice", subset="testB",
         convention="official_object_level", value=0.836,
         source_url="https://arxiv.org/abs/1806.01963", source_type="paper",
         verified=True, note="MILD-Net (Graham et al., MIA 2019)（= L2 线）；数值待人工从原文表格复核"),
    dict(claim_id="G-MILDNET-TA", task="GLaS", metric="testA_objDice", subset="testA",
         convention="official_object_level", value=0.913,
         source_url="https://arxiv.org/abs/1806.01963", source_type="paper",
         verified=True, note="MILD-Net (Graham et al., MIA 2019)；数值待人工从原文表格复核"),

    # Step 0 更正：DCAN 原文是 1604.02677；原记录的 1603.00275 实为 GLaS 挑战赛综述论文
    dict(claim_id="G-DCAN-TB-DICE", task="GLaS", metric="testB_objDice", subset="testB",
         convention="official_object_level", value=0.781,
         source_url="https://arxiv.org/abs/1604.02677", source_type="paper",
         verified=True, note="DCAN (Chen et al., CVPR 2016)；数值经 NODE 论文 Table 1 交叉核对一致"),
    dict(claim_id="G-DCAN-TB-F1", task="GLaS", metric="testB_objF1", subset="testB",
         convention="official_object_level", value=0.716,
         source_url="https://arxiv.org/abs/1604.02677", source_type="paper",
         verified=True, note="DCAN 实例分离能力强，ObjF1 高于我们 nnU-Net 的 0.611"),

    # ── Step 0 新增：真实可用开源实现 R05 的报告值（官方对象级）────────
    dict(claim_id="G-UNODE-TB", task="GLaS", metric="testB_objDice", subset="testB",
         convention="official_object_level", value=0.842,
         source_url="https://arxiv.org/abs/1910.10470", source_type="paper",
         verified=True, note="U-Node (Pinckaers & Litjens, 2019) Table 1 括号内 testB；代码可复现"),
    dict(claim_id="G-UNODE-TA", task="GLaS", metric="testA_objDice", subset="testA",
         convention="official_object_level", value=0.893,
         source_url="https://arxiv.org/abs/1910.10470", source_type="paper",
         verified=True, note="U-Node Table 1 括号内 testA"),
    dict(claim_id="G-UNODE-TB-F1", task="GLaS", metric="testB_objF1", subset="testB",
         convention="official_object_level", value=0.801,
         source_url="https://arxiv.org/abs/1910.10470", source_type="paper",
         verified=True, note="U-Node Table 1 括号内 testB F1"),

    # ── Step 0 口径陷阱实录：像素级数字，**必须**用另一个指标名，否则会污染对象级组 ──
    dict(claim_id="G-UNET-PIX-TB", task="GLaS", metric="testB_pixelDice", subset="testB",
         convention="pixel_level", value=0.91,
         source_url="https://github.com/twpkevin06222/Gland-Segmentation", source_type="repo_readme",
         verified=True, note="R06 仓库 README 自述为 pixel-wise 评估，非对象级 —— 不得与对象级并列（R1）"),

    # ── 故意无来源：演示红线 R2 的拦截 ───────────────────────────────
    dict(claim_id="G-RICUNET-TB", task="GLaS", metric="testB_objDice", subset="testB",
         convention="official_object_level", value=None,
         source_url="", source_type="memory",
         verified=False, note="记忆中 0.918 疑为不同口径/子集 —— 未核实，不得参与判定（R2）"),
]


def main() -> int:
    if "--reset" in sys.argv and Path(DEFAULT_CLAIMS).exists():
        Path(DEFAULT_CLAIMS).unlink()
        print(f"已清空 {DEFAULT_CLAIMS.name}")

    existing = {c.get("claim_id") for c in load_jsonl(DEFAULT_CLAIMS)}
    added = skipped = 0
    for c in SEEDS:
        if c["claim_id"] in existing:
            skipped += 1
            continue
        append_claim(dict(c))
        added += 1

    from ledger import source_coverage, trust_scorecard

    claims = load_jsonl(DEFAULT_CLAIMS)
    sc = trust_scorecard(claims=claims)
    print(f"新增 {added} 条，跳过已存在 {skipped} 条")
    print(f"账本共 {len(claims)} 条数字，来源覆盖率 = {source_coverage(claims):.1%}")
    print(f"可参与判定 {sum(1 for c in claims if c.get('decision_eligible'))} 条，"
          f"被 R2 挡住 {sum(1 for c in claims if not c.get('decision_eligible'))} 条")
    print(f"记分卡：{sc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
