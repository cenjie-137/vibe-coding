# -*- coding: utf-8 -*-
"""登记两个待人工裁定项 + R1 违规发现（阶段 1 收口第二步）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "autorun"))
from ledger import append_run_log  # noqa: E402

append_run_log(dict(
    run_id="step1-pending-20261008",
    task="GLaS",
    phase="governance",
    action=(
        "阶段 1 收口②：把两项需人工裁定的事登记进 state.pending_human_decisions —— "
        "① GLaS_EVAL_CONVENTION（评估口径 A/B，阻塞 Step 2）；"
        "② R1_CALIBRATION_EXEMPTION（红线 R1 报 2 项违规：锚点校准数字与官方口径数字同 metric+subset "
        "被判跨口径并列；因 redlines.py 属 R7 保护文件，Agent 不擅自修改，登记待裁定）。"
        "现状：acceptance 7/8，其余 7 项（judge/ledger/redlines/replay 自检、策略一致性、证据包重放、记分卡）全过。"
    ),
    actor="agent",
    artifact="autorun/state.json, RESEARCH_LOG.md, ROUTE_POOL.yaml",
    destructive=False,
))
print("OK: pending decisions logged")
