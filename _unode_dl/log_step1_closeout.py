# -*- coding: utf-8 -*-
"""阶段 1 收口留痕：补溯源章节 + 更新路线池复现状态。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "autorun"))
from ledger import append_run_log  # noqa: E402

append_run_log(dict(
    run_id="step1-closeout-20261008",
    task="GLaS",
    phase="governance",
    action=(
        "阶段 1 收口：① RESEARCH_LOG.md 新增 §6.11「U-Node 复现 + 口径校验锚点」"
        "（含 <a id='glas-unode-repro'>，使 claims 的 source_url 可解析）；"
        "② 重跑 audit_glas_metric_protocol.py 得到 16 组协议对照，确认 ≈0.02 口径差"
        "根因是「实例标签 vs 二值重标注」而非连通性/匹配/聚合（同形态下各协议差 <0.002）；"
        "③ ROUTE_POOL.yaml 的 R05_UNODE 置 reproduced=true / status=anchor 并回填复现值；"
        "④ §8.2 登记待人工裁定的评估口径项（阻塞 Step 2）。"
    ),
    actor="agent",
    artifact="RESEARCH_LOG.md, ROUTE_POOL.yaml, audit_glas_metric_protocol.py",
    destructive=False,
))
print("OK: step1 closeout logged")
