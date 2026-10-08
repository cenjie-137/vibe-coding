# -*- coding: utf-8 -*-
"""红线 R1–R9 的可执行断言。

AUTONOMY_POLICY.md 第 5 节把红线写成"原则"，靠人自觉遵守。
本文件把它变成**能跑的检查**：Agent 每做完一件事，都可以被这套断言当场审问，
过不了就说明违规。这是"可信自主性"从口号变成机制的关键一步。

九条红线 → 九个纯函数（输入是产物，输出是判定）：

  R1 口径锁定     check_r1(records)                 参与比较的数字必须同 指标+子集+口径
  R2 来源强制     check_r2(claim)                   无来源的数字不得参与判定
  R3 阈值单源     check_r3(policy, judge_src)       停止阈值只能来自 policy.json，不许硬编码
  R4 复现准入     check_r4(route)                   有代码必须先复现成功；无代码须标 low
  R5 评估锁定     check_r5(eval_record, allowlist)  只用官方协议脚本评估
  R6 破坏性操作   check_r6(action)                  删除/覆盖前必须备份
  R7 自我修改禁止 check_r7(changed_files, protected) 不得改动受保护文件
  R8 留痕强制     check_r8(run_log_lines)           每个动作都要合格留痕
  R9 资源边界     check_r9(state, policy)           路线数 / 连续失败不得越界

用法：
    python redlines.py --selftest     # 跑 9 条红线的对抗用例（每条一个应通过、一个应拦下）
    python redlines.py --audit        # 审计当前工作目录的产物，打印一份合规报告
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ledger import is_decision_eligible, validate_run_log

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

NO_CODE_MARKERS = ("", "<无>", "—", "-", "none", "n/a", "null")
# 占位符匹配要容大小写："<待 Step 0 核实>" 和 "<待 step 0 核实>" 都是"没有代码"
NO_CODE_HINTS = ("待核实", "待 step 0 核实", "todo", "tbd", "unknown")


def _is_placeholder(code) -> bool:
    c = str(code or "").strip().lower()
    return c in NO_CODE_MARKERS or any(h in c for h in NO_CODE_HINTS)


@dataclass
class Result:
    code: str
    ok: bool
    msg: str
    evidence: list = field(default_factory=list)

    def __str__(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        return f"[{mark}] {self.code} {self.msg}"


# ── R1 口径锁定 ───────────────────────────────────────────────────
def check_r1(records: list[dict]) -> Result:
    """参与比较的所有数字，必须同 metric + 同 subset + 同 convention。"""
    if len(records) < 2:
        return Result("R1", True, "少于 2 条，无需比较")
    keys = {(r.get("metric"), r.get("subset"), r.get("convention")) for r in records}
    if len(keys) == 1:
        return Result("R1", True, f"口径一致：{next(iter(keys))}")
    return Result("R1", False, "跨口径数字被并列 —— 禁止用于判定", sorted(map(str, keys)))


# ── R2 来源强制 ───────────────────────────────────────────────────
def check_r2(claim: dict) -> Result:
    if is_decision_eligible(claim):
        return Result("R2", True, f"来源齐备：{claim.get('source_url')}")
    if claim.get("decision_eligible"):
        return Result("R2", False, "无有效来源却标记可参与判定", [claim.get("claim_id")])
    return Result("R2", True, "无来源，已标 UNVERIFIED 且不参与判定")


# ── R3 停止阈值单一数据源 ─────────────────────────────────────────
def check_r3(policy: dict, judge_src: str) -> Result:
    thr = policy.get("thresholds", {})
    missing = [k for k in ("small", "big") if k not in thr]
    if missing:
        return Result("R3", False, f"policy 缺阈值定义：{missing}")
    # 阈值一旦被硬编码进代码，改 policy 就失效 —— 这是"规则漂移"的入口
    hard = [n for n in ("0.003", "0.005") if n in judge_src]
    if hard:
        return Result("R3", False, f"judge.py 硬编码了阈值 {hard}，必须只读 policy.json", hard)
    return Result("R3", True, f"阈值单一数据源 OK：small={thr['small']}, big={thr['big']}")


# ── R4 复现准入 ───────────────────────────────────────────────────
def check_r4(route: dict) -> Result:
    has_code = not _is_placeholder(route.get("code"))
    code_verified = bool(route.get("code_verified"))
    conf, status = route.get("confidence"), route.get("status")

    if not has_code:
        if conf != "low":
            return Result("R4", False, f"无开源代码的路线必须标 confidence=low，实际 {conf!r}")
        if status == "anchor":
            return Result("R4", False, "无代码路线不得升级为锚点")
        return Result("R4", True, "无代码路线已标 low 且非锚点")

    # 方法"存在开源代码"≠"我们手上的链接已核实"。没核实就不许声称高置信
    if not code_verified and conf == "high":
        return Result("R4", False, "代码链接尚未核实（code_verified=false）却标 confidence=high —— 先跑 Step 0")

    if status in ("improving", "anchor") and not route.get("reproduced"):
        return Result("R4", False, f"未复现成功却进入 {status} 阶段")
    return Result("R4", True, "复现准入 OK")


# ── R5 评估锁定 ───────────────────────────────────────────────────
def check_r5(eval_record: dict, allowlist: list[str]) -> Result:
    script = str(eval_record.get("script") or "")
    if script in allowlist:
        return Result("R5", True, f"评估脚本在官方白名单：{script}")
    return Result("R5", False, f"评估脚本 {script!r} 不在官方白名单 {allowlist}")


# ── R6 破坏性操作须备份 ───────────────────────────────────────────
def check_r6(action: dict) -> Result:
    if not action.get("destructive"):
        return Result("R6", True, "非破坏性操作")
    if not str(action.get("backup_path") or "").strip():
        return Result("R6", False, "破坏性操作未提供 backup_path")
    if action.get("logged") is False:
        return Result("R6", False, "破坏性操作未留痕")
    return Result("R6", True, f"已备份：{action.get('backup_path')}")


# ── R7 自我修改禁止 ───────────────────────────────────────────────
def _norm(p: str) -> str:
    return str(p).replace("\\", "/").lstrip("./")


def check_r7(changed_files: list[str], protected: list[str]) -> Result:
    prot = [_norm(p) for p in protected]
    bad = [f for f in changed_files if any(_norm(f).endswith(p) for p in prot)]
    if bad:
        return Result("R7", False, "Agent 改动了受保护文件", bad)
    return Result("R7", True, f"未触碰 {len(prot)} 个受保护文件")


# ── R8 留痕强制 ───────────────────────────────────────────────────
def check_r8(run_log_lines: list[dict]) -> Result:
    bad = []
    for i, rec in enumerate(run_log_lines, 1):
        errs = validate_run_log(rec)
        if errs:
            bad.append(f"line{i}: {'; '.join(errs)}")
    if bad:
        return Result("R8", False, f"{len(bad)} 条留痕不合格", bad)
    return Result("R8", True, f"{len(run_log_lines)} 条留痕齐备")


# ── R9 资源边界 ───────────────────────────────────────────────────
def check_r9(state: dict, policy: dict) -> Result:
    lim = policy.get("limits", {})
    max_routes = lim.get("max_routes")
    cap = lim.get("max_consecutive_route_failures")
    idx = state.get("route_index", 0)
    fails = len(state.get("routes_failed", []))
    if max_routes is not None and idx > max_routes:
        return Result("R9", False, f"已试路线 {idx} 条，超过上限 {max_routes}")
    if cap is not None and fails > cap:
        return Result("R9", False, f"连续失败 {fails} 条，超过上限 {cap}（本应已硬停）")
    return Result("R9", True, f"在资源边界内（路线 {idx}/{max_routes}，连续失败 {fails}/{cap}）")


REDLINES = {
    "R1": ("口径锁定", check_r1),
    "R2": ("来源强制", check_r2),
    "R3": ("阈值单一数据源", check_r3),
    "R4": ("复现准入", check_r4),
    "R5": ("评估锁定", check_r5),
    "R6": ("破坏性操作须备份", check_r6),
    "R7": ("自我修改禁止", check_r7),
    "R8": ("留痕强制", check_r8),
    "R9": ("资源边界", check_r9),
}


# ── 审计当前工作目录 ──────────────────────────────────────────────
def audit() -> int:
    from ledger import DEFAULT_CLAIMS, DEFAULT_RUN_LOG, load_jsonl

    policy = json.loads((HERE / "policy.json").read_text(encoding="utf-8"))
    judge_src = (HERE / "judge.py").read_text(encoding="utf-8")
    protected = policy.get("protected_files", [])
    allowlist = policy.get("eval", {}).get("allowlist", [])

    results = [
        check_r3(policy, judge_src),
        check_r8(load_jsonl(DEFAULT_RUN_LOG)),
    ]

    # R1：同名指标（同任务 + 同指标 + 同子集）必须同口径 —— 否则"看似可比、实则不可比"
    claims = load_jsonl(DEFAULT_CLAIMS)
    by_key: dict[tuple, list] = {}
    for c in claims:
        by_key.setdefault((c.get("task"), c.get("metric"), c.get("subset")), []).append(c)
    for key, group in sorted(by_key.items(), key=lambda kv: str(kv[0])):
        results.append(check_r1(group))

    # R4：审计路线池
    pool = ROOT / "ROUTE_POOL.yaml"
    if pool.exists():
        try:
            import yaml  # type: ignore

            for r in (yaml.safe_load(pool.read_text(encoding="utf-8")) or {}).get("routes", []):
                results.append(check_r4(r))
        except ImportError:
            print("(跳过 R4：未安装 PyYAML)")

    print(f"── 红线审计：{ROOT.name} ──")
    for r in results:
        print(" ", r)
        for e in r.evidence[:5]:
            print(f"       · {e}")
    failed = [r for r in results if not r.ok]
    print(f"\n共 {len(results)} 项检查，{len(failed)} 项违规")
    return 1 if failed else 0


# ── 自检 ──────────────────────────────────────────────────────────
def selftest() -> int:
    policy = json.loads((HERE / "policy.json").read_text(encoding="utf-8"))
    judge_src = (HERE / "judge.py").read_text(encoding="utf-8")

    base_cmp = dict(metric="testB_objDice", subset="testB", convention="official_object_level")
    good_claim = dict(
        claim_id="C1", task="GLaS", metric="testB_objDice", subset="testB",
        convention="official_object_level", value=0.836,
        source_url="https://doi.org/x", verified=True, decision_eligible=True,
    )
    no_src = {**good_claim, "source_url": "", "decision_eligible": True}
    good_route = dict(id="R01", code="https://github.com/x", code_verified=True,
                      confidence="high", status="improving", reproduced=True)
    bad_route = dict(id="R02", code="<无>", confidence="high", status="pending")
    unverified_route = dict(id="R05", code="https://github.com/y", code_verified=False,
                            confidence="high", status="pending")
    anchor_unreproduced = dict(id="R06", code="https://github.com/z", code_verified=True,
                               confidence="high", status="anchor", reproduced=False)
    good_eval = dict(script="eval_glas_official.py")
    bad_eval = dict(script="eval_quick_hack.py")
    good_act = dict(destructive=True, backup_path="bak/2026-10-08/", logged=True)
    bad_act = dict(destructive=True, logged=True)

    cases = [
        ("R1 同口径通过", check_r1([base_cmp, dict(base_cmp)]), True),
        ("R1 跨口径拦下", check_r1([base_cmp, {**base_cmp, "convention": "pixel_level"}]), False),
        ("R2 有来源通过", check_r2(good_claim), True),
        ("R2 无来源却可判定 → 拦下", check_r2(no_src), False),
        ("R3 阈值单源通过", check_r3(policy, judge_src), True),
        ("R3 代码硬编码阈值 → 拦下", check_r3(policy, "x = 0.003  # 硬编码"), False),
        ("R4 复现后可改进", check_r4(good_route), True),
        ("R4 无代码却标 high → 拦下", check_r4(bad_route), False),
        ("R4 占位符代码却标 high → 拦下",
         check_r4(dict(id="R03", code="<待 Step 0 核实>", confidence="high", status="pending")), False),
        ("R4 占位符大小写不一致也拦下",
         check_r4(dict(id="R04", code="<待 step 0 核实>", confidence="high", status="pending")), False),
        ("R4 代码链接未核实却标 high → 拦下", check_r4(unverified_route), False),
        ("R4 未复现却标锚点 → 拦下", check_r4(anchor_unreproduced), False),
        ("R5 官方脚本通过", check_r5(good_eval, policy["eval"]["allowlist"]), True),
        ("R5 自造脚本 → 拦下", check_r5(bad_eval, policy["eval"]["allowlist"]), False),
        ("R6 有备份通过", check_r6(good_act), True),
        ("R6 破坏性无备份 → 拦下", check_r6(bad_act), False),
        ("R7 未碰保护文件通过", check_r7(["train_glas.py", "autorun/run_log.jsonl"], policy["protected_files"]), True),
        ("R7 改了 judge.py → 拦下", check_r7(["autorun/judge.py"], policy["protected_files"]), False),
        ("R7 改了 ledger.py → 拦下", check_r7(["autorun/ledger.py"], policy["protected_files"]), False),
        ("R8 留痕齐备通过", check_r8([dict(
            ts="t", run_id="r", task="GLaS", phase="judge", action="decide", actor="agent")]), True),
        ("R8 缺字段 → 拦下", check_r8([{"task": "GLaS"}]), False),
        ("R9 边界内通过", check_r9({"route_index": 1, "routes_failed": []}, policy), True),
        ("R9 连续失败越界 → 拦下", check_r9({"route_index": 1, "routes_failed": ["a", "b", "c", "d"]}, policy), False),
    ]

    ok = 0
    for name, res, expect_ok in cases:
        passed = res.ok == expect_ok
        ok += passed
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        if not passed:
            print(f"        expected ok={expect_ok}, got ok={res.ok} — {res.msg}")
    print(f"\nredlines selftest: {ok}/{len(cases)} passed")
    return 0 if ok == len(cases) else 1


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    if "--audit" in sys.argv:
        return audit()
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
