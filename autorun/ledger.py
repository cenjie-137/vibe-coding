# -*- coding: utf-8 -*-
"""溯源账本 (Provenance Ledger) —— 让每个数字都能被追问"你从哪来的？"

这是"可信自主性"的地基：Agent 报出来的每一个数字、做的每一个动作，
都必须留下可被第三方核查的痕迹。

落地两条红线：
  R2 来源强制：任何基线 / SOTA 数字必须带 source_url；无来源标 UNVERIFIED，不得参与判定。
  R8 留痕强制：每个动作写 run_log.jsonl；每个结论写 RESEARCH_LOG.md。

两个 JSONL 文件（每行一个 JSON 对象，**只追加、不改写**——这是可审计的前提）：

  run_log.jsonl   一行 = 一个动作（谁、何时、做了什么、是否破坏性、有没有备份）
  claims.jsonl    一行 = 一个被报告的数字（指标、子集、口径、来源、能否参与判定）

用法：
    python ledger.py --selftest      # 跑内置用例
    python ledger.py --scorecard     # 打印当前账本的"可信记分卡"
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RUN_LOG = HERE / "run_log.jsonl"
DEFAULT_CLAIMS = HERE / "claims.jsonl"

# ── run_log.jsonl 的字段契约 ──────────────────────────────────────
RUNLOG_REQUIRED = ("ts", "run_id", "task", "phase", "action", "actor")
RUNLOG_PHASES = (
    "research", "reproduce", "improve", "small_round",
    "big_round", "judge", "terminate", "human_review", "governance",
)
ACTORS = ("agent", "human")

# ── claims.jsonl 的字段契约 ───────────────────────────────────────
CLAIM_REQUIRED = (
    "claim_id", "task", "metric", "subset",
    "convention", "value", "source_url", "verified",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# ── 校验 ──────────────────────────────────────────────────────────
def validate_run_log(rec: dict) -> list[str]:
    errs = [f"缺少字段 {k}" for k in RUNLOG_REQUIRED if k not in rec]
    if rec.get("actor") not in ACTORS:
        errs.append(f"actor 必须是 {ACTORS} 之一，实际 {rec.get('actor')!r}")
    if rec.get("phase") not in RUNLOG_PHASES:
        errs.append(f"phase 非法：{rec.get('phase')!r}")
    if rec.get("destructive") and not str(rec.get("backup_path") or "").strip():
        errs.append("R6：destructive=True 但未提供 backup_path")
    if rec.get("decision") and not rec.get("reason"):
        errs.append("judge 决策必须带 reason 代码（否则无法回放）")
    if rec.get("decision") and not isinstance(rec.get("state_snapshot"), dict):
        errs.append("judge 决策必须带 state_snapshot（当时的 state.json，否则 replay.py 无从复现）")
    return errs


def is_decision_eligible(claim: dict) -> bool:
    """R2：只有「已核实 + 有来源」的数字才允许参与判定。"""
    return bool(claim.get("verified")) and bool(str(claim.get("source_url") or "").strip())


def validate_claim(c: dict) -> list[str]:
    errs = [f"缺少字段 {k}" for k in CLAIM_REQUIRED if k not in c]
    if c.get("verified") and not str(c.get("source_url") or "").strip():
        errs.append("R2：verified=True 但 source_url 为空")
    if c.get("decision_eligible") is True and not is_decision_eligible(c):
        errs.append("R2：标记 decision_eligible 但无有效来源")
    return errs


# ── 写入（校验不过就拒绝写入，从源头挡住脏数据）────────────────────
def _append(path: Path, rec: dict, validator) -> dict:
    errs = validator(rec)
    if errs:
        raise ValueError(f"账本拒绝写入 {path.name}： " + "; ".join(errs))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def append_run_log(rec: dict, path=DEFAULT_RUN_LOG) -> dict:
    if not rec.get("ts"):
        rec = {**rec, "ts": now_iso()}
    return _append(Path(path), rec, validate_run_log)


def append_claim(c: dict, path=DEFAULT_CLAIMS) -> dict:
    if not c.get("claim_id"):
        c = {**c, "claim_id": f"C{datetime.now().strftime('%Y%m%d%H%M%S%f')[:18]}"}
    c = {**c, "decision_eligible": is_decision_eligible(c)}
    return _append(Path(path), c, validate_claim)


# ── 读取与统计 ────────────────────────────────────────────────────
def load_jsonl(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def source_coverage(claims: list[dict]) -> float:
    """来源覆盖率 = 可参与判定的数字 / 全部数字。这是论文里的一枚硬指标。"""
    if not claims:
        return 0.0
    return sum(is_decision_eligible(c) for c in claims) / len(claims)


def trust_scorecard(claims=None, run_log=None) -> dict:
    """可信记分卡 —— 论文里那张"我们的 Agent 有多可信"的表。"""
    claims = load_jsonl(DEFAULT_CLAIMS) if claims is None else claims
    run_log = load_jsonl(DEFAULT_RUN_LOG) if run_log is None else run_log
    destructive = [r for r in run_log if r.get("destructive")]
    return {
        "n_claims": len(claims),
        "source_coverage": round(source_coverage(claims), 4),
        "n_actions": len(run_log),
        "human_interventions": sum(1 for r in run_log if r.get("actor") == "human"),
        "destructive_actions": len(destructive),
        "destructive_without_backup": sum(
            1 for r in destructive if not str(r.get("backup_path") or "").strip()
        ),
        "judge_calls": sum(1 for r in run_log if r.get("decision")),
    }


# ── 自检 ──────────────────────────────────────────────────────────
def selftest() -> int:
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    rl, cl = tmp / "run_log.jsonl", tmp / "claims.jsonl"
    ok = total = 0

    def case(name, fn, expect_ok):
        nonlocal ok, total
        total += 1
        try:
            fn()
            got = True
        except (ValueError, AssertionError):
            got = False
        passed = got == expect_ok
        ok += passed
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        if not passed:
            print(f"        expected ok={expect_ok}, got ok={got}")

    good_action = dict(run_id="r1", task="GLaS", phase="reproduce", action="run train", actor="agent")
    case("合法动作可写入", lambda: append_run_log(dict(good_action), rl), True)
    case("缺字段被拒", lambda: append_run_log({"task": "GLaS"}, rl), False)
    case(
        "破坏性无备份被拒 (R6)",
        lambda: append_run_log({**good_action, "destructive": True}, rl),
        False,
    )
    case(
        "破坏性有备份可写入",
        lambda: append_run_log({**good_action, "destructive": True, "backup_path": "bak/"}, rl),
        True,
    )
    case(
        "决策记录缺 state_snapshot 被拒（无法重放）",
        lambda: append_run_log({**good_action, "phase": "judge", "decision": "switch", "reason": "X"}, rl),
        False,
    )
    case(
        "决策记录带 state_snapshot 可写入",
        lambda: append_run_log(
            {**good_action, "phase": "judge", "decision": "switch", "reason": "X",
             "state_snapshot": {"task": "GLaS", "route_index": 0}},
            rl,
        ),
        True,
    )

    good_claim = dict(
        task="GLaS", metric="testB_objDice", subset="testB",
        convention="official_object_level", value=0.836,
        source_url="https://doi.org/10.1016/j.media.2018.11.001", verified=True,
    )
    case("有来源数字可写入", lambda: append_claim(dict(good_claim), cl), True)
    case(
        "有来源但未核实 → 允许写入但不参与判定",
        lambda: append_claim({**good_claim, "verified": False}, cl),
        True,
    )
    case(
        "verified=True 却无来源被拒 (R2)",
        lambda: append_claim({**good_claim, "source_url": ""}, cl),
        False,
    )

    claims = load_jsonl(cl)
    cov = source_coverage(claims)
    assert claims, "claims 应已写入"
    sc = trust_scorecard(claims=claims, run_log=load_jsonl(rl))
    total += 1
    print(f"[PASS] 记分卡可生成：source_coverage={cov:.2f}, n_claims={sc['n_claims']}, n_actions={sc['n_actions']}")
    ok += 1

    print(f"\nledger selftest: {ok}/{total} passed")
    return 0 if ok == total else 1


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    if "--scorecard" in sys.argv:
        print(json.dumps(trust_scorecard(), ensure_ascii=False, indent=2))
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
