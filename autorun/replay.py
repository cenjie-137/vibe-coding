# -*- coding: utf-8 -*-
"""重放验证器 —— "可信"两个字的技术证据。

问题：Agent 说"我是按规则判的"，你凭什么信？
答案：把当时的**状态快照**存下来，事后用同一个判定器再判一遍。
      判出来一模一样 → 说明判定确实由规则决定，不是 LLM 临场发挥。
      判出来不一样 → 说明有人（或模型）偷偷改了判定逻辑。

这是"可复现"从口号变成事实的那一步：
    run_log.jsonl 里每条决策都带 state_snapshot（当时的 state.json）
    → replay.py 拿 state_snapshot + episode 自带的 policy_snapshot.json 重跑 judge.decide()
      （规则版本钉死在"当时"那份，而不是"今天"那份 —— 否则改一次阈值，历史全部失效）
    → 比对 (decision, reason) 是否逐条一致

判定严格程度分两级（刻意设计）：
    硬比对 decision + reason —— 语义层，不一致即 FAIL
    链哈希  连 detail 一起算 —— 连文案改动都会让哈希变，用于检测"悄悄改了判定"

用法：
    python replay.py --trace autorun/run_log.jsonl      # 重放一个账本
    python replay.py --selftest                         # 自检（含"篡改必被发现"用例）
    python replay.py --demo                             # 生成一份完整可重放证据包并当场验证
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import judge  # noqa: E402

DEFAULT_POLICY = HERE / "policy.json"
DEFAULT_TRACE = HERE / "run_log.jsonl"
EPISODES_DIR = HERE / "episodes"


# ── 核心：重放 ────────────────────────────────────────────────────
def decision_records(run_log: list[dict]) -> list[dict]:
    return [r for r in run_log if r.get("decision")]


def chain_hash(entries: list[dict]) -> str:
    """决策链哈希：把整条决策序列压成一个指纹。

    同样的输入必须得到同样的哈希 —— 这是"可重放"的可验证形式：
    论文里报一个哈希，任何人拉下代码跑一遍，哈希对得上就说明没被改过。
    """
    payload = [
        {
            "task": e.get("task"),
            "decision": e.get("decision"),
            "reason": e.get("reason"),
            "detail": e.get("detail"),
            "best": (e.get("state_snapshot") or {}).get("best", {}).get("value"),
        }
        for e in entries
    ]
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def resolve_policy(trace_path, policy_path=None):
    """判定用哪份规则？优先 episode 自带的 policy_snapshot.json。

    为什么必须钉死规则版本：只要改一次判定字段（比如 L2 从 0.836 上移到 0.842），
    拿"今天的 policy"去重放"昨天的记录"，结果必然对不上 —— 那不是留痕错了，
    是拿今天的尺子量昨天的账。所以重放要用**当时**那份规则，才有资格叫"可复现"。

    返回 (规则文件路径, 来源标记)：explicit（显式指定）/ snapshot（episode 快照）/ current（回退到当前）
    """
    if policy_path is not None:
        return Path(policy_path), "explicit"
    snap = Path(trace_path).parent / "policy_snapshot.json"
    if snap.exists():
        return snap, "snapshot"
    return DEFAULT_POLICY, "current"


def replay(trace_path=DEFAULT_TRACE, policy_path=None) -> dict:
    """把账本里每条决策重放一遍，返回一份可写进论文的验证报告。"""
    pol_path, pol_src = resolve_policy(trace_path, policy_path)
    policy = json.loads(Path(pol_path).read_text(encoding="utf-8"))
    trace = [json.loads(l) for l in Path(trace_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    recs = decision_records(trace)

    replayed: list[dict] = []
    mismatches: list[dict] = []
    for i, r in enumerate(recs, 1):
        snap = r.get("state_snapshot")
        if not isinstance(snap, dict):
            mismatches.append({"idx": i, "run_id": r.get("run_id"), "why": "缺少 state_snapshot，无法重放"})
            continue
        if snap.get("task") not in policy.get("targets", {}):
            mismatches.append({
                "idx": i, "run_id": r.get("run_id"),
                "why": f"policy 里没有任务 {snap.get('task')!r}（可能数据集状态变了）",
            })
            continue

        d, reason, detail = judge.decide(snap, policy)
        entry = {
            "task": snap.get("task"),
            "decision": d,
            "reason": reason,
            "detail": detail,
            "state_snapshot": snap,
        }
        replayed.append(entry)

        if (d, reason) != (r.get("decision"), r.get("reason")):
            mismatches.append({
                "idx": i, "run_id": r.get("run_id"),
                "why": "重放结果与留痕不一致",
                "logged": {"decision": r.get("decision"), "reason": r.get("reason")},
                "replayed": {"decision": d, "reason": reason},
            })

    n = len(recs)
    report = {
        "trace": str(Path(trace_path).relative_to(HERE.parent)).replace("\\", "/")
        if str(trace_path).startswith(str(HERE.parent)) else str(trace_path),
        "policy_version": policy.get("version"),
        "policy_source": pol_src,
        "policy_path": str(pol_path.relative_to(HERE.parent)).replace("\\", "/")
        if str(pol_path).startswith(str(HERE.parent)) else str(pol_path),
        "n_decisions": n,
        "n_replayed": len(replayed),
        "n_matched": len(replayed) - len([m for m in mismatches if m.get("replayed")]),
        "n_mismatched": len(mismatches),
        "replayable": len(mismatches) == 0 and n > 0,
        "chain_hash": chain_hash(replayed),
        "mismatches": mismatches,
    }
    return report


# ── 自检 ──────────────────────────────────────────────────────────
def _write_trace(path: Path, states: list[dict], policy: dict, tamper: int | None = None) -> None:
    """把一串状态喂给 judge，生成一份"真实"的决策留痕（可选在指定位置篡改）。"""
    lines = []
    for i, st in enumerate(states, 1):
        d, reason, detail = judge.decide(st, policy)
        rec = {
            "ts": f"2026-10-08T00:00:{i:02d}+08:00",
            "run_id": f"ep-{i:02d}",
            "task": st["task"],
            "phase": "judge",
            "action": "judge.decide",
            "actor": "agent",
            "decision": d,
            "reason": reason,
            "detail": detail,
            "state_snapshot": st,
        }
        if tamper == i:
            rec["decision"] = "continue" if d != "continue" else "stop"
            rec["reason"] = "TAMPERED"
        lines.append(json.dumps(rec, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _demo_states() -> list[dict]:
    """一条真实的 GLaS 成功轨迹：从零跑到达标（best 均为自研结果，R10）。"""
    def st(best, route_index=0, small=None, big=None, failed=None):
        return {
            "task": "GLaS",
            "current_route": "R05_UNODE",
            "route_index": route_index,
            "route_total": 5,
            "small_rounds": {"count": 0, "history": small or []},
            "big_rounds": {"count": 0, "history": big or []},
            "best": {"metric": "testB_objDice", "value": best, "source": "ours"},
            "anchor": {"route": "R05_UNODE", "reproduced_value": 0.838},
            "routes_failed": failed or [],
            "stop_reason": None,
        }

    return [
        st(0.0),                                    # 起跑 → IN_PROGRESS
        st(0.0, small=[0.0, 0.802]),                # 复现成功，拿到起点 → IN_PROGRESS
        st(0.802, big=[0.802, 0.812]),              # 大轮定论 → IN_PROGRESS
        st(0.820, big=[0.802, 0.812, 0.820]),       # 越过 L1(0.8184) → L1_REACHED_KEEP_PUSHING
        st(0.845, big=[0.812, 0.820, 0.845]),       # 越过 L2(0.842) → SUCCESS_L2
    ]


def selftest() -> int:
    import tempfile

    policy = json.loads(DEFAULT_POLICY.read_text(encoding="utf-8"))
    states = _demo_states()
    tmp = Path(tempfile.mkdtemp())
    ok = total = 0

    def check(name, cond, extra=""):
        nonlocal ok, total
        total += 1
        ok += bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {extra}" if extra else ""))

    # 1) 干净的留痕必须 100% 重放通过
    good = tmp / "good.jsonl"
    _write_trace(good, states, policy)
    rep = replay(good, DEFAULT_POLICY)
    check("干净留痕可重放", rep["replayable"] and rep["n_mismatched"] == 0,
          f"{rep['n_matched']}/{rep['n_decisions']} 条一致，hash={rep['chain_hash'][:20]}…")

    # 2) 同样的输入重放两次，哈希必须相同（确定性）
    rep2 = replay(good, DEFAULT_POLICY)
    check("重放确定性（同输入同哈希）", rep["chain_hash"] == rep2["chain_hash"],
          rep["chain_hash"][:20] + "…")

    # 3) 篡改一条决策 → 必须被发现
    bad = tmp / "bad.jsonl"
    _write_trace(bad, states, policy, tamper=3)
    rep3 = replay(bad, DEFAULT_POLICY)
    caught = rep3["n_mismatched"] == 1 and rep3["mismatches"][0]["logged"]["reason"] == "TAMPERED"
    check("篡改留痕必被发现", caught,
          f"抓到 {rep3['n_mismatched']} 条不一致" if caught else f"漏检！{rep3}")

    # 4) 缺少 state_snapshot → 拒绝重放（而不是假装通过）
    noload = tmp / "nosnap.jsonl"
    noload.write_text(json.dumps({"ts": "t", "run_id": "x", "task": "GLaS", "phase": "judge",
                                  "action": "a", "actor": "agent",
                                  "decision": "stop", "reason": "SUCCESS_L2"}) + "\n",
                      encoding="utf-8")
    rep4 = replay(noload, DEFAULT_POLICY)
    check("缺状态快照被拒（不假装通过）", (not rep4["replayable"]) and rep4["n_mismatched"] == 1)

    # 5) 任务不在 policy → 明确报错，不静默跳过
    ghost = tmp / "ghost.jsonl"
    _write_trace(ghost, [{**states[0], "task": "NONEXISTENT"}], policy)
    rep5 = replay(ghost, DEFAULT_POLICY)
    check("未知任务被明确拒绝", rep5["n_mismatched"] == 1)

    # 6) 规则版本钉死：episode 自带的 policy_snapshot 优先于"当前 policy"
    #    造一份"L2 被改高"的规则，历史记录按它判，快照里也放它。
    ep = tmp / "EP_pinned"
    ep.mkdir()
    pinned = json.loads(json.dumps(policy))
    pinned["targets"]["GLaS"]["l2"] = 0.90
    trace = ep / "run_log.jsonl"
    _write_trace(trace, states, pinned)
    (ep / "policy_snapshot.json").write_text(
        json.dumps(pinned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rep_snap = replay(trace)                  # 自动取 episode 快照 → 应一致
    rep_cur = replay(trace, DEFAULT_POLICY)   # 强行用当前 policy → 应不一致
    check("重放钉死 episode 自带规则版本",
          rep_snap["policy_source"] == "snapshot" and rep_snap["replayable"],
          f"policy_source={rep_snap['policy_source']}，{rep_snap['n_matched']}/{rep_snap['n_decisions']} 一致")
    check("拿当前 policy 重放旧记录 → 必对不上（这正是要钉死规则的原因）",
          (not rep_cur["replayable"]) and rep_cur["n_mismatched"] >= 1,
          f"{rep_cur['n_mismatched']} 条不一致")

    print(f"\nreplay selftest: {ok}/{total} passed")
    return 0 if ok == total else 1


# ── Demo：生成一份完整可重放证据包 ────────────────────────────────
def demo() -> int:
    policy = json.loads(DEFAULT_POLICY.read_text(encoding="utf-8"))
    ep = EPISODES_DIR / "EP01_GLaS_demo"
    ep.mkdir(parents=True, exist_ok=True)

    states = _demo_states()
    trace = ep / "run_log.jsonl"
    _write_trace(trace, states, policy)
    (ep / "state.json").write_text(
        json.dumps({**states[-1], "policy_version": policy["version"],
                    "stop_reason": "SUCCESS_L2"}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (ep / "policy_snapshot.json").write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rep = replay(trace)
    (ep / "replay_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"证据包已生成：{ep}")
    print(f"  run_log.jsonl        {rep['n_decisions']} 条决策（每条含 state_snapshot）")
    print(f"  state.json           终态")
    print(f"  policy_snapshot.json 判定时用的规则版本（重放以它为准）")
    print(f"  replay_report.json   重放验证报告")
    print()
    for line in report_lines(rep):
        print(line)
    return 0 if rep["replayable"] else 1


def report_lines(rep: dict) -> list[str]:
    out = [
        f"重放验证：{rep['n_matched']}/{rep['n_decisions']} 条决策一致，"
        f"{rep['n_mismatched']} 条不一致",
        f"policy 版本：{rep['policy_version']}（来源：{rep.get('policy_source')}）",
        f"决策链哈希：{rep['chain_hash']}",
        f"可重放：{'是' if rep['replayable'] else '否'}",
    ]
    for m in rep["mismatches"]:
        out.append(f"  ✗ #{m['idx']} {m.get('run_id')} —— {m.get('why')}")
    return out


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    if "--demo" in sys.argv:
        return demo()

    trace = DEFAULT_TRACE
    if "--trace" in sys.argv:
        i = sys.argv.index("--trace")
        if i + 1 < len(sys.argv):
            trace = Path(sys.argv[i + 1])
    if not Path(trace).exists():
        print(f"找不到账本：{trace}")
        return 1

    rep = replay(trace)
    if "--out" in sys.argv:
        i = sys.argv.index("--out")
        if i + 1 < len(sys.argv):
            Path(sys.argv[i + 1]).write_text(
                json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for line in report_lines(rep):
        print(line)
    return 0 if rep["replayable"] else 1


if __name__ == "__main__":
    sys.exit(main())
