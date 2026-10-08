# -*- coding: utf-8 -*-
"""确定性判定器 —— 全自动分割 Agent 的"裁判"。

设计要点：判定逻辑写在**代码里**，不写在 LLM 的记忆里。
因此换会话、换模型，判定结果完全一致 —— 这是可复现的基础。

用法:
    python judge.py                 # 读 state.json + policy.json，输出决策
    python judge.py --log           # 判定并写进 run_log.jsonl（强制带 state_snapshot）
    python judge.py --selftest      # 跑内置用例，验证判定逻辑正确

输入:
    policy.json  —— 机器可读阈值（唯一数据源）
    state.json   —— 当前运行状态
输出:
    stdout 一行 JSON: {"decision": ..., "reason": ..., "detail": ...}
      decision ∈ {continue, switch, stop}
"""
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
EPS = 1e-9


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def no_gain_streak(history, rel_gain):
    """从指标历史算「连续无提升」计数（相对阈值口径）。

    逐项与「历史最佳」比较：提升 < rel_gain 记为无提升，连续累加；一旦提升达标即清零。
    """
    if len(history) < 2:
        return 0
    best = history[0]
    streak = 0
    for v in history[1:]:
        if (v - best) / max(best, EPS) < rel_gain:
            streak += 1
        else:
            streak = 0
        if v > best:
            best = v
    return streak


def decide(state, policy):
    """核心判定。返回 (decision, reason, detail)。"""
    tgt = policy["targets"].get(state["task"])
    if tgt is None:
        # 数据集被下线 / 尚未就绪 / 名字拼错 —— 不崩，但必须停，交给人看
        return "stop", "UNKNOWN_TASK", f"policy.targets 里没有任务 {state['task']!r}"
    thr = policy["thresholds"]
    lim = policy["limits"]
    best_obj = state.get("best") or {}
    best = best_obj.get("value", 0.0)

    # 0) R10 锚点隔离：best 只能是「自研 / 改进」的结果。
    #    复现别人的方法拿到的分数必须写进 state.anchor；一旦它出现在 best 里，
    #    判定器宁可硬停交人复核，也绝不拿他人成绩判达标 —— 这是可信度的地基。
    if best and best_obj.get("source") != "ours":
        return "stop", "ANCHOR_IN_BEST", (
            f"best.source={best_obj.get('source')!r} 不是 'ours'（红线 R10）；"
            f"复现所得分数应写进 state.anchor，不得计入 best"
        )

    # 1) 达标 L2（若该任务定义了 L2）
    l2 = tgt.get("l2")
    if l2 is not None and best >= l2:
        return "stop", "SUCCESS_L2", f"best={best} >= L2({l2})"

    # 2) 小轮连续无提升 → 回 Step 2 再试
    s_streak = no_gain_streak(state["small_rounds"]["history"], thr["small"]["rel_gain"])
    if s_streak >= thr["small"]["max_no_gain_rounds"]:
        return "switch", "SMALL_ROUND_NO_GAIN", f"小轮连续 {s_streak} 轮相对提升 < {thr['small']['rel_gain']}"

    # 3) 大轮连续无提升 → 弃该路线
    b_streak = no_gain_streak(state["big_rounds"]["history"], thr["big"]["rel_gain"])
    if b_streak >= thr["big"]["max_no_gain_rounds"]:
        return "switch", "BIG_ROUND_NO_GAIN", f"大轮连续 {b_streak} 轮相对提升 < {thr['big']['rel_gain']}"

    # 4) 连续失败路线过多
    if len(state["routes_failed"]) >= lim["max_consecutive_route_failures"]:
        return "stop", "CONSECUTIVE_FAILURE", f"连续失败 {len(state['routes_failed'])} 条路线"

    # 5) 候选池耗尽
    if state["route_index"] >= state["route_total"]:
        if best >= tgt["l1"]:
            return "stop", "SUCCESS_L1", f"池耗尽但已达 L1(best={best})"
        return "stop", "POOL_EXHAUSTED", "候选池耗尽且未达 L1"

    # 6) 已达 L1，继续冲 L2
    if best >= tgt["l1"]:
        return "continue", "L1_REACHED_KEEP_PUSHING", f"best={best} 已达 L1，继续冲 L2"

    # 7) 进行中
    return "continue", "IN_PROGRESS", ""


def base_state(**over):
    s = {
        "task": "GLaS",
        "current_route": "R01",
        "route_index": 0,
        "route_total": 4,
        "small_rounds": {"count": 0, "history": []},
        "big_rounds": {"count": 0, "history": []},
        "best": {"metric": "testB_objDice", "value": 0.0, "source": "ours"},
        "anchor": None,
        "routes_failed": [],
        "stop_reason": None,
    }
    s.update(over)
    return s


def ours(v: float) -> dict:
    """自研成绩的 best 写法 —— 带 source='ours'，才允许参与达标判定（R10）。"""
    return {"metric": "testB_objDice", "value": v, "source": "ours"}


def selftest():
    policy = load_json(HERE / "policy.json")
    cases = [
        ("达标L2", base_state(best=ours(0.845)), "stop", "SUCCESS_L2"),
        ("小轮判负", base_state(small_rounds={"count": 3, "history": [0.800, 0.8005, 0.801, 0.8012]}), "switch", "SMALL_ROUND_NO_GAIN"),
        ("大轮判负", base_state(big_rounds={"count": 3, "history": [0.820, 0.821, 0.8215]}), "switch", "BIG_ROUND_NO_GAIN"),
        ("池耗尽未达标", base_state(route_index=4, route_total=4), "stop", "POOL_EXHAUSTED"),
        ("池耗尽但达L1", base_state(route_index=4, route_total=4, best=ours(0.82)), "stop", "SUCCESS_L1"),
        ("连续失败", base_state(routes_failed=["R1", "R2", "R3"]), "stop", "CONSECUTIVE_FAILURE"),
        ("已达L1继续冲", base_state(best=ours(0.82)), "continue", "L1_REACHED_KEEP_PUSHING"),
        ("进行中", base_state(), "continue", "IN_PROGRESS"),
        ("未知任务", base_state(task="NOT_A_DATASET"), "stop", "UNKNOWN_TASK"),
        # R10 对抗用例：把复现（别人方法）的分数塞进 best，必须硬停而不是判达标
        ("复现值混进best", base_state(best={"metric": "testB_objDice", "value": 0.842, "source": "reproduce"}),
         "stop", "ANCHOR_IN_BEST"),
        ("best缺source", base_state(best={"metric": "testB_objDice", "value": 0.9}), "stop", "ANCHOR_IN_BEST"),
    ]
    ok = 0
    for name, st, exp_d, exp_r in cases:
        d, r, detail = decide(st, policy)
        passed = (d == exp_d and r == exp_r)
        ok += passed
        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] {name:12s} → {d:8s} {r:28s} {detail}")
        if not passed:
            print(f"        expected: {exp_d} {exp_r}")
    print(f"\nselftest: {ok}/{len(cases)} passed")
    return 0 if ok == len(cases) else 1


def log_decision(state_path=None, run_log=None):
    """判定一次，并把结果写进 run_log.jsonl —— 强制带 state_snapshot。

    为什么必须带快照：replay.py 事后就靠这份快照复算。一条没有快照的决策记录，
    等于"我说我是按规则判的"却拿不出证据 —— 账本会直接拒收（ledger.validate_run_log），
    重放也会判它不合格。所以把"留痕"做成一条命令，避免手写时漏掉快照。
    """
    from ledger import DEFAULT_RUN_LOG, append_run_log

    state = load_json(state_path or (HERE / "state.json"))
    policy = load_json(HERE / "policy.json")
    d, reason, detail = decide(state, policy)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rec = append_run_log(
        dict(
            run_id=f"judge-{ts}",
            task=state.get("task"),
            phase="judge",
            action="judge.decide",
            actor="agent",
            decision=d,
            reason=reason,
            detail=detail,
            state_snapshot=state,
        ),
        path=run_log or DEFAULT_RUN_LOG,
    )
    return d, reason, detail, rec


def main():
    if "--selftest" in sys.argv:
        return selftest()
    if "--log" in sys.argv:
        d, reason, detail, _ = log_decision()
        print(json.dumps({"decision": d, "reason": reason, "detail": detail}, ensure_ascii=False))
        print("已留痕 → autorun/run_log.jsonl（含 state_snapshot）")
        return 0
    state = load_json(HERE / "state.json")
    policy = load_json(HERE / "policy.json")
    d, r, detail = decide(state, policy)
    print(json.dumps({"decision": d, "reason": r, "detail": detail}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
