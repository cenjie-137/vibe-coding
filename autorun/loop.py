# -*- coding: utf-8 -*-
"""loop.py — 循环编排器（Phase 2）

一句话：**一条命令跑完一整轮** —— 读 state → 判定 → 执行 → 回填 → 判定 → 按决策走，
直到终态（达标 / 池耗尽 / 连续失败 / 需服务器）。

它把 Step 2 里手工做的四件事固化成一条命令：
    judge.py（决策） + runner.py（执行） + 状态回填 + 证据包重放

职责边界（谁干什么，钉死不混）：
    judge.py   判定（纯函数，读 policy + state，不碰文件）
    runner.py  执行（产数 / 派发；不判定、不改 state）
    loop.py    编排（推进 state、写留痕、切证据包、按决策走）

纪律：
    幂等      —— `--run-id` 命中账本即整段跳过（账本只追加、不改写）。
    按位置切片 —— 证据包从「本轮首条记录」切到文件末尾（**不按 run_id 过滤** ——
                  judge 记录的 run_id 是 judge-<时间戳>，过滤会把它漏掉，Step 2 踩过这个坑）。
    沙箱可跑  —— `--state` / `--run-log` / `--episode` 可覆盖，验证时不动真账本。

用法：
    python loop.py --task GLaS                          # 一条命令跑一整轮
    python loop.py --task GLaS --max-rounds 8
    python loop.py --task GLaS --dry-run                # 只看计划，不执行、不留痕
    python loop.py --task GLaS --state <p> --run-log <p> --episode <name>   # 沙箱
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import judge  # noqa: E402
import replay  # noqa: E402
import runner  # noqa: E402
from ledger import append_run_log, append_claim, load_jsonl, DEFAULT_CLAIMS  # noqa: E402

STATE = HERE / "state.json"
POLICY = HERE / "policy.json"
RUN_LOG = HERE / "run_log.jsonl"
EPISODES = HERE / "episodes"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path: Path, obj: dict) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rel(path) -> str:
    p = Path(path)
    return p.relative_to(ROOT).as_posix() if str(p).startswith(str(ROOT)) else str(p)


def _default_state(task: str) -> Path:
    """该 task 的默认状态文件：有 `state_<task>.json` 就用它，否则回落到 state.json（GLaS）。"""
    p = HERE / f"state_{task}.json"
    return p if p.exists() else STATE


def argval(flag: str, default=None):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def advance_route(state: dict, route_ids: list[str]) -> str:
    """按该 task 的路线池顺序取下一条：当前路线记失败（锚点除外），计数归零。"""
    cur = state.get("current_route")
    pool = {r["id"]: r for r in runner.load_routes(state.get("task") or "GLaS")}
    if cur and pool.get(cur, {}).get("status") != "anchor" \
            and cur not in (state.get("routes_failed") or []):
        state["routes_failed"] = list(state.get("routes_failed") or []) + [cur]
    nxt = int(state.get("route_index", 0)) + 1
    state["route_index"] = nxt
    state["current_route"] = route_ids[nxt] if nxt < len(route_ids) else None
    state["route_total"] = len(route_ids)
    state["small_rounds"] = {"count": 0, "history": []}
    state["big_rounds"] = {"count": 0, "history": []}
    state["phase"] = "improve"
    state["updated_at"] = now()
    return f"{cur} → {state['current_route']}（route_index {nxt}，小轮/大轮归零）"


def backfill(state: dict, res: dict) -> str:
    """把一轮结果写回 state。锚点路线走 anchor（R10），其余走 small_rounds + best。"""
    m = res.get("metric")
    if not m:
        return "无指标（该路线的配置已试完）"
    v = float(m["value"])
    if res.get("target") == "anchor":
        state["anchor"] = {**(state.get("anchor") or {}),
                           "route": res["route"], "metric": m["name"], "subset": m["subset"],
                           "convention": m["convention"], "reproduced_value": v,
                           "source_url": res.get("source_url"), "verified": True,
                           "note": "runner 本地复现（官方脚本、实例标签口径）；按红线 R10 不进 best。"}
        state["updated_at"] = now()
        return f"anchor={v}（R10：不进 best）"

    sr = state.setdefault("small_rounds", {"count": 0, "history": []})
    sr.setdefault("history", []).append(v)
    sr["count"] = max(0, len(sr["history"]) - 1)
    sr["note"] = "每轮由 runner 执行一次（零训练路线的每次试算 = 一次小轮，2026-10-08 裁定）。"
    best = state.get("best") or {}
    if v > float(best.get("value") or 0.0):
        state["best"] = {"metric": m["name"], "value": v, "source": "ours",
                         "note": f"{res['route']}：{res['detail']}"}
        upd = "best 更新"
    else:
        upd = "未超 best"
    state["updated_at"] = now()
    return f"small_rounds count={sr['count']}，{upd}"


def build_episode(state_path, run_log, start_idx: int, name: str) -> tuple[Path, dict]:
    """按位置切片生成本轮可重放证据包。

    name 含路径分隔符时按路径处理（沙箱用），否则落在 autorun/episodes/<name>。
    """
    cycle = load_jsonl(run_log)[start_idx:]
    ep = Path(name)
    if ep.parent == Path("."):
        ep = EPISODES / name
    ep.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(POLICY, ep / "policy_snapshot.json")
    (ep / "run_log.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in cycle) + "\n", encoding="utf-8")
    save(ep / "state.json", load(state_path))
    rep = replay.replay(ep / "run_log.jsonl")
    save(ep / "replay_report.json", rep)
    return ep, rep


def main() -> int:
    task = argval("--task", "GLaS")
    max_rounds = int(argval("--max-rounds", "8"))
    dry = "--dry-run" in sys.argv
    state_path = Path(argval("--state", str(_default_state(task))))
    run_log = Path(argval("--run-log", str(RUN_LOG)))
    claims = Path(argval("--claims", str(DEFAULT_CLAIMS)))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = argval("--run-id", f"loop-{task}-{ts}")
    episode = argval("--episode", f"EP_loop_{task}_{ts}")

    state = load(state_path)
    if state.get("task") != task:
        print(f"✗ state.task={state.get('task')!r} ≠ --task={task!r}")
        return 2

    policy = load(POLICY)
    route_ids = [r["id"] for r in runner.load_routes(task)]

    if dry:
        d, r, detail = judge.decide(state, policy)
        print(json.dumps({"decision": d, "reason": r, "detail": detail,
                          "route": state.get("current_route"), "plan": runner.plan(state)},
                         ensure_ascii=False, indent=2))
        return 0

    if state.get("stop_reason"):
        print(f"已是终态：{state['stop_reason']}（如要重跑，先清 state.stop_reason）")
        return 0

    if run_id in {rec.get("run_id") for rec in load_jsonl(run_log)}:
        print(f"{run_id} 已留痕过，跳过（幂等，不重复追加）。")
        return 0

    start_idx = len(load_jsonl(run_log))       # 证据包切片起点（按位置）
    print(f"▶ loop 起跑：task={task}  route={state.get('current_route')}  "
          f"route_index={state.get('route_index')}  max_rounds={max_rounds}")
    rounds = 0
    decisions: list[str] = []

    while rounds < max_rounds:
        d, reason, detail = judge.decide(state, policy)
        judge.log_decision(state_path=state_path, run_log=run_log)   # 判定留痕（带 state_snapshot）
        decisions.append(f"{d}/{reason}")

        if d == "stop":
            state["stop_reason"] = reason
            state["phase"] = "terminate"
            state["updated_at"] = now()
            save(state_path, state)
            append_run_log(dict(run_id=f"{run_id}-stop", task=task, phase="terminate",
                                action=f"循环终止：{reason} —— {detail}", actor="agent",
                                artifact=rel(state_path), destructive=False), path=run_log)
            print(f"■ 终止 {reason}：{detail}")
            break

        if d == "switch":
            act = advance_route(state, route_ids)
            save(state_path, state)
            append_run_log(dict(run_id=f"{run_id}-switch-{state['route_index']}", task=task,
                                phase="improve",
                                action=f"按 judge 决策 switch/{reason} 换路线：{act}",
                                actor="agent", artifact=rel(state_path), destructive=False),
                           path=run_log)
            print(f"→ 换路线：{act}")
            rounds += 1
            continue

        # d == continue → 执行当前路线的一轮
        res = runner.run(state)

        if res["status"] == "needs_remote":
            save(state_path, state)
            hf = f"autorun/handoff/{res['route']}.json"
            append_run_log(dict(run_id=f"{run_id}-dispatch-{res['route']}", task=task,
                                phase="improve",
                                action=f"路线 {res['route']} 需服务器执行，已派发 {hf}。"
                                       f"runner 不伪造数字，等服务器回填 autorun/results/{res['route']}.json 后续跑。",
                                actor="agent", artifact=hf, destructive=False), path=run_log)
            print(f"⏸ 派发到服务器：{res['route']}（本地无法训练）→ {hf}")
            break

        if res["status"] == "error":
            append_run_log(dict(run_id=f"{run_id}-error-{rounds}", task=task, phase="improve",
                                action=f"执行出错：{res['note']}", actor="agent",
                                destructive=False), path=run_log)
            print(f"✗ 执行出错：{res['note']}")
            break

        # status == done → 回填 state
        note = backfill(state, res)
        save(state_path, state)
        append_run_log(dict(
            run_id=f"{run_id}-round-{rounds + 1}", task=task,
            phase="reproduce" if res.get("target") == "anchor" else "improve",
            action=f"执行 {res['route']} 一轮：{res['detail']}；回填 {note}",
            actor="agent", artifact=", ".join(res.get("artifacts") or []) or None,
            destructive=False), path=run_log)
        if res.get("metric"):
            m = res["metric"]
            append_claim(dict(
                task=task, metric=m["name"], subset=m["subset"], convention=m["convention"],
                value=m["value"], source_url=res.get("source_url"), source_type="self_measured",
                verified=True,
                role="anchor_calibration" if res.get("target") == "anchor" else "improvement",
                note=res["detail"]), path=claims)
            print(f"· 轮 {rounds + 1}: {res['route']} → {m['name']}={m['value']}（{note}）")
        else:
            print(f"· 轮 {rounds + 1}: {res['route']} → 无指标（{note}）")
        rounds += 1

    ep, rep = build_episode(state_path, run_log, start_idx, episode)
    print(f"\n── 本轮小结 ──")
    print(f"  执行轮数：{rounds}；决策序列：{', '.join(decisions)}")
    print(f"  当前路线：{state.get('current_route')}（route_index {state.get('route_index')}）")
    print(f"  终态：{state.get('stop_reason') or '未终止（等待服务器回填 / 达到 max-rounds）'}")
    print(f"  证据包：{rel(ep)}")
    for line in replay.report_lines(rep):
        print(f"  {line}")
    return 0 if rep["replayable"] else 1


if __name__ == "__main__":
    sys.exit(main())
