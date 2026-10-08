# -*- coding: utf-8 -*-
"""step2_cycle.py — Step 2：把 GLaS 的一轮「复现 → 改进 → 判定 → 换路线」钉成一条可复现命令。

Step 2 的纪律是**先手工走通一轮**（暴露数据格式 / 口径 / 回填格式的实际问题），
再把手工步骤固化成脚本（Phase 2 的 runner.py / loop.py）。本脚本就是那一轮的固化：
它只**推进状态 + 写留痕**，不做任何训练。

一轮的输入（均由前面的步骤产出，本脚本不重算）：
  · 锚点   R05 U-Node 复现 testB ObjDice 0.8428 → state.anchor（红线 R10：不计入 best）
  · 改进   R04 自家 nnU-Net 5 折 ensemble + 后处理，10 个配置的扫描结果由 tune_glas_pp.py 给出

判定口径（2026-10-08 人工裁定，本脚本沿用）：
  · **零训练路线的每次「试算」计为一次小轮**（small round）。
    理由：判定器只认「小轮/大轮连续无提升 → 换路线」，而 R04 是纯后处理、不产生训练轮次；
    但每次后处理试算本身就是一次「低成本试错」，成本比 1 折 30 epoch 还低，属同一语义。
    没有这条口径，判定器无法让循环离开一条已耗尽的零训练路线（Step 2 暴露的实际缺口）。

幂等：run_id 已存在于 run_log.jsonl 则整段跳过，避免重复追加（账本只追加、不改写）。
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
from ledger import append_run_log, append_claim, load_jsonl, DEFAULT_RUN_LOG  # noqa: E402

STATE = HERE / "state.json"
POLICY = HERE / "policy.json"
EPISODE = HERE / "episodes" / "EP03_GLaS_cycle"

RUN_IDS = (
    "step2-advance-r05-to-r04",
    "improve-20261008-r04",
    "switch-20261008-r04-to-r06",
)

# ── R04 扫描结果（testB ObjDice = 判定指标），顺序同 tune_glas_pp.py 的 CONFIGS ──
R04_SWEEP = [
    ("raw", 0.8184),
    ("open m30 k3 x1", 0.8185),
    ("open m30 k3 x2", 0.8183),
    ("open m30 k3 x3", 0.8184),
    ("close k3 x1", 0.8142),
    ("close k3 x2", 0.8078),
    ("close k5 x1", 0.8078),
    ("openx3+close", 0.8172),
    ("openx1+close", 0.8172),
    ("ws fd0.3 m30", 0.7249),
    ("wssel s0.85 fd0.4", 0.7634),
]
R04_RAW_F1 = 0.6113      # testB ObjF1，自家 nnU-Net 基线
R04_BEST_F1 = 0.7222     # testB ObjF1，open m30 k3 x3 后处理（选配置用 testA）
R04_BEST_CFG = "open m30 k3 x3"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def already_done() -> bool:
    ids = {r.get("run_id") for r in load_jsonl(DEFAULT_RUN_LOG)}
    return any(i in ids for i in RUN_IDS)


def build_episode() -> dict:
    """把本轮追加进账本的那一段（从 step2-advance 起到文件末尾）打成可重放证据包。

    必须按**位置切片**而不是按 run_id 过滤：judge 决策记录的 run_id 是
    judge-<时间戳>，不在 RUN_IDS 里；只按 run_id 过滤会把决策记录漏掉，
    证据包里就一条决策都没有 —— 这正是第一次跑时暴露的坑。
    """
    log = load_jsonl(DEFAULT_RUN_LOG)
    start = next(i for i, r in enumerate(log) if r.get("run_id") == "step2-advance-r05-to-r04")
    cycle = log[start:]
    EPISODE.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(POLICY, EPISODE / "policy_snapshot.json")
    (EPISODE / "run_log.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in cycle) + "\n", encoding="utf-8")
    save(EPISODE / "state.json", load(STATE))
    rep = replay.replay(EPISODE / "run_log.jsonl")
    save(EPISODE / "replay_report.json", rep)
    return rep


def main() -> int:
    if "--episode-only" in sys.argv:
        rep = build_episode()
        print(json.dumps({k: rep[k] for k in
                          ("n_decisions", "n_matched", "n_mismatched", "replayable", "chain_hash")},
                         ensure_ascii=False, indent=2))
        return 0 if rep["replayable"] else 1

    if already_done():
        print("Step 2 这一轮已留痕过（run_id 命中），跳过，不重复追加。")
        return 0

    state = load(STATE)
    ts = now()

    # ── A) 锚点已成立，按 ROUTE_POOL 顺序推进到改进路线 R04（R05 保持 anchor，不算失败）──
    state["current_route"] = "R04_NNUNetImproved"
    state["route_index"] = 1
    state["phase"] = "improve"
    state["updated_at"] = ts
    append_run_log(dict(
        run_id="step2-advance-r05-to-r04", task="GLaS", phase="improve",
        action="锚点 R05 U-Node 复现成立（testB ObjDice 0.8428，state.anchor），按 ROUTE_POOL 顺序"
               "推进到第 2 顺位改进路线 R04_NNUNetImproved：route_index 0→1。R05 保持 anchor，"
               "不计入 routes_failed（它是锚点，不是失败路线）。",
        actor="agent", artifact="autorun/state.json", destructive=False, ts=ts,
    ))

    # ── B) 记录 R04 改进结果，并回填 state.best ──
    hist = [v for _, v in R04_SWEEP]
    state["best"] = {
        "metric": "testB_objDice",
        "value": hist[0],
        "source": "ours",
        "note": ("R04：自家 nnU-Net 5 折 ensemble 预测 + 后处理。判定指标 testB ObjDice 全程 0.8184，"
                 "与基线同值（后处理改不动「实例分离」这个真瓶颈）。ObjF1 0.6113→0.7222 属非判定指标。"
                 "按裁定，零训练路线的每次试算计一次小轮，10 次均无提升（相对提升 <0.3%）。"),
    }
    state["small_rounds"] = {
        "count": len(hist) - 1,
        "history": hist,
        "note": ("零训练试错轮（R04 后处理扫描，testB ObjDice，顺序同 tune_glas_pp.py 的 CONFIGS）；"
                 "history[0] 为基线起点。判定口径见 step2_cycle.py 文件头。"),
    }
    state["updated_at"] = ts
    save(STATE, state)

    append_run_log(dict(
        run_id="improve-20261008-r04", task="GLaS", phase="improve",
        action=(f"R04 改进：在自家 nnU-Net 预测（glas_test_pred）上扫描 10 种后处理（开/闭运算、"
                f"分水岭、选择性分水岭）。判定指标 testB ObjDice 全程 {hist[0]}（零提升，"
                f"相对提升均 <0.3%）；ObjF1 {R04_RAW_F1}→{R04_BEST_F1}（+0.11，非判定指标）。"
                f"选配置用 testA（{R04_BEST_CFG}，testA ObjDice 0.8738）、报告用 testB。"
                f"结论：R04 对判定指标零提升 —— 真瓶颈是实例分离，后处理动不了它。"),
        actor="agent", artifact="tune_glas_pp.py, glas_test_pred_pp/", destructive=False, ts=ts,
    ))
    append_claim(dict(
        task="GLaS", metric="testB_objF1", subset="testB", convention="official_object_level",
        value=R04_BEST_F1, source_url="RESEARCH_LOG.md#glas-r04-pp", source_type="self_measured",
        verified=True, role="improvement",
        note=(f"R04 后处理（{R04_BEST_CFG}）后的 testB ObjF1；基线为 {R04_RAW_F1}。"
              f"非判定指标（判定指标是 testB_objDice），仅作改进幅度记录。"),
    ))

    # ── C) 判定（写进 run_log.jsonl，强制带 state_snapshot）──
    d1, r1, detail1, _ = judge.log_decision()

    # ── D) 按决策换路线 ──
    if d1 == "switch":
        state = load(STATE)
        state["routes_failed"] = list(state.get("routes_failed") or []) + ["R04_NNUNetImproved"]
        state["current_route"] = "R06_GLaS_UNet"
        state["route_index"] = 2
        state["small_rounds"] = {"count": 0, "history": []}
        state["big_rounds"] = {"count": 0, "history": []}
        state["updated_at"] = now()
        save(STATE, state)
        append_run_log(dict(
            run_id="switch-20261008-r04-to-r06", task="GLaS", phase="improve",
            action=(f"按 judge 决策 {d1}/{r1} 换路线：R04_NNUNetImproved 记入 routes_failed"
                    f"（对判定指标零提升），route_index 1→2，current_route → R06_GLaS_UNet，"
                    f"小轮/大轮计数归零。"),
            actor="agent", artifact="autorun/state.json", destructive=False, ts=now(),
        ))
        # ── E) 换路线后再判一次（确认新状态下的决策）──
        d2, r2, detail2, _ = judge.log_decision()
    else:
        d2 = r2 = detail2 = None

    # ── F) 证据包（可重放）──
    rep = build_episode()

    print(json.dumps({
        "judge#1": {"decision": d1, "reason": r1, "detail": detail1},
        "judge#2": {"decision": d2, "reason": r2, "detail": detail2},
        "replay": {k: rep[k] for k in ("n_decisions", "n_matched", "n_mismatched", "replayable", "chain_hash")},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
