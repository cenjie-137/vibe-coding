# -*- coding: utf-8 -*-
"""runner.py — 路线执行器（Phase 2）

一句话：**给 state，执行「当前路线」的一轮，产出可回填的结果。**
它只负责「跑」和「产数」；判定归 judge.py，状态推进归 loop.py。

两种执行环境（这是本项目的现实约束：训练在服务器、后处理在本地）：

  · 本地可跑（后处理 / 评估类）→ 真跑，产出真实指标。
  · 需服务器（训练类）          → **不伪造数字**，产出一份 handoff（要什么、
                                  结果写成什么格式、怎么续跑），等服务器回填结果文件。

设计红线（自己也得守规矩）：
  R5 评估锁定 —— 指标一律由 allowlist 里的官方脚本算出：本脚本用 subprocess 调
                 `eval_glas_official.py` 并解析它的 stdout，**runner 自己不算任何指标**。
  R2 来源强制 —— 每个数字都带 source（脚本 + 命令）；服务器结果文件缺来源一律拒收。
  不伪造     —— 跑不了就 `needs_remote`，绝不编一个数字出来（这是可信度的地基）。

用法：
    python runner.py --plan              # 打印当前路线要做什么（不执行）
    python runner.py --run               # 执行当前路线的一轮
    python runner.py --run --json        # 机器可读输出（loop.py 用）
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):   # ROOT：复用 tune_glas_pp / eval_glas_official；HERE：ledger 等
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml  # noqa: E402

ROUTE_POOL = ROOT / "ROUTE_POOL.yaml"
STATE = HERE / "state.json"

# ── 本地素材（GLaS）──────────────────────────────────────────────
EVAL_SCRIPT = ROOT / "eval_glas_official.py"     # R5：allowlist 里唯一允许的评估脚本
GLAS_GT = ROOT / "data" / "glas" / "masks"
GLAS_META = ROOT / "data" / "glas" / "metadata.csv"
GLAS_PRED = ROOT / "glas_test_pred"              # 自家 nnU-Net 5 折 ensemble 预测（二值）
GLAS_PP_ROOT = ROOT / "glas_test_pred_pp"        # 后处理产物（每配置一个子目录）
GLAS_UNODE_LABEL = ROOT / "glas_unode_pred_label"  # U-Node 官方权重预测（实例标签）

RESULTS_DIR = HERE / "results"                   # 服务器回填的结果文件放这里
HANDOFF_DIR = HERE / "handoff"                   # 训练类路线的派发单放这里

# 官方脚本输出的解析（`  testB      n= 20  ObjDice=0.8184  ObjF1=0.6113  ObjHD= 118.85`）
EVAL_LINE = re.compile(
    r"^\s*(test[AB])\s+n=\s*(\d+)\s+ObjDice=([\d.]+)\s+ObjF1=([\d.]+)\s+ObjHD=\s*([\d.]+)"
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_routes() -> list[dict]:
    """ROUTE_POOL.yaml 的 routes（顺序 = 尝试优先级）。"""
    pool = yaml.safe_load(ROUTE_POOL.read_text(encoding="utf-8"))
    return pool.get("routes") or []


def slug(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()


# ── R5：评估只走官方脚本，runner 不算指标 ─────────────────────────
def run_official_eval(pred_dir: Path, pred_kind: str = "binary") -> dict:
    """调 eval_glas_official.py 并解析 stdout。返回 {testA:{...}, testB:{...}}。"""
    cmd = [sys.executable, str(EVAL_SCRIPT),
           "--gt_dir", str(GLAS_GT), "--pred_dir", str(pred_dir),
           "--metadata", str(GLAS_META)]
    if pred_kind == "label":
        cmd += ["--pred_kind", "label"]
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"官方评估脚本失败（rc={p.returncode}）：\n{p.stdout}\n{p.stderr}")
    out: dict[str, dict] = {}
    for line in (p.stdout or "").splitlines():
        m = EVAL_LINE.match(line)
        if m:
            out[m.group(1)] = {"n": int(m.group(2)), "objDice": float(m.group(3)),
                               "objF1": float(m.group(4)), "objHD": float(m.group(5))}
    if "testB" not in out:
        raise RuntimeError(f"没能从评估输出解析到 testB：\n{p.stdout}")
    return out


def _cmd_str(pred_dir: Path, pred_kind: str = "binary") -> str:
    s = (f"python eval_glas_official.py --gt_dir data/glas/masks --pred_dir "
         f"{pred_dir.relative_to(ROOT).as_posix()} --metadata data/glas/metadata.csv")
    return s + (" --pred_kind label" if pred_kind == "label" else "")


# ── 本地处理器 ────────────────────────────────────────────────────
def handle_r04(state: dict) -> dict:
    """R04：自家 nnU-Net 预测 + 后处理。**每个后处理配置 = 一次小轮**（人工裁定口径）。

    按 state.small_rounds.history 的长度轮转 tune_glas_pp.CONFIGS，每次只试一个配置 ——
    一次试算就是一次「低成本试错」，与「1 折 30 epoch 的小轮」同一语义。
    （用 history 长度而不是 count：count = len-1（history[0] 是基线起点），
      直接用 count 会让第一个配置被重复试一次。）
    """
    import numpy as np
    from PIL import Image
    import tune_glas_pp  # 复用已核实的后处理实现，避免两处各写一份

    configs = tune_glas_pp.CONFIGS
    idx = len((state.get("small_rounds") or {}).get("history") or [])
    if idx >= len(configs):
        return {"route": "R04_NNUNetImproved", "status": "done", "metric": None,
                "note": f"R04 的 {len(configs)} 个后处理配置已试完（idx={idx}），无更多可试。"}

    label, fn, kw = configs[idx]
    if fn is None:
        pred_dir, detail = GLAS_PRED, "raw（无后处理）"
    else:
        pred_dir = GLAS_PP_ROOT / slug(label)
        pred_dir.mkdir(parents=True, exist_ok=True)
        if not any(pred_dir.glob("*.png")):
            for img in sorted(GLAS_PRED.glob("*.png")):
                b = tune_glas_pp.load_mask_binary(img)
                out = fn(b, **kw)
                Image.fromarray((out.astype(np.uint8)) * 255).save(pred_dir / img.name)
        detail = f"后处理 {label}"

    ev = run_official_eval(pred_dir)
    return {
        "route": "R04_NNUNetImproved", "status": "done",
        "metric": {"name": "testB_objDice", "value": ev["testB"]["objDice"],
                   "subset": "testB", "convention": "official_object_level"},
        "extra": {"testA_objDice": ev["testA"]["objDice"],
                  "testB_objF1": ev["testB"]["objF1"]},
        "artifacts": [pred_dir.relative_to(ROOT).as_posix()],
        "source_url": f"{_cmd_str(pred_dir)}（R5 allowlist 官方脚本）",
        "detail": f"config[{idx}]={label}；{detail}；testA ObjDice={ev['testA']['objDice']:.4f}",
    }


def handle_r05(state: dict) -> dict:
    """R05：U-Node 复现（锚点）。产物按红线 R10 写 state.anchor，不进 best。"""
    ev = run_official_eval(GLAS_UNODE_LABEL, pred_kind="label")
    return {
        "route": "R05_UNODE", "status": "done", "target": "anchor",
        "metric": {"name": "testB_objDice", "value": ev["testB"]["objDice"],
                   "subset": "testB", "convention": "official_object_level"},
        "extra": {"testA_objDice": ev["testA"]["objDice"],
                  "testB_objF1": ev["testB"]["objF1"]},
        "artifacts": [GLAS_UNODE_LABEL.relative_to(ROOT).as_posix()],
        "source_url": f"{_cmd_str(GLAS_UNODE_LABEL, 'label')}（R5 allowlist 官方脚本）",
        "detail": "U-Node 官方权重预测，实例标签形态评估（与论文 0.842 同口径）",
    }


# ── 训练类路线：派发到服务器（本地不伪造数字）──────────────────────
def handle_remote(route: dict, state: dict) -> dict:
    rid = route["id"]
    res_file = RESULTS_DIR / f"{rid}.json"

    # 服务器已回填结果 → 采纳（但必须先过 R2：无来源一律拒收）
    if res_file.exists():
        r = load_json(res_file)
        if not str(r.get("source_url") or "").strip():
            return {"route": rid, "status": "error",
                    "note": f"{res_file.name} 缺 source_url（红线 R2），拒绝采纳 —— 宁可不认，不认脏数。"}
        return {"route": rid, "status": "done",
                "metric": {"name": r.get("metric", "testB_objDice"), "value": r["value"],
                           "subset": r.get("subset", "testB"),
                           "convention": r.get("convention", "official_object_level")},
                "extra": r.get("extra") or {},
                "source_url": r["source_url"],
                "detail": f"来自服务器回填 {res_file.name}"}

    # 没回填 → 产出一张派发单（要什么 / 结果什么格式 / 怎么续跑）
    handoff = {
        "route": rid, "method": route.get("method"),
        "code": route.get("code"), "paper": route.get("paper"),
        "generated_at": now(), "task": state.get("task"),
        "why_remote": "训练类路线需 GPU + 数据集，本地不可执行（本地只跑后处理/评估类路线）。",
        "expected_result_file": f"autorun/results/{rid}.json",
        "expected_schema": {
            "route": rid, "metric": "testB_objDice", "value": "<float>",
            "subset": "testB", "convention": "official_object_level",
            "source_url": "<必须：结果由 eval_glas_official.py 产出，填脚本+命令或日志链接>",
            "evidence": "<权重目录 / 推理输出目录 / 训练日志>",
        },
        "how_to_resume": "服务器产出上述结果文件后，重跑 `python autorun/loop.py --task GLaS`，"
                         "loop 会自动读取并回填，无需人工改 state。",
        "note": "本文件是**派发单**，不是结果。runner 不会替服务器编一个数字。",
    }
    HANDOFF_DIR.mkdir(parents=True, exist_ok=True)
    (HANDOFF_DIR / f"{rid}.json").write_text(
        json.dumps(handoff, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"route": rid, "status": "needs_remote", "handoff": handoff,
            "note": f"{rid} 需服务器执行；已产出派发单 autorun/handoff/{rid}.json"}


# ── 调度 ──────────────────────────────────────────────────────────
LOCAL_HANDLERS = {"R04_NNUNetImproved": handle_r04, "R05_UNODE": handle_r05}


def run(state: dict) -> dict:
    rid = state.get("current_route")
    route = next((r for r in load_routes() if r["id"] == rid), None)
    if route is None:
        return {"route": rid, "status": "error",
                "note": f"ROUTE_POOL.yaml 里没有路线 {rid!r}（state 与路线池不一致）"}
    if rid in LOCAL_HANDLERS:
        try:
            return LOCAL_HANDLERS[rid](state)
        except Exception as exc:  # 本地执行失败 → 如实报错，不伪造
            return {"route": rid, "status": "error", "note": f"本地执行失败：{type(exc).__name__}: {exc}"}
    return handle_remote(route, state)


def plan(state: dict) -> dict:
    """只描述「当前路线会怎么做」，不执行。"""
    rid = state.get("current_route")
    route = next((r for r in load_routes() if r["id"] == rid), None)
    if route is None:
        return {"route": rid, "status": "error", "note": f"路线池里没有 {rid!r}"}
    if rid in LOCAL_HANDLERS:
        return {"route": rid, "status": "local",
                "note": f"本地可执行：{route.get('method')}",
                "will_run": ("后处理 → 官方脚本 eval_glas_official.py（R5）→ 产出 testB_objDice"
                             if rid == "R04_NNUNetImproved" else
                             "官方脚本 eval_glas_official.py --pred_kind label → 锚点值（R10 不进 best）")}
    res_file = RESULTS_DIR / f"{rid}.json"
    return {"route": rid, "status": "needs_remote" if not res_file.exists() else "resumable",
            "note": f"需服务器：{route.get('method')}",
            "expected_result_file": f"autorun/results/{rid}.json",
            "handoff_preview": f"autorun/handoff/{rid}.json（--run 时生成）"}


def main() -> int:
    as_json = "--json" in sys.argv
    state = load_json(STATE)
    result = plan(state) if "--plan" in sys.argv else run(state)
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"[{result['status']}] {result['route']}")
        for k in ("detail", "note"):
            if result.get(k):
                print(f"  {k}: {result[k]}")
        if result.get("metric"):
            m = result["metric"]
            print(f"  metric: {m['name']}={m['value']}（{m['subset']} / {m['convention']}）")
        if result.get("handoff"):
            print(f"  派发单：autorun/handoff/{result['route']}.json")
    return 0 if result["status"] != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
