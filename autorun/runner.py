# -*- coding: utf-8 -*-
"""runner.py — 路线执行器（Phase 2/3）

一句话：**给 state，执行「当前路线」的一轮，产出可回填的结果。**
它只负责「跑」和「产数」；判定归 judge.py，状态推进归 loop.py。

两种执行环境（这是本项目的现实约束：训练在服务器、后处理在本地）：

  · 本地可跑（后处理 / 评估类）→ 真跑，产出真实指标。
  · 需服务器（训练类）          → **不伪造数字**，产出一份 handoff（要什么、
                                  结果写成什么格式、怎么续跑），等服务器回填结果文件。

按 task 分派（Phase 3 新增）：
  `TASKS` 注册表把「评估脚本 / GT 目录 / 本地素材 / 判定指标默认值 / 路线池文件」
  按数据集收拢，`run(state)` 读 `state["task"]` 选对应配置与本地处理器。
  GLaS 的处理器逻辑与 Phase 2 **完全一致**，只是把模块级常量改成从 `TASKS["GLaS"]` 取；
  PUMA 目前无本地素材（数据/权重都在服务器），故无本地处理器 → 一律走 needs_remote 派发。

设计红线（自己也得守规矩）：
  R5 评估锁定 —— 指标一律由 allowlist 里的官方脚本算出：本脚本用 subprocess 调
                 `eval_<task>_official.py` 并解析它的 stdout，**runner 自己不算任何指标**。
  R2 来源强制 —— 每个数字都带 source（脚本 + 命令）；服务器结果文件缺来源一律拒收。
  不伪造     —— 跑不了就 `needs_remote`，绝不编一个数字出来（这是可信度的地基）。

用法：
    python runner.py --plan [--task PUMA]     # 打印当前路线要做什么（不执行）
    python runner.py --run  [--task PUMA]     # 执行当前路线的一轮
    python runner.py --run --json             # 机器可读输出（loop.py 用）
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
for _p in (str(ROOT), str(HERE)):   # ROOT：复用 tune_glas_pp / eval_*_official；HERE：ledger 等
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml  # noqa: E402

RESULTS_DIR = HERE / "results"                   # 服务器回填的结果文件放这里
HANDOFF_DIR = HERE / "handoff"                   # 训练类路线的派发单放这里


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def slug(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()


# ── R5：官方脚本输出的解析（每个 task 一个；runner 只解析，不算指标）────
GLAS_EVAL_LINE = re.compile(
    r"^\s*(test[AB])\s+n=\s*(\d+)\s+ObjDice=([\d.]+)\s+ObjF1=([\d.]+)\s+ObjHD=\s*([\d.]+)"
)
PUMA_EVAL_LINE = re.compile(
    r"^\s*(\S+)\s+n=\s*(\d+)\s+microDice=([\d.]+)\s+macroDice=([\d.]+)"
)


def _parse_glas(stdout: str) -> dict:
    """GLaS：`  testB      n= 20  ObjDice=0.8184  ObjF1=0.6113  ObjHD= 118.85`"""
    out: dict[str, dict] = {}
    for line in (stdout or "").splitlines():
        m = GLAS_EVAL_LINE.match(line)
        if m:
            out[m.group(1)] = {"n": int(m.group(2)), "objDice": float(m.group(3)),
                               "objF1": float(m.group(4)), "objHD": float(m.group(5))}
    if "testB" not in out:
        raise RuntimeError(f"没能从评估输出解析到 testB：\n{stdout}")
    return out


def _parse_puma(stdout: str) -> dict:
    """PUMA：`  val        n=205  microDice=0.6762  macroDice=0.6543`"""
    out: dict[str, dict] = {}
    for line in (stdout or "").splitlines():
        m = PUMA_EVAL_LINE.match(line)
        if m:
            out[m.group(1)] = {"n": int(m.group(2)), "microDice": float(m.group(3)),
                               "macroDice": float(m.group(4))}
    if not out:
        raise RuntimeError(f"没能从评估输出解析到 microDice 行：\n{stdout}")
    return out


# ── 本地素材 / 路线池：按 task 收拢 ───────────────────────────────
# 注：PUMA 的 gt_dir / pred_dir 指向「服务器产出后同步回本地」的位置；本地暂无 → 本地评估会如实报错。
TASKS: dict[str, dict] = {
    "GLaS": {
        "eval_script": ROOT / "eval_glas_official.py",
        "gt_dir": ROOT / "data" / "glas" / "masks",
        "metadata": ROOT / "data" / "glas" / "metadata.csv",
        "pred_dir": ROOT / "glas_test_pred",
        "pp_root": ROOT / "glas_test_pred_pp",
        "unode_label": ROOT / "glas_unode_pred_label",
        "parse": _parse_glas,
        "metric": "testB_objDice", "subset": "testB", "convention": "official_object_level",
        "route_pool": ROOT / "ROUTE_POOL.yaml",
    },
    "PUMA": {
        "eval_script": ROOT / "eval_puma_official.py",
        "gt_dir": ROOT / "data" / "PUMA" / "masks",
        "metadata": None,
        "pred_dir": ROOT / "puma_pred",
        "parse": _parse_puma,
        "metric": "micro_dice", "subset": "val", "convention": "official_micro_dice",
        "route_pool": ROOT / "ROUTE_POOL_PUMA.yaml",
    },
}


def task_cfg(task: str) -> dict:
    cfg = TASKS.get(task)
    if cfg is None:
        raise KeyError(f"未知 task {task!r}（TASKS 里没有；新增数据集需在此登记评估脚本与素材）")
    return cfg


def load_routes(task: str = "GLaS") -> list[dict]:
    """按 task 读路线池（顺序 = 尝试优先级）。

    GLaS 用受保护的 ROUTE_POOL.yaml；其余 task 用 `ROUTE_POOL_<task>.yaml`
    （新增数据集自己的池，不改动受保护文件 —— 红线 R7）。
    """
    cfg = TASKS.get(task) or {}
    pool_file = Path(cfg.get("route_pool") or (ROOT / f"ROUTE_POOL_{task}.yaml"))
    if not pool_file.exists():
        return []
    pool = yaml.safe_load(pool_file.read_text(encoding="utf-8")) or {}
    return pool.get("routes") or []


# ── R5：评估只走官方脚本，runner 不算指标 ─────────────────────────
def run_official_eval(cfg: dict, pred_dir: Path, pred_kind: str = "binary") -> dict:
    """调该 task 的官方评估脚本并解析 stdout。返回脚本给出的原始结构。"""
    cmd = [sys.executable, str(cfg["eval_script"]),
           "--gt_dir", str(cfg["gt_dir"]), "--pred_dir", str(pred_dir)]
    if cfg.get("metadata"):
        cmd += ["--metadata", str(cfg["metadata"])]
    if pred_kind == "label":
        cmd += ["--pred_kind", "label"]
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"官方评估脚本失败（rc={p.returncode}）：\n{p.stdout}\n{p.stderr}")
    return cfg["parse"](p.stdout)


def _cmd_str(cfg: dict, pred_dir: Path, pred_kind: str = "binary") -> str:
    rel = pred_dir.relative_to(ROOT).as_posix() if str(pred_dir).startswith(str(ROOT)) else str(pred_dir)
    s = (f"python {Path(cfg['eval_script']).name} --gt_dir "
         f"{cfg['gt_dir'].relative_to(ROOT).as_posix()} --pred_dir {rel}")
    if cfg.get("metadata"):
        s += f" --metadata {cfg['metadata'].relative_to(ROOT).as_posix()}"
    return s + (" --pred_kind label" if pred_kind == "label" else "")


# ── 本地处理器（GLaS）─────────────────────────────────────────────
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

    cfg = task_cfg("GLaS")
    configs = tune_glas_pp.CONFIGS
    idx = len((state.get("small_rounds") or {}).get("history") or [])
    if idx >= len(configs):
        return {"route": "R04_NNUNetImproved", "status": "done", "metric": None,
                "note": f"R04 的 {len(configs)} 个后处理配置已试完（idx={idx}），无更多可试。"}

    label, fn, kw = configs[idx]
    if fn is None:
        pred_dir, detail = cfg["pred_dir"], "raw（无后处理）"
    else:
        pred_dir = cfg["pp_root"] / slug(label)
        pred_dir.mkdir(parents=True, exist_ok=True)
        if not any(pred_dir.glob("*.png")):
            for img in sorted(cfg["pred_dir"].glob("*.png")):
                b = tune_glas_pp.load_mask_binary(img)
                out = fn(b, **kw)
                Image.fromarray((out.astype(np.uint8)) * 255).save(pred_dir / img.name)
        detail = f"后处理 {label}"

    ev = run_official_eval(cfg, pred_dir)
    return {
        "route": "R04_NNUNetImproved", "status": "done",
        "metric": {"name": "testB_objDice", "value": ev["testB"]["objDice"],
                   "subset": "testB", "convention": "official_object_level"},
        "extra": {"testA_objDice": ev["testA"]["objDice"],
                  "testB_objF1": ev["testB"]["objF1"]},
        "artifacts": [pred_dir.relative_to(ROOT).as_posix()],
        "source_url": f"{_cmd_str(cfg, pred_dir)}（R5 allowlist 官方脚本）",
        "detail": f"config[{idx}]={label}；{detail}；testA ObjDice={ev['testA']['objDice']:.4f}",
    }


def handle_r05(state: dict) -> dict:
    """R05：U-Node 复现（锚点）。产物按红线 R10 写 state.anchor，不进 best。"""
    cfg = task_cfg("GLaS")
    pred_dir = cfg["unode_label"]
    ev = run_official_eval(cfg, pred_dir, pred_kind="label")
    return {
        "route": "R05_UNODE", "status": "done", "target": "anchor",
        "metric": {"name": "testB_objDice", "value": ev["testB"]["objDice"],
                   "subset": "testB", "convention": "official_object_level"},
        "extra": {"testA_objDice": ev["testA"]["objDice"],
                  "testB_objF1": ev["testB"]["objF1"]},
        "artifacts": [pred_dir.relative_to(ROOT).as_posix()],
        "source_url": f"{_cmd_str(cfg, pred_dir, 'label')}（R5 allowlist 官方脚本）",
        "detail": "U-Node 官方权重预测，实例标签形态评估（与论文 0.842 同口径）",
    }


# ── 训练类路线：派发到服务器（本地不伪造数字）──────────────────────
def handle_remote(route: dict, state: dict, cfg: dict) -> dict:
    rid = route["id"]
    res_file = RESULTS_DIR / f"{rid}.json"

    # 服务器已回填结果 → 采纳（但必须先过 R2：无来源一律拒收）
    if res_file.exists():
        r = load_json(res_file)
        if not str(r.get("source_url") or "").strip():
            return {"route": rid, "status": "error",
                    "note": f"{res_file.name} 缺 source_url（红线 R2），拒绝采纳 —— 宁可不认，不认脏数。"}
        return {"route": rid, "status": "done",
                "metric": {"name": r.get("metric", cfg["metric"]), "value": r["value"],
                           "subset": r.get("subset", cfg["subset"]),
                           "convention": r.get("convention", cfg["convention"])},
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
            "route": rid, "metric": cfg["metric"], "value": "<float>",
            "subset": cfg["subset"], "convention": cfg["convention"],
            "source_url": f"<必须：结果由 {Path(cfg['eval_script']).name} 产出，填脚本+命令或日志链接>",
            "evidence": "<权重目录 / 推理输出目录 / 训练日志>",
        },
        "how_to_resume": f"服务器产出上述结果文件后，重跑 `python autorun/loop.py --task "
                         f"{state.get('task')}`，loop 会自动读取并回填，无需人工改 state。",
        "note": "本文件是**派发单**，不是结果。runner 不会替服务器编一个数字。",
    }
    HANDOFF_DIR.mkdir(parents=True, exist_ok=True)
    (HANDOFF_DIR / f"{rid}.json").write_text(
        json.dumps(handoff, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"route": rid, "status": "needs_remote", "handoff": handoff,
            "note": f"{rid} 需服务器执行；已产出派发单 autorun/handoff/{rid}.json"}


# ── 调度 ──────────────────────────────────────────────────────────
LOCAL_HANDLERS: dict[str, dict] = {
    "GLaS": {"R04_NNUNetImproved": handle_r04, "R05_UNODE": handle_r05},
    "PUMA": {},   # PUMA 无本地素材 → 全部走 needs_remote 派发
}


def run(state: dict) -> dict:
    task = state.get("task") or "GLaS"
    try:
        cfg = task_cfg(task)
    except KeyError as exc:
        return {"route": state.get("current_route"), "status": "error", "note": str(exc)}

    rid = state.get("current_route")
    route = next((r for r in load_routes(task) if r["id"] == rid), None)
    if route is None:
        return {"route": rid, "status": "error",
                "note": f"{task} 的路线池里没有路线 {rid!r}（state 与路线池不一致）"}
    handler = LOCAL_HANDLERS.get(task, {}).get(rid)
    if handler is not None:
        try:
            return handler(state)
        except Exception as exc:  # 本地执行失败 → 如实报错，不伪造
            return {"route": rid, "status": "error", "note": f"本地执行失败：{type(exc).__name__}: {exc}"}
    return handle_remote(route, state, cfg)


def plan(state: dict) -> dict:
    """只描述「当前路线会怎么做」，不执行。"""
    task = state.get("task") or "GLaS"
    try:
        cfg = task_cfg(task)
    except KeyError as exc:
        return {"route": state.get("current_route"), "status": "error", "note": str(exc)}

    rid = state.get("current_route")
    route = next((r for r in load_routes(task) if r["id"] == rid), None)
    if route is None:
        return {"route": rid, "status": "error", "note": f"{task} 路线池里没有 {rid!r}"}
    if rid in LOCAL_HANDLERS.get(task, {}):
        will_run = {
            "R04_NNUNetImproved": f"后处理 → 官方脚本 {Path(cfg['eval_script']).name}（R5）→ 产出 {cfg['metric']}",
            "R05_UNODE": f"官方脚本 {Path(cfg['eval_script']).name} --pred_kind label → 锚点值（R10 不进 best）",
        }.get(rid, "本地可执行")
        return {"route": rid, "status": "local",
                "note": f"本地可执行：{route.get('method')}", "will_run": will_run}
    res_file = RESULTS_DIR / f"{rid}.json"
    return {"route": rid, "status": "needs_remote" if not res_file.exists() else "resumable",
            "note": f"需服务器：{route.get('method')}",
            "expected_result_file": f"autorun/results/{rid}.json",
            "handoff_preview": f"autorun/handoff/{rid}.json（--run 时生成）"}


def _default_state(task: str) -> Path:
    p = HERE / f"state_{task}.json"
    return p if p.exists() else HERE / "state.json"


def main() -> int:
    as_json = "--json" in sys.argv
    task = None
    if "--task" in sys.argv:
        task = sys.argv[sys.argv.index("--task") + 1]
    state_path = Path(sys.argv[sys.argv.index("--state") + 1]) if "--state" in sys.argv \
        else _default_state(task or "GLaS")
    state = load_json(state_path)
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
