# -*- coding: utf-8 -*-
"""datasets.yaml → policy.json 的编译器（人运行，Agent 不得运行）。

为什么要有这个脚本？
  手抄十几个数据集的阈值到 policy.json，迟早抄错一个数字 —— 而阈值是判定器的唯一依据，
  抄错就等于判定错。所以让机器去抄：**目录是手写的，policy.json 是算出来的**。

顺带把两条红线焊死在自己的动作里（自己也得守规矩）：
  R6 破坏性操作须备份：覆盖 policy.json 前，先备份到 autorun/backups/
  R8 留痕强制：每次写入都往 run_log.jsonl 追加一条记录

用法：
    python build_policy.py --check    # 只校验 + 打印差异，不写盘（默认）
    python build_policy.py --write    # 校验通过才写盘（先备份、后写入、再留痕）

注意：本脚本改的是 policy.json —— 那是受保护文件（红线 R7）。
      **只有人能跑它**，Agent 在自主运行期间不得执行。
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml

from ledger import append_run_log, now_iso

HERE = Path(__file__).resolve().parent
DATASETS_YAML = HERE / "datasets.yaml"
POLICY_JSON = HERE / "policy.json"
BACKUP_DIR = HERE / "backups"

ACTIVE = "active"
PREPARING = "preparing"
PLANNED = "planned"
STATUSES = (ACTIVE, PREPARING, PLANNED)


# ── 读取 ──────────────────────────────────────────────────────────
def load_catalog() -> dict:
    return yaml.safe_load(DATASETS_YAML.read_text(encoding="utf-8"))


def load_policy() -> dict:
    return json.loads(POLICY_JSON.read_text(encoding="utf-8"))


# ── 校验：进 policy.json 之前，先把脏数据挡在门外 ──────────────────
def validate(cat: dict) -> list[str]:
    errs: list[str] = []
    conv_ids = set(cat.get("conventions") or {})
    if not conv_ids:
        errs.append("conventions 注册表为空 —— 红线 R1 需要受控口径词表")
    seen: set[str] = set()

    for d in cat.get("datasets") or []:
        t = d.get("task") or "<无名>"
        if t in seen:
            errs.append(f"{t}: task 名重复（task 是 targets 的键，必须唯一）")
        seen.add(t)

        status = d.get("status")
        if status not in STATUSES:
            errs.append(f"{t}: status 必须是 {STATUSES} 之一，实际 {status!r}")

        # active 的数据集是"要真跑的"，字段必须齐
        if status == ACTIVE:
            for k in ("metric", "subset", "convention"):
                if not d.get(k):
                    errs.append(f"{t}: active 必须填 {k}（否则无法判定）")
            if d.get("convention") and d["convention"] not in conv_ids:
                errs.append(f"{t}: convention {d['convention']!r} 不在注册表 —— 红线 R1")
            if d.get("l1") is None:
                errs.append(f"{t}: active 但 l1 为空 —— 没有保底线，判定器无从下手")

        # R2：任何数字都必须带来源 + 已核实
        for lvl in ("l1", "l2"):
            v = d.get(lvl)
            if v is None:
                continue
            if not str(d.get(f"{lvl}_source") or "").strip():
                errs.append(f"{t}: {lvl}={v} 却没有 {lvl}_source —— 红线 R2")
            if not d.get("verified"):
                errs.append(f"{t}: {lvl}={v} 但 verified=false —— 红线 R2")

    return errs


# ── 编译 ──────────────────────────────────────────────────────────
def build_targets(cat: dict) -> dict:
    """只把 active 的数据集编译进 targets —— 判定器只认这个段。"""
    out = {}
    for d in cat["datasets"]:
        if d.get("status") != ACTIVE:
            continue
        out[d["task"]] = {
            "metric": d["metric"],
            "subset": d["subset"],
            "convention": d["convention"],
            "eval_script": d.get("eval_script"),
            "l1": d["l1"],
            "l2": d.get("l2"),
            "l1_note": _note(d, "l1"),
            "l2_note": _note(d, "l2"),
        }
    return out


def _note(d: dict, lvl: str) -> str:
    if d.get(lvl) is None:
        return "未核实 / 未定义"
    src = str(d.get(f"{lvl}_source") or "").strip()
    base = d.get("note") or ""
    return f"{base} ｜ 来源：{src}" if base else f"来源：{src}"


def build_catalog(cat: dict) -> dict:
    """全部数据集（含 planned）的清单 —— 给人看的全貌，判定器不读。"""
    out = {}
    for d in cat["datasets"]:
        out[d["task"]] = {
            "status": d["status"],
            "dim": d.get("dim"),
            "modality": d.get("modality"),
            "classes": d.get("classes"),
            "metric": d.get("metric"),
            "subset": d.get("subset"),
            "convention": d.get("convention"),
            "eval_script": d.get("eval_script"),
            "verified": bool(d.get("verified")),
            "note": d.get("note"),
        }
    return out


def build_allowlist(cat: dict) -> list[str]:
    """R5 评估锁定：白名单只放 active 数据集已确认的官方协议脚本。"""
    out = []
    for d in cat["datasets"]:
        if d.get("status") == ACTIVE and d.get("eval_script"):
            if d["eval_script"] not in out:
                out.append(d["eval_script"])
    return sorted(out)


def bump_version(v: str) -> str:
    parts = str(v).split(".")
    try:
        major, minor = int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return "1.1"
    return f"{major}.{minor + 1}"


def compile_policy(cat: dict, policy: dict) -> dict:
    new = json.loads(json.dumps(policy))  # 深拷贝，别污染入参
    new["targets"] = build_targets(cat)
    new["catalog"] = build_catalog(cat)
    new["_catalog_note"] = (
        "全部数据集清单（含 planned），由 datasets.yaml 编译而来；"
        "判定器只读 targets，不读 catalog。"
    )
    new["eval"] = {**new.get("eval", {}), "allowlist": build_allowlist(cat)}
    new["version"] = bump_version(policy.get("version", "1.0"))
    new["_compiled_from"] = "autorun/datasets.yaml"
    new["_compiled_at"] = now_iso()
    return new


# ── 差异比较 ──────────────────────────────────────────────────────
def diff_sections(cur: dict, new: dict) -> list[str]:
    lines = []
    for key in ("targets", "catalog", "eval"):
        a = json.dumps(cur.get(key), ensure_ascii=False, sort_keys=True)
        b = json.dumps(new.get(key), ensure_ascii=False, sort_keys=True)
        if a != b:
            lines.append(f"  ~ {key} 有变化")
        else:
            lines.append(f"  = {key} 无变化")
    if cur.get("version") != new.get("version"):
        lines.append(f"  ~ version: {cur.get('version')} → {new.get('version')}")
    return lines


def _snapshot(policy: dict) -> dict:
    return {k: policy.get(k) for k in ("targets", "catalog", "eval")}


# ── 主流程 ────────────────────────────────────────────────────────
def main() -> int:
    write = "--write" in sys.argv
    cat = load_catalog()
    policy = load_policy()

    errs = validate(cat)
    if errs:
        print("✗ 目录校验未通过，已拒绝编译：")
        for e in errs:
            print("   ·", e)
        return 1

    n_active = sum(1 for d in cat["datasets"] if d["status"] == ACTIVE)
    n_prep = sum(1 for d in cat["datasets"] if d["status"] == PREPARING)
    n_all = len(cat["datasets"])
    print(f"目录校验通过：共 {n_all} 个数据集 —— active {n_active}（进判定）、"
          f"preparing {n_prep}（基线待测）、planned {n_all - n_active - n_prep}（待核实）")

    new = compile_policy(cat, policy)
    print("\n与当前 policy.json 的差异：")
    for line in diff_sections(policy, new):
        print(line)

    if not write:
        print("\n（--check 模式，未写盘。确认无误后跑 --write）")
        return 0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup = BACKUP_DIR / f"policy_{ts}.json"
    shutil.copy2(POLICY_JSON, backup)                       # R6：先备份
    POLICY_JSON.write_text(
        json.dumps(new, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    append_run_log(dict(                                    # R8：再留痕
        run_id=f"build-policy-{ts}",
        task="ALL",
        phase="governance",
        action=f"build_policy.py --write：targets {len(policy.get('targets', {}))} → {len(new['targets'])}，"
               f"catalog {n_all}，allowlist {new['eval']['allowlist']}",
        actor="human",
        destructive=True,
        backup_path=str(backup.relative_to(HERE.parent)).replace("\\", "/"),
        artifact=str(POLICY_JSON.relative_to(HERE.parent)).replace("\\", "/"),
    ))

    print(f"\n✓ 已写入 {POLICY_JSON.name}")
    print(f"  version {policy.get('version')} → {new['version']}")
    print(f"  备份：{backup}")
    print(f"  留痕：autorun/run_log.jsonl")
    print("\n⚠ 手工待办（红线 R7：只有人能改规则文件）：")
    print("  1. 同步更新 AUTONOMY_POLICY.md 第 1 节的目标表（新增数据集）")
    print("  2. git commit（message 写明：新增了哪些数据集、为什么）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
