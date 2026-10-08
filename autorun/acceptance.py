# -*- coding: utf-8 -*-
"""第一步验收跑批 —— 一条命令跑完所有组件的自检与审计。

为什么需要它？
  论文里说"我们的判定器是确定性的、账本是可审计的"，就得有**一键复现的证据**。
  审稿人（或三个月后的你自己）只要跑 `python acceptance.py`，
  就能看到每个组件自己的对抗用例过了几条、审计有没有违规。

用法：
    python acceptance.py            # 跑全部，打印表格，写 acceptance_report.json
    python acceptance.py --html     # 额外生成 HTML 验收报告
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPORT_JSON = HERE / "acceptance_report.json"

# (显示名, 命令参数, 该跑出什么)
SELFTESTS = [
    ("判定器 judge", ["judge.py", "--selftest"]),
    ("溯源账本 ledger", ["ledger.py", "--selftest"]),
    ("红线检查 redlines", ["redlines.py", "--selftest"]),
    ("重放验证 replay", ["replay.py", "--selftest"]),
]
AUDITS = [
    ("红线审计（当前产物）", ["redlines.py", "--audit"]),
    ("策略编译一致性", ["build_policy.py", "--check"]),
    ("证据包重放", ["replay.py", "--trace", "episodes/EP01_GLaS_demo/run_log.jsonl"]),
]

PASSED_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s+passed")


def run(args: list[str]) -> tuple[int, str]:
    p = subprocess.run(
        [sys.executable, *args], cwd=HERE,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def main() -> int:
    results = []

    for name, args in SELFTESTS:
        rc, out = run(args)
        m = PASSED_RE.search(out)
        passed, total = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        results.append({
            "kind": "selftest", "name": name, "cmd": " ".join(args),
            "ok": rc == 0 and total > 0 and passed == total,
            "detail": f"{passed}/{total} 用例通过" if m else "未解析到结果",
        })

    for name, args in AUDITS:
        rc, out = run(args)
        results.append({
            "kind": "audit", "name": name, "cmd": " ".join(args),
            "ok": rc == 0,
            "detail": _summarize(out),
        })

    rc, out = run(["ledger.py", "--scorecard"])
    try:
        scorecard = json.loads(out)
    except json.JSONDecodeError:
        scorecard = {}
        results.append({"kind": "audit", "name": "可信记分卡", "cmd": "ledger.py --scorecard",
                        "ok": False, "detail": "无法解析"})
    else:
        results.append({"kind": "audit", "name": "可信记分卡", "cmd": "ledger.py --scorecard",
                        "ok": True,
                        "detail": f"来源覆盖率 {scorecard.get('source_coverage', 0):.1%}，"
                                  f"{scorecard.get('n_claims', 0)} 条数字 / {scorecard.get('n_actions', 0)} 个动作"})

    n_ok = sum(r["ok"] for r in results)
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "policy_version": json.loads((HERE / "policy.json").read_text(encoding="utf-8")).get("version"),
        "n_checks": len(results), "n_passed": n_ok, "all_passed": n_ok == len(results),
        "results": results,
        "scorecard": scorecard,
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"── 第一步验收：{n_ok}/{len(results)} 项通过 ──")
    for r in results:
        tag = "自检" if r["kind"] == "selftest" else "审计"
        print(f"  [{'PASS' if r['ok'] else 'FAIL'}] {tag} {r['name']:22s} {r['detail']}")
    print(f"\n报告已写入 {REPORT_JSON.relative_to(ROOT)}")

    if "--html" in sys.argv:
        html = build_html(report)
        out = HERE / "acceptance_report.html"
        out.write_text(html, encoding="utf-8")
        print(f"HTML 报告已写入 {out.relative_to(ROOT)}")

    return 0 if report["all_passed"] else 1


def _summarize(out: str) -> str:
    for line in out.splitlines():
        if "项检查" in line or "违规" in line:
            return line.strip()
    return (out.strip().splitlines() or ["(无输出)"])[-1]


def build_html(report: dict) -> str:
    rows = "\n".join(
        f'<tr><td>{"自检" if r["kind"] == "selftest" else "审计"}</td>'
        f'<td>{r["name"]}</td>'
        f'<td class="{"ok" if r["ok"] else "bad"}">{"通过" if r["ok"] else "未通过"}</td>'
        f'<td class="mono">{r["cmd"]}</td><td>{r["detail"]}</td></tr>'
        for r in report["results"]
    )
    sc = report.get("scorecard", {})
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>治理层第一步验收报告</title>
<style>
 body{{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;margin:0;padding:32px;
      background:#f6f7f9;color:#1a1d21;line-height:1.6}}
 .wrap{{max-width:1000px;margin:0 auto}}
 h1{{font-size:24px;margin:0 0 4px}} .sub{{color:#6b7280;font-size:13px;margin-bottom:24px}}
 .kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px}}
 .kpi{{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px 16px}}
 .kpi .v{{font-size:22px;font-weight:700}} .kpi .l{{font-size:12px;color:#6b7280}}
 table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid #e5e7eb;
        border-radius:10px;overflow:hidden;font-size:13px}}
 th,td{{padding:9px 12px;text-align:left;border-bottom:1px solid #f0f1f3}}
 th{{background:#fafbfc;font-weight:600;font-size:12px;color:#4b5563}}
 tr:last-child td{{border-bottom:none}}
 .ok{{color:#0f7b3f;font-weight:600}} .bad{{color:#b91c1c;font-weight:600}}
 .mono{{font-family:ui-monospace,Consolas,monospace;font-size:12px;color:#4b5563}}
 .note{{margin-top:20px;font-size:12px;color:#6b7280}}
</style></head><body><div class="wrap">
<h1>治理层第一步验收报告</h1>
<div class="sub">生成时间 {report["generated_at"]} ｜ policy 版本 {report.get("policy_version")} ｜
命令 <span class="mono">python autorun/acceptance.py</span></div>
<div class="kpis">
 <div class="kpi"><div class="v">{report["n_passed"]}/{report["n_checks"]}</div><div class="l">检查项通过</div></div>
 <div class="kpi"><div class="v">{sc.get("n_claims", "-")}</div><div class="l">账本数字条数</div></div>
 <div class="kpi"><div class="v">{sc.get("source_coverage", 0):.1%}</div><div class="l">来源覆盖率</div></div>
 <div class="kpi"><div class="v">{sc.get("destructive_without_backup", "-")}</div><div class="l">无备份的破坏性操作</div></div>
</div>
<table><thead><tr><th>类型</th><th>组件</th><th>结果</th><th>命令</th><th>说明</th></tr></thead>
<tbody>{rows}</tbody></table>
<div class="note">本报告由 <span class="mono">acceptance.py</span> 自动生成，数据来源
 <span class="mono">autorun/acceptance_report.json</span>。所有数字均可在本地复跑得到相同结果。</div>
</div></body></html>"""


if __name__ == "__main__":
    sys.exit(main())
