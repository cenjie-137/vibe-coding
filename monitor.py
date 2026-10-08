# -*- coding: utf-8 -*-
"""本地训练进度监视器：实时拉取远端 nnU-Net 训练日志，打印每个 fold 的 Epoch / Dice / Loss / GPU。

用法（双击 查看进度.bat 或命令行）:
    python monitor.py               # 打印一次快照
    python monitor.py --watch 60    # 每 60 秒刷新一次
"""
import re
import sys
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")  # 屏蔽旧版 paramiko/cryptography 的 deprecation 警告

import paramiko

KEY = r"c:\Users\32201\Desktop\T\vibe coding\DRIVE\.ssh_keys\id_ed25519"
HOST = "connect.cqa1.seetacloud.com"
PORT = 14531
USER = "root"
LOG_DIR = "/root/autodl-tmp"
RESULTS = "/root/autodl-tmp/nnUNet/results"
DATASETS = {"drive": ("071", "DRIVE"), "glas": ("070", "GLAS")}


def run(cli, cmd, timeout=60):
    _, out, err = cli.exec_command(cmd, timeout=timeout)
    return out.read().decode(errors="replace"), err.read().decode(errors="replace")


def parse_log(text):
    ep = re.findall(r"Epoch (\d+)", text)
    dice = re.findall(r"best EMA pseudo Dice: ([\d.]+)", text)
    tr = re.findall(r"train_loss ([-]?[\d.]+)", text)
    va = re.findall(r"val_loss ([-]?[\d.]+)", text)
    return {
        "epoch": int(ep[-1]) if ep else None,
        "best_dice": float(dice[-1]) if dice else None,
        "train_loss": float(tr[-1]) if tr else None,
        "val_loss": float(va[-1]) if va else None,
    }


def snapshot():
    key = paramiko.Ed25519Key.from_private_key_file(KEY)
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, pkey=key, timeout=30)

    print("=" * 70)
    print(f"训练进度快照  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    rows = []
    for task, (did, tname) in DATASETS.items():
        base = f"{RESULTS}/Dataset{did}_{tname}/nnUNetTrainer__nnUNetPlans__2d"
        for f in range(5):
            log = f"{LOG_DIR}/train_{task}_fold{f}.log"
            o, _ = run(cli, f"[ -f {log} ] && tail -n 400 {log}")
            info = parse_log(o) if o.strip() else {}
            done, _ = run(cli, f"[ -f {base}/fold_{f}/validation/summary.json ] && echo DONE || echo -")
            done = done.strip()

            # 若该 fold 已完成，取 summary Dice
            if done == "DONE" and info.get("best_dice") is None:
                s, _ = run(cli, f"cat {base}/fold_{f}/validation/summary.json")
                m = re.search(r'"foreground_mean"\s*:\s*\{[^}]*?"Dice"\s*:\s*([\d.]+)', s)
                if m:
                    info["best_dice"] = float(m.group(1))
            rows.append((task, f, info, done))

    if not any(r[2] for r in rows):
        print("(暂无训练日志)")
    else:
        print(f"{'任务':<6}{'fold':<6}{'进度':<10}{'best Dice':<12}{'train_loss':<12}{'val_loss':<12}{'状态'}")
        print("-" * 70)
        for task, f, info, done in rows:
            if not info:
                print(f"{task:<6}{f:<6}{'-':<10}{'-':<12}{'-':<12}{'-':<12}{'未启动'}")
                continue
            ep = f"Ep {info['epoch']}" if info.get("epoch") is not None else "-"
            d = f"{info['best_dice']:.4f}" if info.get("best_dice") is not None else "-"
            t = f"{info['train_loss']:.4f}" if info.get("train_loss") is not None else "-"
            v = f"{info['val_loss']:.4f}" if info.get("val_loss") is not None else "-"
            print(f"{task:<6}{f:<6}{ep:<10}{d:<12}{t:<12}{v:<12}{done}")

    print("-" * 70)
    o, _ = run(cli, "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader")
    used, total, util = (o.strip().split(", ") + ["?", "?", "?"])[:3] if o.strip() else ("?", "?", "?")
    print(f"GPU: 显存 {used}/{total}  利用率 {util}")
    cli.close()


def main():
    watch = 0
    if "--watch" in sys.argv:
        try:
            watch = int(sys.argv[sys.argv.index("--watch") + 1])
        except (IndexError, ValueError):
            watch = 60
    while True:
        try:
            snapshot()
        except Exception as e:
            print(f"[连接失败] {e}")
        if not watch:
            break
        print(f"\n(每 {watch} 秒刷新，Ctrl+C 退出)")
        time.sleep(watch)


if __name__ == "__main__":
    main()