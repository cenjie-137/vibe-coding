# -*- coding: utf-8 -*-
"""DRIVE 官方测试集提交包生成器（在 DRIVE 5 折训练完成后运行）。

流程：nnUNetv2_predict(5折集成) -> nii.gz 转二值 PNG(1.png~20.png) -> 打包 zip -> 下载到本地。

官方要求（https://drive.grand-challenge.org/）：
  提交一个 zip，内含 20 张二值 PNG，命名必须为 1.png ~ 20.png（对应 test 集 01~20）。
  评估按测试集逐图 Dice（仅 FOV 内像素），排行榜按平均 Dice 排序。
"""
import re
import ssl
import time
from pathlib import Path

import paramiko

KEY = r"c:\Users\32201\Desktop\T\vibe coding\DRIVE\.ssh_keys\id_ed25519"
HOST = "connect.cqa1.seetacloud.com"
PORT = 14531
USER = "root"

NN_PREDICT = "/root/miniconda3/envs/nnunet/bin/nnUNetv2_predict"
PY = "/root/miniconda3/envs/nnunet/bin/python3.10"
EXPORT = ("export nnUNet_raw=/root/autodl-tmp/nnUNet/raw "
          "nnUNet_preprocessed=/root/autodl-tmp/nnUNet/preprocessed "
          "nnUNet_results=/root/autodl-tmp/nnUNet/results")
RESULTS = "/root/autodl-tmp/nnUNet/results"
TS_DIR = "/root/autodl-tmp/nnUNet/raw/Dataset071_DRIVE/imagesTs"
PRED_DIR = "/root/autodl-tmp/drive_test_pred"
ZIP_REMOTE = "/root/autodl-tmp/drive_submission.zip"
ZIP_LOCAL = Path(r"c:\Users\32201\Desktop\T\vibe coding\DRIVE\DRIVE_nnunet_baseline_submission.zip")

# 远端转换脚本（nii.gz -> 二值 PNG -> zip），写入后执行，避免引号转义问题
CONVERT_SCRIPT = r"""
import re, zipfile
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from PIL import Image

PRED = Path("/root/autodl-tmp/drive_test_pred")
OUT = Path("/root/autodl-tmp/drive_submission")
TS = Path("/root/autodl-tmp/nnUNet/raw/Dataset071_DRIVE/imagesTs")
OUT.mkdir(exist_ok=True)

preds = {}
for p in PRED.glob("*.nii.gz"):
    m = re.search(r"DRIVE_test_(\d+)", p.name)
    if m:
        preds[int(m.group(1))] = p

def input_size(num):
    cand = list(TS.glob("DRIVE_test_%02d_*.png" % num))
    if cand:
        with Image.open(cand[0]) as im:
            return im.size  # (W, H)
    return None

count = 0
for num in sorted(preds):
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(preds[num]))).squeeze()
    mask = (np.asarray(arr) > 0).astype(np.uint8) * 255
    ws = input_size(num)
    if ws and (mask.shape[1], mask.shape[0]) != ws:
        mask = mask.T  # 对齐输入宽高
    Image.fromarray(mask).save(OUT / "%d.png" % num)
    count += 1

with zipfile.ZipFile("/root/autodl-tmp/drive_submission.zip", "w") as z:
    for num in range(1, 21):
        z.write(OUT / "%d.png" % num, arcname="%d.png" % num)
print("CONVERT_DONE", count)
"""


def run(cli, cmd, timeout=120):
    _, out, err = cli.exec_command(cmd, timeout=timeout)
    return out.read().decode(errors="replace"), err.read().decode(errors="replace")


def connect():
    key = paramiko.Ed25519Key.from_private_key_file(KEY)
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, pkey=key, timeout=30)
    return cli


def main():
    cli = connect()

    # 1. 确认 5 折权重已就绪
    missing = []
    for f in range(5):
        ckpt = f"{RESULTS}/Dataset071_DRIVE/nnUNetTrainer__nnUNetPlans__2d/fold_{f}/checkpoint_final.pth"
        o, _ = run(cli, f"[ -f {ckpt} ] && echo OK || echo MISSING")
        if o.strip() != "OK":
            missing.append(f)
    if missing:
        print(f"[跳过] 以下 fold 权重未完成: {missing}，请等训练结束再运行。")
        cli.close()
        return

    # 2. 预测（5 折集成，后台运行并轮询）
    run(cli, f"rm -rf {PRED_DIR} /root/autodl-tmp/predict_drive.log")
    cmd = (f"{EXPORT}; setsid nohup {NN_PREDICT} -i {TS_DIR} -o {PRED_DIR} "
           f"-d 71 -c 2d -f 0 1 2 3 4 > /root/autodl-tmp/predict_drive.log 2>&1 < /dev/null &")
    run(cli, cmd)
    print("[predict] 已启动，等待完成...")

    t0 = time.time()
    while True:
        time.sleep(30)
        o, _ = run(cli, f"ls {PRED_DIR}/*.nii.gz 2>/dev/null | wc -l")
        n = int(o.strip() or 0)
        if n >= 20:
            break
        if time.time() - t0 > 3600:
            print("[predict] 超时，查看 /root/autodl-tmp/predict_drive.log")
            cli.close()
            return
        print(f"  已生成 {n}/20 ...")

    # 3. 写入并执行转换脚本
    sftp = cli.open_sftp()
    with sftp.open("/root/autodl-tmp/conv_drive_preds.py", "w") as f:
        f.write(CONVERT_SCRIPT)
    sftp.close()
    o, e = run(cli, f"{PY} /root/autodl-tmp/conv_drive_preds.py", timeout=600)
    print("[convert]", (o.strip() or "") + (e.strip() or ""))

    # 4. 下载 zip 到本地
    sftp = cli.open_sftp()
    sftp.get(ZIP_REMOTE, str(ZIP_LOCAL))
    sftp.close()
    print("已保存:", ZIP_LOCAL)
    cli.close()


if __name__ == "__main__":
    main()