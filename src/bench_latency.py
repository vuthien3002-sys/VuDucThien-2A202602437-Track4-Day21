"""Bonus B3: đo latency bài kiểm tra calibration trên CPU, đúng cách (bỏ lần chạy đầu, >= 20 lần, p50/p95).

Hai mức đo cho mỗi frame (dữ liệu đã nạp sẵn vào RAM, không tính đọc file):
  * projection : velo_to_cam + P2 + lọc FOV cho toàn bộ point cloud (project_velo_to_image)
  * qa_check   : toàn bộ bài kiểm tra 1 frame = chọn điểm trong 3D box + chiếu + đếm điểm trong 2D box

Chạy từ gốc repo:  python -m src.bench_latency
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.exp_calib_sweep import CLASS_GROUP, measure, prepare_frame
from starter.datasets import load_frame
from starter.projection import project_velo_to_image


def hardware_info() -> str:
    """Tên CPU + RAM. Windows: hỏi PowerShell (Get-CimInstance); máy khác: dùng platform."""
    cpu, ram = platform.processor() or "unknown", "unknown"
    if platform.system() == "Windows":
        try:
            ps = ("(Get-CimInstance Win32_Processor).Name; "
                  "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)")
            out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True,
                                 text=True, timeout=30).stdout.split("\n")
            cpu, ram = out[0].strip() or cpu, f"{out[1].strip()} GB"
        except (OSError, subprocess.SubprocessError, IndexError):
            pass
    return (f"CPU: {cpu} ({os.cpu_count()} logical cores), RAM: {ram}, GPU: không dùng\n"
            f"OS: {platform.platform()}, Python {platform.python_version()}, numpy {np.__version__}")


def bench(fn, runs: int) -> np.ndarray:
    times = []
    for _ in range(runs + 1):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return np.array(times[1:]) * 1000          # bỏ lần chạy đầu (khởi tạo, cache), đổi sang ms


def main() -> None:
    ap = argparse.ArgumentParser(description="B3: đo latency p50/p95 của bài kiểm tra calibration trên CPU",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--runs", type=int, default=50, help="số lần đo (chưa tính lần chạy đầu bị bỏ)")
    ap.add_argument("--out", default="results/latency_qa.csv", help="CSV: mỗi dòng 1 lần chạy")
    args = ap.parse_args()

    cases = [("data/kitti_mini", "000011"), ("data/nuscenes_mini_subset", "scene-0103_010")]
    rows = []
    for root, frame in cases:
        fr = load_frame(root, frame)
        stages = {
            "projection": lambda: project_velo_to_image(fr["points"], fr["calib"], fr["image"].shape),
            "qa_check": lambda: measure(prepare_frame(fr, set(CLASS_GROUP)), fr["calib"], fr["image"].shape),
        }
        for stage, fn in stages.items():
            for i, ms in enumerate(bench(fn, args.runs)):
                rows.append({"dataset": Path(root).name, "frame": frame, "n_points": len(fr["points"]),
                             "stage": stage, "run": i + 1, "ms": round(float(ms), 3)})
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    hw = hardware_info()
    Path(args.out).with_name("latency_hardware.txt").write_text(hw + "\n", encoding="utf-8")
    summary = df.groupby(["dataset", "frame", "n_points", "stage"])["ms"].agg(
        runs="size", p50=lambda x: round(float(np.percentile(x, 50)), 2),
        p95=lambda x: round(float(np.percentile(x, 95)), 2)).reset_index()
    summary.to_csv(Path(args.out).with_name("latency_summary.csv"), index=False)
    print(hw)
    print(summary.to_string(index=False))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
