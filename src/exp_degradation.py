"""Bonus B2: bài kiểm tra calibration (hit_ratio) còn dùng được không khi point cloud bị suy giảm?

2 loại suy giảm × 3 mức (+ mức gốc), dùng starter/perturb.py với seed cố định:
  * random_dropout keep_ratio 1.0 / 0.7 / 0.5 / 0.3   (mưa, bụi, LiDAR rẻ hơn)
  * gaussian_noise sigma 0 / 0.02 / 0.05 / 0.10 m     (nhiễu đo khoảng cách)
Với mỗi mức: đo hit_ratio (mẫu số cố định) ở yaw 0° và 1°, và tỉ lệ frame bị cảnh báo với ngưỡng 95%.
Câu hỏi: suy giảm có làm bài kiểm tra báo nhầm (ở 0°) hoặc bỏ sót (ở 1°) không?

Chạy từ gốc repo:  python -m src.exp_degradation
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.exp_calib_sweep import measure, perturbed_calib, prepare_frame, CLASS_GROUP
from src.plot_results import BLUE, ORANGE, ratio_axis
from starter.datasets import list_frames, load_frame
from starter.perturb import gaussian_noise, random_dropout

DEGRADATIONS = {
    "random_dropout": ("keep_ratio", [1.0, 0.7, 0.5, 0.3], lambda p, v, seed: random_dropout(p, v, seed=seed)),
    "gaussian_noise": ("sigma_m", [0.0, 0.02, 0.05, 0.10], lambda p, v, seed: gaussian_noise(p, v, seed=seed)),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="B2: stress test suy giảm point cloud lên bài kiểm tra calibration",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data-root", default="data/kitti_mini", help="dataset")
    ap.add_argument("--yaw-levels", nargs="+", type=float, default=[0, 0.5, 1.0], help="mức lệch yaw (độ)")
    ap.add_argument("--threshold", type=float, default=0.95, help="ngưỡng cảnh báo hit_ratio theo frame")
    ap.add_argument("--seed", type=int, default=0, help="seed cho mọi phép ngẫu nhiên")
    ap.add_argument("--out", default="results/degradation_sweep.csv", help="CSV kết quả (1 dòng / frame × cấu hình)")
    args = ap.parse_args()

    rows = []
    for frame in list_frames(args.data_root):
        fr = load_frame(args.data_root, frame)
        for deg, (param, values, fn) in DEGRADATIONS.items():
            for value in values:
                prep = prepare_frame(dict(fr, points=fn(fr["points"], value, args.seed)), set(CLASS_GROUP))
                for yaw in args.yaw_levels:
                    objs = measure(prep, perturbed_calib(fr["calib"], "yaw", yaw), fr["image"].shape)
                    n = sum(o["n_points_fixed"] for o in objs)
                    hits = sum(o["hits_fixed"] for o in objs)
                    rows.append({"frame": frame, "degradation": deg, "param": param, "value": value,
                                 "yaw_deg": yaw, "object_points": n, "hits": hits,
                                 "hit_ratio": round(hits / n, 4) if n else float("nan")})
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    # Tổng hợp: hit_ratio gộp 20 frame + tỉ lệ frame bị cảnh báo (hit_ratio < ngưỡng)
    g = df.groupby(["degradation", "value", "yaw_deg"])
    summary = g.agg(object_points=("object_points", "sum"), hits=("hits", "sum"),
                    flag_rate=("hit_ratio", lambda r: round(float((r < args.threshold).mean()), 3))).reset_index()
    summary["hit_ratio"] = (summary["hits"] / summary["object_points"]).round(4)
    out_sum = Path(args.out).with_name("degradation_summary.csv")
    summary.to_csv(out_sum, index=False)
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (deg, (param, values, _)) in zip(axes, DEGRADATIONS.items()):
        for color, yaw in zip((BLUE, ORANGE), (0.0, 1.0)):
            s = summary[(summary.degradation == deg) & (summary.yaw_deg == yaw)]
            ax.plot(s["value"], 100 * s["hit_ratio"], marker="o", color=color,
                    label=f"yaw {yaw:g}°" + (" (calib đúng)" if yaw == 0 else " (calib lệch)"))
            ends = s.iloc[[0, -1]]                         # chỉ ghi nhãn 2 đầu: mức gốc và mức xấu nhất
            for k, (x, y, fl) in enumerate(zip(ends["value"], 100 * ends["hit_ratio"], ends["flag_rate"])):
                ax.annotate(f"{y:.1f}% | cờ {100 * fl:.0f}%", (x, y), xytext=(0, 8 if yaw == 0 else -16),
                            textcoords="offset points", ha="left" if k == 0 else "right", fontsize=8)
        ratio_axis(ax, param + (" (tỉ lệ điểm giữ lại)" if deg == "random_dropout" else " (m)"))
        ax.set_title(deg, loc="left")
        ax.legend(loc="lower left")
        if deg == "random_dropout":
            ax.invert_xaxis()                              # trái = dữ liệu tốt, phải = dữ liệu xấu
    fig.suptitle(f"B2: hit_ratio gộp 20 frame KITTI, 'cờ' = % frame bị cảnh báo (< {100 * args.threshold:.0f}%)",
                 x=0.01, ha="left")
    fig.tight_layout()
    out_fig = Path("results/figures/degradation_sweep.png")
    fig.savefig(out_fig, dpi=150)
    print(f"-> {args.out}\n-> {out_sum}\n-> {out_fig}")


if __name__ == "__main__":
    main()
