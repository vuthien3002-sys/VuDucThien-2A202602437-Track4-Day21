"""Vẽ biểu đồ + bảng phát hiện drift từ các CSV của CP3. Chạy từ gốc repo: python -m src.plot_results

Đọc:  results/yaw_perturb_sweep.csv, results/calib_sweep_summary_{kitti_mini,nusc}.csv,
      results/calib_sweep_objects_kitti_mini.csv
Ghi:  results/figures/yaw_sweep.png             (script mẫu: 3 frame)
      results/figures/claim_rot_vs_trans.png    (kiểm chứng claim: theo class, theo khoảng cách)
      results/figures/kitti_vs_nuscenes.png     (bonus B5)
      results/figures/drift_detection.png + results/drift_detection_kitti.csv (ngưỡng phát hiện)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Bảng màu categorical đã kiểm định CVD (dataviz reference palette), dùng theo thứ tự cố định
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK_2,
    "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "legend.frameon": False, "lines.linewidth": 2,
    "lines.markersize": 7,
})


def ratio_axis(ax, xlabel: str) -> None:
    ax.set_ylim(0, 105)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("% điểm của vật nằm trong 2D box")


def end_label(ax, x, y, text, color) -> None:
    """Nhãn trực tiếp ở cuối đường (chữ màu mực, chấm màu series đã nằm ngay cạnh)."""
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9)


def pick(s: pd.DataFrame, axis: str, group: str = "all", dist: str = "all") -> pd.DataFrame:
    return s[(s.axis == axis) & (s.group == group) & (s.dist_bin == dist)].sort_values("level")


def plot_template(res: Path, fig_dir: Path) -> None:
    df = pd.read_csv(res / "yaw_perturb_sweep.csv", dtype={"frame": str})   # giữ "000011", không thành 11
    names = {"000008": "000008 (đông xe)", "000011": "000011 (nhiều người đi bộ)", "000049": "000049 (bị che khuất)"}
    fig, ax = plt.subplots(figsize=(7, 4))
    for color, (frame, g) in zip((BLUE, ORANGE, AQUA), df.groupby("frame")):
        ax.plot(g["yaw_deg"], 100 * g["hit_ratio"], marker="o", color=color, label=names.get(frame, frame))
        end_label(ax, g["yaw_deg"].iloc[-1], 100 * g["hit_ratio"].iloc[-1], f"{100 * g['hit_ratio'].iloc[-1]:.1f}%", color)
    ratio_axis(ax, "Lệch yaw (độ)")
    ax.set_xlim(-0.1, 3.5)
    ax.set_title("KITTI: lệch yaw làm điểm LiDAR trượt khỏi 2D box (script mẫu)", loc="left")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(fig_dir / "yaw_sweep.png", dpi=150)
    plt.close(fig)


def plot_claim(res: Path, fig_dir: Path) -> None:
    s = pd.read_csv(res / "calib_sweep_summary_kitti_mini.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))

    # (1) yaw theo class: vật hẹp (người) so với vật rộng (xe)
    for color, (grp, name) in zip((BLUE, ORANGE), (("vehicle", "Xe (Car/Van)"), ("pedestrian", "Người đi bộ"))):
        g = pick(s, "yaw", grp)
        n = int(g["n_objects"].iloc[0])
        a1.plot(g["level"], 100 * g["hit_ratio_fixed"], marker="o", color=color, label=f"{name}, {n} vật")
        dy = 7 if grp == "vehicle" else -15                 # nhãn xe ở trên, nhãn người ở dưới: không chồng nhau
        for x, y in zip(g["level"], 100 * g["hit_ratio_fixed"]):
            if x in (0, 1):
                a1.annotate(f"{y:.1f}", (x, y), xytext=(4, dy), textcoords="offset points", fontsize=8)
    ratio_axis(a1, "Lệch yaw (độ)")
    a1.set_title("(a) Lệch yaw theo class, 20 frame KITTI", loc="left")
    a1.legend(loc="lower left")

    # (2) theo khoảng cách: lệch xoay (yaw 1°) so với lệch dịch ngang (ty 10 cm)
    # màu khác panel (a) vì ở đây màu mã hoá loại lệch, không phải class
    bins = ["0-15m", "15-30m", ">30m"]
    x = np.arange(len(bins))
    cases = (("yaw", 1.0, "Xoay: yaw 1°", AQUA), ("ty", 0.10, "Dịch ngang: ty 10 cm", YELLOW))
    width = 0.36
    for k, (axis, level, name, color) in enumerate(cases):
        g = s[(s.axis == axis) & (s.level == level) & (s.group == "all")].set_index("dist_bin").loc[bins]
        xs = x + (k - 0.5) * (width + 0.02)               # khe 2% giữa 2 cột
        a2.bar(xs, 100 * g["hit_ratio_fixed"], width=width, color=color, label=name)
        for xi, y in zip(xs, 100 * g["hit_ratio_fixed"]):
            a2.annotate(f"{y:.1f}", (xi, y), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    base = s[(s.axis == "yaw") & (s.level == 0) & (s.group == "all")].set_index("dist_bin").loc[bins]
    a2.set_xticks(x, [f"{b}\n({int(n)} vật)" for b, n in zip(bins, base["n_objects"])])
    ratio_axis(a2, "Khoảng cách vật tới camera")
    a2.set_title("(b) Theo khoảng cách: xoay hại vật xa, dịch thì không", loc="left")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)   # đặt dưới trục, không đè lên cột
    fig.tight_layout()
    fig.savefig(fig_dir / "claim_rot_vs_trans.png", dpi=150)
    plt.close(fig)


def plot_datasets(res: Path, fig_dir: Path) -> None:
    """B5: cùng metric, cùng mức lệch yaw trên 2 dataset, chia theo class."""
    sk = pd.read_csv(res / "calib_sweep_summary_kitti_mini.csv")
    sn = pd.read_csv(res / "calib_sweep_summary_nusc.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (grp, name) in zip(axes, (("vehicle", "Xe"), ("pedestrian", "Người đi bộ"))):
        for color, (s, ds) in zip((BLUE, ORANGE), ((sk, "KITTI (64 beam, f≈721 px)"),
                                                    (sn, "nuScenes (32 beam, f≈1253-1266 px)"))):
            g = pick(s, "yaw", grp)
            ax.plot(g["level"], 100 * g["hit_ratio_fixed"], marker="o", color=color,
                    label=f"{ds}: {int(g['n_objects'].iloc[0])} vật")
        ratio_axis(ax, "Lệch yaw (độ)")
        ax.set_title(f"{name}", loc="left")
        ax.legend(loc="lower left")
    fig.suptitle("Bonus B5: lệch yaw cho cùng xu hướng trên KITTI và nuScenes (tiêu cự bị triệt tiêu)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(fig_dir / "kitti_vs_nuscenes.png", dpi=150)
    plt.close(fig)


def drift_detection(res: Path, fig_dir: Path, thresholds=(0.98, 0.95, 0.90)) -> None:
    """Mỗi frame KITTI tính 1 hit_ratio (gộp mọi vật). Cảnh báo drift khi hit_ratio < ngưỡng.
    Mức 0° = calib đúng -> mọi cảnh báo ở 0° là báo nhầm (false alarm)."""
    o = pd.read_csv(res / "calib_sweep_objects_kitti_mini.csv", dtype={"frame": str})
    o = o[o.axis == "yaw"]
    fr = o.groupby(["frame", "level"])[["hits_fixed", "n_points_fixed"]].sum().reset_index()
    fr["hit_ratio"] = fr["hits_fixed"] / fr["n_points_fixed"]
    rows = []
    for th in thresholds:
        for level, g in fr.groupby("level"):
            flagged = g[g["hit_ratio"] < th]
            rows.append({"threshold": th, "yaw_deg": level, "frames": len(g), "flagged": len(flagged),
                         "flag_rate": round(len(flagged) / len(g), 3),
                         "missed_frames": " ".join(sorted(set(g["frame"]) - set(flagged["frame"])))
                         if level > 0 else ""})
    table = pd.DataFrame(rows)
    table.to_csv(res / "drift_detection_kitti.csv", index=False)
    print(table.pivot_table(index="yaw_deg", columns="threshold", values="flag_rate").to_string())

    fig, ax = plt.subplots(figsize=(7, 4))
    for frame, g in fr.groupby("frame"):
        ax.plot(g["level"], 100 * g["hit_ratio"], color="#b9b8b2", linewidth=1, marker="o", markersize=3)
    med = fr.groupby("level")["hit_ratio"].median()
    ax.plot(med.index, 100 * med.values, color=BLUE, marker="o", label="Trung vị 20 frame")
    ax.axhline(95, color=ORANGE, linewidth=1.5, linestyle="--", label="Ngưỡng cảnh báo 95%")
    worst0 = fr[fr.level == 0].sort_values("hit_ratio").iloc[0]
    ax.annotate(f"mức sàn ở 0°: frame {worst0['frame']} chỉ {100 * worst0['hit_ratio']:.1f}%",
                (0, 100 * worst0["hit_ratio"]), xytext=(0.3, 28), textcoords="data", fontsize=8,
                color=INK_2, arrowprops={"arrowstyle": "-", "color": INK_2, "linewidth": 0.8})
    ratio_axis(ax, "Lệch yaw (độ)")
    ax.set_title("Phát hiện drift theo từng frame (mỗi đường xám = 1 frame KITTI)", loc="left")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(fig_dir / "drift_detection.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Vẽ biểu đồ kết quả CP3 từ các CSV trong results/")
    ap.add_argument("--results", default="results", help="thư mục chứa CSV")
    args = ap.parse_args()
    res = Path(args.results)
    fig_dir = res / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_template(res, fig_dir)
    plot_claim(res, fig_dir)
    plot_datasets(res, fig_dir)
    drift_detection(res, fig_dir)
    print(f"-> {fig_dir}/yaw_sweep.png, claim_rot_vs_trans.png, kitti_vs_nuscenes.png, drift_detection.png")


if __name__ == "__main__":
    main()
