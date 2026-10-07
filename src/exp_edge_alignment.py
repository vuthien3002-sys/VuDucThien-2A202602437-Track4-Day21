"""Bonus B1: so sánh 2 metric kiểm tra calibration trên cùng dữ liệu, cùng mức lệch.

  Metric A  hit_ratio   : % điểm của vật (trong 3D box GT) rơi vào 2D box label   -> cần label
  Metric B  edge_score  : điểm LiDAR nằm ở "mép độ sâu" có trùng cạnh ảnh (Canny) không -> KHÔNG cần label

edge_score của 1 frame:
  1. Ảnh xám -> Gaussian blur -> Canny -> distance transform: mỗi pixel biết khoảng cách (px) tới cạnh gần nhất.
  2. Chiếu point cloud, dựng ảnh độ sâu thưa; điểm là "mép độ sâu" nếu trong ô cao 5 x rộng k px quanh nó có điểm
     xa hơn > 1 m (điểm nằm ở rìa vật phía trước nền) và điểm gần hơn 40 m. k = 5 cho KITTI (64 beam, 2 điểm liền
     nhau cách ~1 px trên ảnh), k = 15 cho nuScenes (32 beam, cách ~7 px) -> ô phải đủ rộng để chứa hàng xóm.
  3. score = trung bình exp(-d^2 / (2·sigma^2)) với d = khoảng cách từ mép độ sâu tới cạnh ảnh, sigma = 3 px.
     Calib đúng -> mép vật trong LiDAR trùng cạnh trong ảnh -> score cao.
Ý tưởng theo Levinson & Thrun, "Automatic Online Calibration of Cameras and Lasers", RSS 2013 (tự cài đặt lại).

So sánh khả năng phát hiện bằng AUROC giữa 20 frame calib đúng và 20 frame lệch (không cần chọn ngưỡng).
Thêm: trên nuScenes, metric nào phát hiện được lỗi Time (bỏ bù chuyển động)?

Chạy từ gốc repo:  python -m src.exp_edge_alignment
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exp_calib_sweep import CLASS_GROUP, measure, perturbed_calib, prepare_frame
from src.plot_results import BLUE, INK_2, ORANGE, SURFACE
from starter.datasets import list_frames, load_frame
from starter.projection import project_velo_to_image


def edge_distance_map(image: np.ndarray, canny=(50, 150)) -> np.ndarray:
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, *canny)
    return cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)      # 0 tại pixel cạnh


def depth_edge_points(points: np.ndarray, calib, shape, jump_m=1.0, max_depth=40.0, kernel_w=5):
    """Trả (uv, depth) của các điểm nằm ở mép độ sâu (nền phía sau xa hơn > jump_m trong ô 5 x kernel_w)."""
    uv, depth, _ = project_velo_to_image(points, calib, shape)
    ui, vi = uv[:, 0].astype(int), uv[:, 1].astype(int)
    D = np.zeros(shape[:2], np.float32)
    order = np.argsort(-depth)                                       # gán xa trước, gần sau -> pixel giữ điểm gần nhất
    D[vi[order], ui[order]] = depth[order]
    far = cv2.dilate(D, np.ones((5, kernel_w), np.uint8))            # độ sâu lớn nhất trong ô (ô trống = 0)
    edge = (far[vi, ui] - depth > jump_m) & (depth < max_depth)
    return uv[edge], depth[edge]


def edge_score(points, calib, shape, dist_map, sigma=3.0, kernel_w=5) -> tuple[float, int]:
    uv, _ = depth_edge_points(points, calib, shape, kernel_w=kernel_w)
    if len(uv) == 0:
        return float("nan"), 0
    d = dist_map[uv[:, 1].astype(int), uv[:, 0].astype(int)]
    return float(np.mean(np.exp(-d ** 2 / (2 * sigma ** 2)))), len(uv)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score calib đúng > score calib lệch), hoà tính 0.5 (Mann-Whitney). 1.0 = tách hoàn hảo, 0.5 = đoán mò."""
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(((pos > neg) + 0.5 * (pos == neg)).mean())


def hit_ratio(fr, calib) -> float:
    objs = measure(prepare_frame(fr, set(CLASS_GROUP)), calib, fr["image"].shape)
    n = sum(o["n_points_fixed"] for o in objs)
    return sum(o["hits_fixed"] for o in objs) / n if n else float("nan")


def kitti_sweep(levels) -> pd.DataFrame:
    rows = []
    for frame in list_frames("data/kitti_mini"):
        fr = load_frame("data/kitti_mini", frame)
        pts = fr["points"][np.isfinite(fr["points"]).all(axis=1)]
        dmap = edge_distance_map(fr["image"])
        for yaw in levels:
            calib = perturbed_calib(fr["calib"], "yaw", yaw)
            score, n_edge = edge_score(pts, calib, fr["image"].shape, dmap)
            rows.append({"dataset": "kitti_mini", "frame": frame, "yaw_deg": yaw, "edge_score": round(score, 4),
                         "n_edge_points": n_edge, "hit_ratio": round(hit_ratio(fr, calib), 4)})
    return pd.DataFrame(rows)


def nusc_time(kernel_w: int) -> pd.DataFrame:
    """nuScenes: có bù chuyển động (đúng) so với không bù (lỗi Time), cả 2 metric."""
    root, rows = "data/nuscenes_mini_subset", []
    for frame in list_frames(root):
        for ego in (True, False):
            fr = load_frame(root, frame, use_ego_motion=ego)
            pts = fr["points"][np.isfinite(fr["points"]).all(axis=1)]
            score, n_edge = edge_score(pts, fr["calib"], fr["image"].shape, edge_distance_map(fr["image"]),
                                       kernel_w=kernel_w)
            rows.append({"dataset": "nuscenes", "frame": frame, "scene": frame[:10], "ego_motion": ego,
                         "edge_score": round(score, 4), "n_edge_points": n_edge,
                         "hit_ratio": round(hit_ratio(fr, fr["calib"]), 4)})
    return pd.DataFrame(rows)


def demo_image(out: Path, frame="000011", yaws=(0.0, 1.0)) -> None:
    """Minh hoạ: cạnh Canny (xám) + mép độ sâu LiDAR, xanh = trùng cạnh ảnh (< 3 px), đỏ = lệch."""
    fr = load_frame("data/kitti_mini", frame)
    pts = fr["points"][np.isfinite(fr["points"]).all(axis=1)]
    dmap = edge_distance_map(fr["image"])
    tiles = []
    for yaw in yaws:
        calib = perturbed_calib(fr["calib"], "yaw", yaw)
        uv, _ = depth_edge_points(pts, calib, fr["image"].shape)
        d = dmap[uv[:, 1].astype(int), uv[:, 0].astype(int)]
        vis = (0.45 * fr["image"]).astype(np.uint8)
        vis[dmap == 0] = (170, 170, 170)
        for (u, v), di in zip(uv.astype(int), d):
            cv2.circle(vis, (int(u), int(v)), 2, (60, 200, 60) if di < 3 else (40, 40, 230), -1)
        score, n = edge_score(pts, calib, fr["image"].shape, dmap)
        cv2.rectangle(vis, (0, 0), (760, 30), (0, 0, 0), -1)
        cv2.putText(vis, f"KITTI {frame} yaw {yaw:g} deg: edge_score={score:.3f} ({n} depth-edge pts, "
                         f"green = within 3 px of a Canny edge)", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(vis)
    cv2.imwrite(str(out), np.vstack(tiles))


def main() -> None:
    ap = argparse.ArgumentParser(description="B1: so sánh hit_ratio (cần label) với edge_score (không cần label)",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--yaw-levels", nargs="+", type=float, default=[0, 0.25, 0.5, 1.0, 2.0], help="mức lệch yaw (độ)")
    ap.add_argument("--nusc-kernel-w", type=int, default=15, help="bề rộng ô tìm mép độ sâu cho nuScenes (px)")
    ap.add_argument("--results", default="results", help="thư mục kết quả")
    args = ap.parse_args()
    res = Path(args.results)

    k = kitti_sweep(args.yaw_levels)
    k.to_csv(res / "edge_vs_hit_kitti.csv", index=False)
    base = k[k.yaw_deg == 0].set_index("frame")
    rows = []
    for yaw in args.yaw_levels[1:]:
        cur = k[k.yaw_deg == yaw].set_index("frame")
        rows.append({"test": f"KITTI yaw {yaw:g} deg vs 0", "auroc_hit_ratio": auroc(base.hit_ratio, cur.hit_ratio),
                     "auroc_edge_score": auroc(base.edge_score, cur.edge_score),
                     "frames_edge_drops": int((cur.edge_score < base.edge_score).sum()),
                     "frames_hit_drops": int((cur.hit_ratio < base.hit_ratio).sum()), "frames": len(cur)})

    n = nusc_time(args.nusc_kernel_w)
    n.to_csv(res / "edge_vs_hit_nusc_time.csv", index=False)
    for scene in ("all", "scene-0103", "scene-1094"):
        s = n if scene == "all" else n[n.scene == scene]
        ok, bad = s[s.ego_motion].set_index("frame"), s[~s.ego_motion].set_index("frame")
        rows.append({"test": f"nuScenes Time error ({scene})", "auroc_hit_ratio": auroc(ok.hit_ratio, bad.hit_ratio),
                     "auroc_edge_score": auroc(ok.edge_score, bad.edge_score),
                     "frames_edge_drops": int((bad.edge_score < ok.edge_score).sum()),
                     "frames_hit_drops": int((bad.hit_ratio < ok.hit_ratio).sum()), "frames": len(ok)})
    # Chỉ xét frame có lỗi Time đủ lớn (điểm dời trung vị > 8 px, đo bởi src.make_failures) - frame xe đứng yên
    # thì bỏ bù cũng không sai gì, không thể đòi metric nào phát hiện.
    ts = res / "time_sync_nusc.csv"
    if ts.exists():
        big = set(pd.read_csv(ts).query("px_shift_median > 8").frame)
        s = n[n.frame.isin(big)]
        ok, bad = s[s.ego_motion].set_index("frame"), s[~s.ego_motion].set_index("frame")
        rows.append({"test": "nuScenes Time error (frames shifted > 8 px)",
                     "auroc_hit_ratio": auroc(ok.hit_ratio, bad.hit_ratio),
                     "auroc_edge_score": auroc(ok.edge_score, bad.edge_score),
                     "frames_edge_drops": int((bad.edge_score < ok.edge_score).sum()),
                     "frames_hit_drops": int((bad.hit_ratio < ok.hit_ratio).sum()), "frames": len(ok)})
    summary = pd.DataFrame(rows).round(3)
    summary.to_csv(res / "edge_vs_hit_summary.csv", index=False)
    print(summary.to_string(index=False))

    metrics = ((BLUE, "hit_ratio", "hit_ratio (cần label)"), (ORANGE, "edge_score", "edge_score (không cần label)"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    # (a) giá trị tuyệt đối từng frame: ngưỡng cố định chỉ tốt khi 2 cụm 0° và lệch không chồng nhau
    levels = [lv for lv in args.yaw_levels if lv in (0, 0.5, 1.0)]
    for m, (color, col, name) in enumerate(metrics):
        for i, lv in enumerate(levels):
            v = 100 * k[k.yaw_deg == lv][col].to_numpy()
            xs = i + (m - 0.5) * 0.35 + np.linspace(-0.1, 0.1, len(v))     # rải ngang cố định, không ngẫu nhiên
            a1.scatter(xs, v, s=16, color=color, label=name if i == 0 else None, edgecolors=SURFACE, linewidths=0.5)
    a1.set_xticks(range(len(levels)), [f"yaw {lv:g}°" for lv in levels])
    a1.set_ylim(0, 105)
    a1.set_ylabel("giá trị metric (%, edge_score × 100)")
    a1.set_title("(a) Giá trị tuyệt đối, mỗi chấm = 1 frame KITTI", loc="left")
    a1.legend(loc="lower left")
    a1.annotate("edge_score ở 0° đã trải 29-78\n-> chồng lên cụm lệch 1°", (1.22, 28), fontsize=8, color=INK_2)
    # (b) chuẩn hoá theo chính frame đó ở 0°: cả 2 metric đều giảm khi lệch
    for color, col, name in metrics:
        rel = k.pivot(index="frame", columns="yaw_deg", values=col)
        rel = 100 * rel.div(rel[0.0], axis=0)
        q = rel.quantile([0.25, 0.5, 0.75])
        a2.fill_between(q.columns, q.loc[0.25], q.loc[0.75], color=color, alpha=0.15, linewidth=0)
        a2.plot(q.columns, q.loc[0.5], marker="o", color=color, label=name)
    a2.set_ylim(0, 105)
    a2.set_xlabel("Lệch yaw (độ)")
    a2.set_ylabel("% so với chính frame đó khi calib đúng")
    a2.set_title("(b) Tương đối: trung vị 20 frame, dải = 25-75%", loc="left")
    a2.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(res / "figures" / "edge_vs_hit.png", dpi=150)
    demo_image(res / "figures" / "edge_alignment_demo.png")
    print(f"-> {res}/edge_vs_hit_*.csv, figures/edge_vs_hit.png, figures/edge_alignment_demo.png")


if __name__ == "__main__":
    main()
