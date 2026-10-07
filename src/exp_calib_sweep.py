"""Topic A (mở rộng từ src/exp_yaw_sweep.py): quét lệch calibration theo 6 trục, đo theo TỪNG VẬT THỂ.

So với script mẫu, bản này thêm:
  * 6 trục lệch: yaw / pitch / roll (độ) và tx / ty / tz (mét, trong velodyne frame: x trước, y trái, z lên).
  * Ghi CSV theo từng vật thể (class, khoảng cách, bề rộng box trên ảnh) để chia nhóm theo class và khoảng cách.
  * Hai cách tính hit_ratio:
      - hit_ratio_template: như script mẫu, mẫu số = điểm trong 3D box VÀ còn nằm trong ảnh SAU khi lệch.
        Điểm bị đẩy ra ngoài ảnh biến mất khỏi mẫu số -> tỉ lệ bị đánh giá cao hơn thực tế ở vật sát mép ảnh.
      - hit_ratio_fixed: mẫu số cố định = điểm trong 3D box nằm trong ảnh theo calib GỐC.
        Điểm bị đẩy ra ngoài ảnh được tính là trượt.
  * Chạy được cả KITTI và nuScenes (cùng code, cùng metric), có tuỳ chọn tắt bù chuyển động của nuScenes.

Không có phép ngẫu nhiên nào -> chạy lại luôn ra đúng cùng số.

Chạy từ gốc repo (xem thêm --help):
    python -m src.exp_calib_sweep                                   # KITTI, 20 frame, 6 trục
    python -m src.exp_calib_sweep --data-root data/nuscenes_mini_subset --axes yaw ty --tag nusc
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.exp_yaw_sweep import points_in_box
from starter.datasets import dataset_type, list_frames, load_frame
from starter.projection import perturb_extrinsic, project_velo_to_image, velo_to_cam

ROT_AXES = ("roll", "pitch", "yaw")
TRANS_AXES = ("tx", "ty", "tz")
# Gom class của 2 dataset về 3 nhóm để so sánh được KITTI với nuScenes
CLASS_GROUP = {
    "Car": "vehicle", "Van": "vehicle", "Truck": "vehicle", "Bus": "vehicle",
    "Pedestrian": "pedestrian",
    "Cyclist": "cyclist", "Bicycle": "cyclist", "Motorcycle": "cyclist",
}
DIST_BINS = [0, 15, 30, np.inf]
DIST_LABELS = ["0-15m", "15-30m", ">30m"]


def perturbed_calib(calib, axis: str, level: float):
    """Calib bị lệch đúng 1 trục, các trục khác giữ nguyên 0."""
    if axis in ROT_AXES:
        return perturb_extrinsic(calib, **{f"{axis}_deg": level})
    t = [0.0, 0.0, 0.0]
    t[TRANS_AXES.index(axis)] = level
    return perturb_extrinsic(calib, t_xyz_m=tuple(t))


def prepare_frame(fr: dict, classes: set[str]) -> dict:
    """Phần không phụ thuộc mức lệch: tính 1 lần cho mỗi frame."""
    pts = fr["points"][np.isfinite(fr["points"]).all(axis=1)]
    cam_true = velo_to_cam(pts[:, :3], fr["calib"])
    _, _, mask_true = project_velo_to_image(pts, fr["calib"], fr["image"].shape)
    objects = []
    for i, obj in enumerate(fr["labels"]):
        if obj.type not in classes:
            continue
        idx = np.flatnonzero(points_in_box(cam_true, obj))      # điểm thật sự thuộc vật (theo calib gốc)
        objects.append({
            "obj_id": i, "type": obj.type, "group": CLASS_GROUP.get(obj.type, "other"),
            "distance_m": round(float(np.hypot(obj.location[0], obj.location[2])), 1),
            "bbox_w_px": round(float(obj.bbox[2] - obj.bbox[0]), 1),
            "bbox_h_px": round(float(obj.bbox[3] - obj.bbox[1]), 1),
            "truncated": round(float(obj.truncated), 2), "occluded": int(obj.occluded),
            "bbox": obj.bbox, "idx": idx, "idx_visible": idx[mask_true[idx]],
        })
    return {"pts": pts, "objects": objects}


def measure(prep: dict, calib, image_shape) -> list[dict]:
    """Chiếu bằng calib (có thể đã lệch) rồi đếm điểm của từng vật còn rơi trong 2D box của nó."""
    uv, _, mask = project_velo_to_image(prep["pts"], calib, image_shape)
    uv_all = np.full((len(prep["pts"]), 2), np.nan)
    uv_all[mask] = uv
    rows = []
    for o in prep["objects"]:
        x1, y1, x2, y2 = o["bbox"]
        u, v = uv_all[:, 0], uv_all[:, 1]
        in_box2d = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)    # NaN (ngoài ảnh) -> False
        sel = o["idx"][mask[o["idx"]]]                               # mẫu số kiểu script mẫu
        hits = int(in_box2d[sel].sum())
        hits_fixed = int(in_box2d[o["idx_visible"]].sum())           # chỉ đếm trên mẫu số cố định
        n_fixed = len(o["idx_visible"])
        rows.append({k: o[k] for k in ("obj_id", "type", "group", "distance_m", "bbox_w_px", "bbox_h_px",
                                       "truncated", "occluded")}
                    | {"n_points_fixed": n_fixed, "n_points_template": len(sel),
                       "hits_fixed": hits_fixed, "hits_template": hits})
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Gộp theo (dataset, axis, level, nhóm class, nhóm khoảng cách): tỉ lệ tính trên tổng số điểm."""
    df = df.assign(dist_bin=pd.cut(df["distance_m"], DIST_BINS, labels=DIST_LABELS, right=False).astype(str))
    keys = ["dataset", "axis", "level", "unit"]
    parts = []
    for extra in ([], ["group"], ["dist_bin"], ["group", "dist_bin"]):
        g = df.groupby(keys + extra, sort=True).agg(
            n_objects=("obj_id", "size"), n_points_fixed=("n_points_fixed", "sum"),
            hits_fixed=("hits_fixed", "sum"), n_points_template=("n_points_template", "sum"),
            hits_template=("hits_template", "sum")).reset_index()
        for col in ("group", "dist_bin"):
            if col not in g:
                g[col] = "all"
        parts.append(g)
    s = pd.concat(parts, ignore_index=True)
    s["hit_ratio_fixed"] = (s["hits_fixed"] / s["n_points_fixed"]).round(4)
    s["hit_ratio_template"] = (s["hits_template"] / s["n_points_template"]).round(4)
    return s[keys + ["group", "dist_bin", "n_objects", "n_points_fixed", "hit_ratio_fixed",
                     "n_points_template", "hit_ratio_template"]]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Quét lệch calibration LiDAR->camera theo từng trục, đo % điểm của mỗi vật thể "
                    "còn rơi đúng vào 2D box của nó. Ghi 1 CSV theo vật thể + 1 CSV tổng hợp.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data-root", default="data/kitti_mini",
                    help="data/kitti_mini, data/synthetic hoặc data/nuscenes_mini_subset")
    ap.add_argument("--frames", nargs="*", default=None, help="danh sách frame; bỏ trống = mọi frame trong dataset")
    ap.add_argument("--frame-step", type=int, default=1, help="chỉ lấy mỗi frame thứ k (giảm thời gian chạy)")
    ap.add_argument("--axes", nargs="+", default=list(ROT_AXES + TRANS_AXES), choices=ROT_AXES + TRANS_AXES,
                    help="các trục lệch cần quét, mỗi lần chỉ lệch 1 trục")
    ap.add_argument("--rot-levels", nargs="+", type=float, default=[0, 0.5, 1, 2, 3],
                    help="các mức lệch góc (độ) cho roll/pitch/yaw")
    ap.add_argument("--trans-levels", nargs="+", type=float, default=[0, 0.02, 0.05, 0.10, 0.20],
                    help="các mức lệch dịch (mét) cho tx/ty/tz")
    ap.add_argument("--classes", nargs="+", default=list(CLASS_GROUP), help="class được tính")
    ap.add_argument("--ignore-ego-motion", action="store_true",
                    help="chỉ cho nuScenes: tắt bù chuyển động giữa thời điểm LiDAR và camera")
    ap.add_argument("--tag", default=None, help="hậu tố tên file kết quả (mặc định = tên dataset)")
    ap.add_argument("--out-dir", default="results", help="thư mục ghi CSV")
    args = ap.parse_args()

    ds_name = Path(args.data_root).name
    tag = args.tag or ds_name
    frames = (args.frames or list_frames(args.data_root))[::args.frame_step]
    kwargs = ({"use_ego_motion": not args.ignore_ego_motion}
              if dataset_type(args.data_root) == "nuscenes" else {})

    rows = []
    for frame in frames:
        fr = load_frame(args.data_root, frame, **kwargs)
        prep = prepare_frame(fr, set(args.classes))
        for axis in args.axes:
            levels, unit = (args.rot_levels, "deg") if axis in ROT_AXES else (args.trans_levels, "m")
            for level in levels:
                calib = perturbed_calib(fr["calib"], axis, level)
                for r in measure(prep, calib, fr["image"].shape):
                    rows.append({"dataset": ds_name, "frame": frame, "axis": axis, "level": level,
                                 "unit": unit} | r)
        print(f"{frame}: {len(prep['objects'])} objects")

    df = pd.DataFrame(rows)
    df["hit_ratio_fixed"] = (df["hits_fixed"] / df["n_points_fixed"]).round(4)
    df["hit_ratio_template"] = (df["hits_template"] / df["n_points_template"]).round(4)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_csv, sum_csv = out_dir / f"calib_sweep_objects_{tag}.csv", out_dir / f"calib_sweep_summary_{tag}.csv"
    df.to_csv(obj_csv, index=False)
    summary = summarize(df)
    summary.to_csv(sum_csv, index=False)

    view = summary[(summary["group"] == "all") & (summary["dist_bin"] == "all")]
    print(view.pivot_table(index=["axis", "level"], values="hit_ratio_fixed").to_string())
    print(f"-> {obj_csv} ({len(df)} dong)\n-> {sum_csv} ({len(summary)} dong)")


if __name__ == "__main__":
    main()
