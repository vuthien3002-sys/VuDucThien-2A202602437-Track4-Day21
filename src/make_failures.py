"""CP4: tạo ảnh failure case + bảng phân tích lỗi thời gian nuScenes. Chạy từ gốc repo: python -m src.make_failures

  fail_01_yaw_1deg_far_pedestrian.png  Geometry: người đi bộ 34 m (box 15 px) mất gần hết điểm khi yaw 1°
  fail_02_nusc_no_ego_motion.png       Time (+Metric): bỏ bù chuyển động 35.6 ms, điểm VÀ box cùng lệch
  fail_03_metric_template_edge_car.png Metric: script mẫu báo 100% cho xe sát mép ảnh dù 90% điểm đã bị đẩy ra ngoài
  results/time_sync_nusc.csv           độ lệch pixel theo độ sâu, cho cả 80 keyframe nuScenes
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.exp_yaw_sweep import points_in_box
from starter.datasets import list_frames, load_frame
from starter.projection import cam_to_image, draw_box2d, perturb_extrinsic, project_velo_to_image, velo_to_cam

GREEN, RED, WHITE, BLACK, CYAN = (60, 200, 60), (40, 40, 230), (255, 255, 255), (0, 0, 0), (230, 200, 40)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def finite_points(fr: dict) -> np.ndarray:
    return fr["points"][np.isfinite(fr["points"]).all(axis=1)]


def object_uv(fr: dict, obj, calib) -> np.ndarray:
    """Pixel (M, 2) của các điểm thuộc vật (chọn bằng calib GỐC) khi chiếu bằng `calib`; điểm ra ngoài ảnh bị bỏ."""
    pts = finite_points(fr)
    sel = points_in_box(velo_to_cam(pts[:, :3], fr["calib"]), obj)
    uv, _, _ = project_velo_to_image(pts[sel], calib, fr["image"].shape)
    return uv


def in_box(uv: np.ndarray, bbox) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)


def label(img: np.ndarray, lines: list[str], scale: float = 0.55) -> np.ndarray:
    """Dải chữ đen phía trên ảnh (cv2.putText không hiển thị được tiếng Việt có dấu -> viết không dấu)."""
    h = 12 + 24 * len(lines)
    bar = np.zeros((h, img.shape[1], 3), np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(bar, t, (8, 24 + 24 * i), FONT, scale, WHITE, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def crop_zoom(image: np.ndarray, center, half_wh, zoom: float | None = None, target_w: int | None = None):
    """Cắt vùng quanh center rồi phóng to (theo zoom, hoặc sao cho rộng target_w pixel);
    trả ảnh + hàm đổi toạ độ pixel gốc -> pixel ảnh phóng."""
    cx, cy = center
    hw, hh = half_wh
    x0, y0 = int(max(0, cx - hw)), int(max(0, cy - hh))
    x1, y1 = int(min(image.shape[1], cx + hw)), int(min(image.shape[0], cy + hh))
    if target_w:
        zoom = target_w / (x1 - x0)
    crop = cv2.resize(image[y0:y1, x0:x1], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    return crop, (lambda uv: (np.asarray(uv, float) - [x0, y0]) * zoom)


def draw_hits(canvas: np.ndarray, uv_zoom: np.ndarray, hit: np.ndarray, r: int = 4) -> None:
    for (u, v), h in zip(uv_zoom.astype(int), hit):
        cv2.circle(canvas, (int(u), int(v)), r, GREEN if h else RED, -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(u), int(v)), r, BLACK, 1, cv2.LINE_AA)


def zoomed_box(canvas, to_zoom, bbox, color, thickness=2) -> None:
    (x1, y1), (x2, y2) = to_zoom([bbox[:2], bbox[2:]]).astype(int)
    cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)


def fail_geometry(out: Path, frame="000011", obj_id=3, yaws=(0.0, 0.5, 1.0)) -> None:
    fr = load_frame("data/kitti_mini", frame)
    obj = fr["labels"][obj_id]
    dist = float(np.hypot(obj.location[0], obj.location[2]))
    bw = obj.bbox[2] - obj.bbox[0]
    f = fr["calib"].P2[0, 0]
    center = ((obj.bbox[0] + obj.bbox[2]) / 2, (obj.bbox[1] + obj.bbox[3]) / 2)
    panels = []
    for yaw in yaws:
        uv = object_uv(fr, obj, perturb_extrinsic(fr["calib"], yaw_deg=yaw))
        hit = in_box(uv, obj.bbox)
        canvas, to_zoom = crop_zoom(fr["image"], center, (60, 45), zoom=3.4)
        zoomed_box(canvas, to_zoom, obj.bbox, GREEN)
        draw_hits(canvas, to_zoom(uv), hit)
        shift = f * np.tan(np.deg2rad(yaw))
        panels.append(label(canvas, [f"yaw {yaw:g} deg: {hit.sum()}/{len(hit)} pts in box = {100 * hit.mean():.1f}%",
                                     f"shift f*tan(yaw) = {shift:.1f} px vs box {bw:.1f} px"]))
    row = np.hstack(panels)
    ctx = draw_box2d(fr["image"], obj.bbox, color=RED, label=f"ped {dist:.1f} m")
    ctx = cv2.resize(ctx, (row.shape[1], int(ctx.shape[0] * row.shape[1] / ctx.shape[1])))
    ctx = label(ctx, [f"FAIL 01 (Geometry) - KITTI {frame}, Pedestrian #{obj_id} at {dist:.1f} m, 2D box only {bw:.1f} px wide",
                      "green = LiDAR point of this person inside its 2D box, red = outside (points chosen with TRUE calib)"])
    cv2.imwrite(str(out / "fail_01_yaw_1deg_far_pedestrian.png"), np.vstack([ctx, row]))


def fail_metric(out: Path, frame="000011", obj_id=4, yaws=(0.0, 3.0)) -> None:
    fr = load_frame("data/kitti_mini", frame)
    obj = fr["labels"][obj_id]
    pts = finite_points(fr)
    sel = points_in_box(velo_to_cam(pts[:, :3], fr["calib"]), obj)
    n_fixed = int(project_velo_to_image(pts[sel], fr["calib"], fr["image"].shape)[2].sum())
    H, W = fr["image"].shape[:2]
    panels = []
    for yaw in yaws:
        uv = object_uv(fr, obj, perturb_extrinsic(fr["calib"], yaw_deg=yaw))
        hit = in_box(uv, obj.bbox)
        canvas, to_zoom = crop_zoom(fr["image"], (100, H / 2), (100, H / 2), zoom=2.0)
        zoomed_box(canvas, to_zoom, obj.bbox, GREEN)
        draw_hits(canvas, to_zoom(uv), hit, r=3)
        panels.append(label(canvas, [f"yaw {yaw:g} deg: points left in image = {len(uv)}/{n_fixed}",
                                     f"template ratio = {hit.sum()}/{len(uv)} = {100 * hit.mean():.1f}%",
                                     f"fixed ratio    = {hit.sum()}/{n_fixed} = {100 * hit.sum() / n_fixed:.1f}%"]))
    row = np.hstack(panels)
    head = label(np.zeros((1, row.shape[1], 3), np.uint8),
                 [f"FAIL 03 (Metric) - KITTI {frame}, Car #{obj_id} at {np.hypot(*obj.location[[0, 2]]):.1f} m, "
                  f"truncated {obj.truncated:.2f} (cut by left image edge)",
                  "yaw pushes its points out of the image; template metric drops them from the denominator",
                  "-> reports 100% 'aligned' while 90% of the car's points are lost"], scale=0.5)
    cv2.imwrite(str(out / "fail_03_metric_template_edge_car.png"), np.vstack([head, row]))


def time_sync_table(res: Path) -> pd.DataFrame:
    """Với mỗi keyframe nuScenes: chiếu cùng 1 point cloud bằng calib có / không bù chuyển động,
    đo điểm bị dời bao nhiêu pixel theo độ sâu."""
    root = "data/nuscenes_mini_subset"
    rows = []
    for frame in list_frames(root):
        a, b = load_frame(root, frame), load_frame(root, frame, use_ego_motion=False)
        pts = finite_points(a)
        cam_a, cam_b = velo_to_cam(pts[:, :3], a["calib"]), velo_to_cam(pts[:, :3], b["calib"])
        _, _, ma = cam_to_image(cam_a, a["calib"].P2, a["image"].shape)
        _, _, mb = cam_to_image(cam_b, b["calib"].P2, b["image"].shape)
        both = ma & mb
        uva = cam_to_image(cam_a[both], a["calib"].P2, a["image"].shape)[0]
        uvb = cam_to_image(cam_b[both], b["calib"].P2, b["image"].shape)[0]
        shift, depth = np.linalg.norm(uva - uvb, axis=1), cam_a[both, 2]
        # góc xe quay giữa 2 thời điểm = góc của R_a·R_bᵀ (xe đang rẽ -> lệch giống hệt lỗi yaw calibration)
        R = a["calib"].Tr_velo_to_cam[:, :3] @ b["calib"].Tr_velo_to_cam[:, :3].T
        rot_deg = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        row = {"frame": frame, "dt_cam_minus_lidar_ms": (a["timestamp_camera_us"] - a["timestamp_lidar_us"]) / 1000,
               "ego_shift_m": round(float(np.linalg.norm(a["calib"].Tr_velo_to_cam[:, 3] - b["calib"].Tr_velo_to_cam[:, 3])), 3),
               "ego_rot_deg": round(rot_deg, 3),
               "inside_image_ego": int(ma.sum()), "inside_image_noego": int(mb.sum()),
               "px_shift_median": round(float(np.median(shift)), 2), "px_shift_p95": round(float(np.percentile(shift, 95)), 2)}
        for lo, hi in ((0, 10), (10, 20), (20, 40), (40, np.inf)):
            m = (depth >= lo) & (depth < hi)
            row[f"px_shift_median_{lo}_{hi if hi < np.inf else 'inf'}m"] = round(float(np.median(shift[m])), 2) if m.any() else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    df["ego_speed_mps"] = (df["ego_shift_m"] / (df["dt_cam_minus_lidar_ms"].abs() / 1000)).round(2)
    df.to_csv(res / "time_sync_nusc.csv", index=False)
    return df


def time_row(frame: str, min_points: int = 8) -> np.ndarray:
    """1 hàng ảnh: vật bị dời nhiều nhất trong frame, trái = có bù chuyển động, phải = không bù."""
    root = "data/nuscenes_mini_subset"
    a, b = load_frame(root, frame), load_frame(root, frame, use_ego_motion=False)
    pts = finite_points(a)
    cam_a, cam_b = velo_to_cam(pts[:, :3], a["calib"]), velo_to_cam(pts[:, :3], b["calib"])
    ma = cam_to_image(cam_a, a["calib"].P2, a["image"].shape)[2]
    mb = cam_to_image(cam_b, b["calib"].P2, b["image"].shape)[2]
    best = None
    for obj in a["labels"]:
        sel = points_in_box(cam_a, obj) & ma & mb        # điểm của vật, thấy được ở cả 2 cách chiếu
        if sel.sum() < min_points:
            continue
        uv_a = cam_to_image(cam_a[sel], a["calib"].P2, a["image"].shape)[0]
        uv_b = cam_to_image(cam_b[sel], b["calib"].P2, b["image"].shape)[0]
        shift = float(np.median(np.linalg.norm(uv_a - uv_b, axis=1)))
        if best is None or shift > best[0]:
            best = (shift, obj, uv_a, uv_b)
    shift, obj_a, uv_pa, uv_pb = best
    # 2 danh sách label có thể khác thứ tự/độ dài (box bị lọc khác nhau) -> ghép theo vị trí gần nhất, không theo chỉ số
    obj_b = min(b["labels"], key=lambda o: float(np.linalg.norm(o.location - obj_a.location)))
    dist = float(np.hypot(obj_a.location[0], obj_a.location[2]))
    center = ((obj_a.bbox[0] + obj_a.bbox[2]) / 2, (obj_a.bbox[1] + obj_a.bbox[3]) / 2)
    half = (max(110, 0.8 * (obj_a.bbox[2] - obj_a.bbox[0])), max(75, 0.8 * (obj_a.bbox[3] - obj_a.bbox[1])))
    dt = (a["timestamp_camera_us"] - a["timestamp_lidar_us"]) / 1000
    R = a["calib"].Tr_velo_to_cam[:, :3] @ b["calib"].Tr_velo_to_cam[:, :3].T
    rot = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    move = float(np.linalg.norm(a["calib"].Tr_velo_to_cam[:, 3] - b["calib"].Tr_velo_to_cam[:, 3]))
    panels = []
    for fr, obj, uv, name in ((a, obj_a, uv_pa, "ego-motion compensated"), (b, obj_b, uv_pb, "NO compensation")):
        own = object_uv(fr, obj, fr["calib"])          # hit_ratio như thí nghiệm: điểm + box của chính phiên bản đó
        ratio = 100 * in_box(own, obj.bbox).mean()
        canvas, to_zoom = crop_zoom(a["image"], center, half, target_w=560)
        zoomed_box(canvas, to_zoom, obj_a.bbox, GREEN)                  # box đúng trên ảnh
        if fr is b:
            zoomed_box(canvas, to_zoom, obj.bbox, CYAN, 1)              # box label tính lại khi không bù: dời theo điểm
        draw_hits(canvas, to_zoom(uv), np.ones(len(uv), bool), r=3)
        inside = int(project_velo_to_image(pts, fr["calib"], fr["image"].shape)[2].sum())
        panels.append(label(canvas, [f"{name}: inside_image = {inside}",
                                     f"points shifted by median {0 if fr is a else shift:.1f} px; "
                                     f"hit_ratio vs its label box = {ratio:.1f}%"], scale=0.5))
    row = np.hstack(panels)
    return label(row, [f"{frame}: camera {abs(dt):.1f} ms {'before' if dt < 0 else 'after'} LiDAR, ego moved {move:.2f} m "
                       f"and turned {rot:.2f} deg -> {obj_a.type} at {dist:.1f} m"], scale=0.55)


def fail_time(out: Path, frames=("scene-0103_010", "scene-1094_014")) -> None:
    rows = [time_row(f) for f in frames]
    width = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 4, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=BLACK) for r in rows]
    head = label(np.zeros((1, width, 3), np.uint8),
                 ["FAIL 02 (Time) - nuScenes, LiDAR projected without ego-motion compensation",
                  "green box = true object in image; cyan box = label recomputed without compensation",
                  "row 1: driving straight -> shift ~ 1/distance (near objects most); row 2: turning -> shift at ALL distances",
                  "points AND label box shift together -> hit_ratio stays 100%: it cannot see Time errors"], scale=0.5)
    cv2.imwrite(str(out / "fail_02_nusc_no_ego_motion.png"), np.vstack([head] + rows))


def main() -> None:
    ap = argparse.ArgumentParser(description="CP4: tạo 3 ảnh failure case + bảng lệch thời gian nuScenes")
    ap.add_argument("--results", default="results", help="thư mục kết quả")
    args = ap.parse_args()
    res = Path(args.results)
    fig = res / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    fail_geometry(fig)
    fail_metric(fig)
    fail_time(fig)
    df = time_sync_table(res)
    print(df.describe().loc[["mean", "min", "max"]].round(2).to_string())
    print(df[df.frame == "scene-0103_010"].T.to_string())
    print(f"-> {fig}/fail_01..03 *.png, {res}/time_sync_nusc.csv")


if __name__ == "__main__":
    main()
