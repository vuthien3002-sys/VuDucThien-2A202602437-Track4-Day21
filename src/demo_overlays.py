"""Demo CP2 topic A: overlay điểm LiDAR lên ảnh ở 3 frame có vật ở 3 khoảng cách khác nhau.

Mỗi 2D box được ghi thêm khoảng cách của vật (m), để thấy rõ frame gần / trung bình / xa.
Chạy từ gốc repo:
    python -m src.demo_overlays
    python -m src.demo_overlays --data-root data/kitti_mini --frames 000019 000011 000004
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from starter.datasets import load_frame
from starter.projection import draw_box2d, overlay_points, project_velo_to_image


def object_distance(obj) -> float:
    """Khoảng cách ngang (m) từ camera tới tâm đáy box: sqrt(x^2 + z^2) trong camera frame."""
    return float(np.hypot(obj.location[0], obj.location[2]))


def render(data_root: str, frame: str, width: int) -> tuple[np.ndarray, list[float]]:
    fr = load_frame(data_root, frame)
    uv, depth, mask = project_velo_to_image(fr["points"], fr["calib"], fr["image"].shape)
    vis = overlay_points(fr["image"], uv, depth)
    dists = []
    for obj in fr["labels"]:
        d = object_distance(obj)
        dists.append(d)
        vis = draw_box2d(vis, obj.bbox, label=f"{obj.type} {d:.0f}m")
    title = (f"{Path(data_root).name} {frame}: inside_image={int(mask.sum())}/{len(mask)}"
             f"  objects {min(dists):.0f}-{max(dists):.0f} m" if dists else f"{frame}")
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(vis, (0, 0), (tw + 20, th + 18), (0, 0, 0), -1)   # nền đen cho dễ đọc
    cv2.putText(vis, title, (10, th + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    scale = width / vis.shape[1]                     # ảnh KITTI không cùng kích thước tuyệt đối
    return cv2.resize(vis, (width, int(round(vis.shape[0] * scale)))), dists


def main() -> None:
    ap = argparse.ArgumentParser(description="Overlay LiDAR lên ảnh ở nhiều frame, xếp chồng thành 1 ảnh demo")
    ap.add_argument("--data-root", default="data/kitti_mini", help="thư mục dataset (KITTI hoặc nuScenes)")
    ap.add_argument("--frames", nargs="+", default=["000019", "000011", "000004"],
                    help="danh sách frame, mặc định: gần (000019), trung bình (000011), xa (000004)")
    ap.add_argument("--width", type=int, default=1242, help="bề rộng mỗi ảnh sau khi resize (pixel)")
    ap.add_argument("--out", default="results/figures/demo_overlay_3_distances.png", help="file ảnh đầu ra")
    args = ap.parse_args()

    tiles = []
    for frame in args.frames:
        tile, dists = render(args.data_root, frame, args.width)
        tiles.append(tile)
        print(f"{frame}: {len(dists)} objects, distance {min(dists):.1f}-{max(dists):.1f} m")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(tiles))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
