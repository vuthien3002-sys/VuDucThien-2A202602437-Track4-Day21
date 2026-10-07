"""UI tương tác cho Topic A: kéo thanh trượt làm lệch calibration LiDAR -> camera, xem ngay điểm LiDAR
trượt khỏi vật và hit_ratio (mẫu số cố định, như src/exp_calib_sweep.py) thay đổi.

Chạy từ gốc repo (mở 1 cửa sổ, chỉ cần matplotlib có sẵn trong requirements.txt):
    python -m src.app
    python -m src.app --data-root data/nuscenes_mini_subset --frame scene-0103_010
Chụp 1 trạng thái ra file mà không mở cửa sổ:
    python -m src.app --frame 000011 --yaw 1 --mode hit --snapshot results/figures/ui_snapshot.png

Phím tắt: ← / → đổi frame, m đổi chế độ màu (độ sâu / trúng-trượt), r đặt lại calibration.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

from src.exp_calib_sweep import CLASS_GROUP, measure, prepare_frame
from starter.datasets import dataset_type, list_frames, load_frame
from starter.projection import perturb_extrinsic, project_velo_to_image

DATASETS = {"KITTI": "data/kitti_mini", "nuScenes": "data/nuscenes_mini_subset", "synthetic": "data/synthetic"}
SLIDERS = [("yaw", -3, 3, 0.05, "°"), ("pitch", -3, 3, 0.05, "°"), ("roll", -3, 3, 0.05, "°"),
           ("tx", -0.3, 0.3, 0.01, "m"), ("ty", -0.3, 0.3, 0.01, "m"), ("tz", -0.3, 0.3, 0.01, "m")]
THRESHOLD = 0.95                  # ngưỡng cảnh báo theo frame, chọn ở REPORT mục 2 (báo nhầm 1/20 frame KITTI)
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, GOOD, WARN, BAD = "#2a78d6", "#1baf7a", "#eda100", "#e34948"
GROUP_NAME = {"vehicle": "Xe", "pedestrian": "Người đi bộ", "cyclist": "Xe đạp"}


def ratio_color(r: float) -> str:
    return INK_2 if np.isnan(r) else GOOD if r >= 0.9 else WARN if r >= 0.5 else BAD


def pct(r: float) -> str:
    return "—" if np.isnan(r) else f"{100 * r:.1f} %"


class CalibLab:
    def __init__(self, plt, root: str, frame: str | None, init: dict, mode: str):
        from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

        self.plt, self.root, self.mode, self.ego = plt, root, mode, True
        self.frames = list_frames(root)
        self.i = self.frames.index(frame) if frame in self.frames else 0
        self.fig = plt.figure(figsize=(15, 8.6), facecolor=SURFACE)
        self.fig.canvas.manager.set_window_title("Day 6 - LiDAR-camera calibration lab")
        self.ax_img = self.fig.add_axes([0.01, 0.34, 0.68, 0.60])
        self.ax_curve = self.fig.add_axes([0.745, 0.62, 0.235, 0.30], facecolor=SURFACE)
        self.ax_text = self.fig.add_axes([0.715, 0.27, 0.28, 0.30])
        self.ax_text.axis("off")

        self.sliders = {}
        for k, (name, lo, hi, step, unit) in enumerate(SLIDERS):
            col, row = divmod(k, 3)
            ax = self.fig.add_axes([0.07 + 0.34 * col, 0.235 - 0.055 * row, 0.25, 0.03], facecolor=GRID)
            s = Slider(ax, f"{name} ({unit})", lo, hi, valinit=init.get(name, 0.0), valstep=step, color=BLUE)
            s.on_changed(lambda _: self.update())
            self.sliders[name] = s
        self.fig.text(0.07, 0.275, "Xoay LiDAR trong velodyne frame (perturb_extrinsic)", color=INK_2, fontsize=9)
        self.trans_note = self.fig.text(0.41, 0.275, "", color=INK_2, fontsize=9)
        self.fig.text(0.01, 0.315, "Kéo thanh trượt để làm lệch calibration  ·  phím ← / → đổi frame  ·  m đổi chế độ màu"
                      "  ·  r đặt lại calib", color=INK_2, fontsize=9)

        self.buttons = []
        for x, label, fn in ((0.07, "◀ Frame trước", lambda: self.step(-1)), (0.19, "Frame sau ▶", lambda: self.step(1)),
                             (0.31, "Đặt lại calib (r)", self.reset)):
            b = Button(self.fig.add_axes([x, 0.04, 0.1, 0.05]), label, color=GRID, hovercolor="#d0cfca")
            b.on_clicked(lambda _, f=fn: f())
            self.buttons.append(b)
        self.radio = RadioButtons(self.fig.add_axes([0.44, 0.015, 0.1, 0.095], facecolor=SURFACE),
                                  list(DATASETS), active=list(DATASETS.values()).index(root) if root in DATASETS.values() else 0)
        self.radio.on_clicked(self.change_dataset)
        self.check = CheckButtons(self.fig.add_axes([0.56, 0.015, 0.2, 0.095], facecolor=SURFACE),
                                  ["Tô trúng / trượt box (m)", "Bù chuyển động (nuScenes)"], [mode == "hit", True])
        self.check.on_clicked(self.toggle)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.load()

    # ---------- dữ liệu ----------
    def load(self) -> None:
        frame = self.frames[self.i]
        kwargs = {"use_ego_motion": self.ego} if dataset_type(self.root) == "nuscenes" else {}
        self.fr = load_frame(self.root, frame, **kwargs)
        self.pts = self.fr["points"][np.isfinite(self.fr["points"]).all(axis=1)]
        self.prep = prepare_frame(self.fr, set(CLASS_GROUP))
        self.shape = self.fr["image"].shape

        ax = self.ax_img
        ax.clear()
        ax.imshow(self.fr["image"][:, :, ::-1])            # BGR -> RGB
        ax.set_axis_off()
        ax.set_anchor("N")                                 # ảnh KITTI dẹt: căn lên trên, thẳng hàng với biểu đồ
        axes_note = ("nuScenes: x phải, y trước, z lên" if dataset_type(self.root) == "nuscenes"
                     else "KITTI: x trước, y trái, z lên")
        self.trans_note.set_text(f"Dịch LiDAR (m) theo trục LiDAR - {axes_note}")
        self.scat = ax.scatter([], [], s=1.5, c=[], cmap="jet_r", vmin=0, vmax=50, linewidths=0)
        self.scat_obj = ax.scatter([], [], s=9, linewidths=0.4, edgecolors="black")
        self.box_artists = []

        # Đường hit_ratio theo yaw của riêng frame này (các trục khác = 0) để thấy vị trí hiện tại trên đường cong
        self.yaw_grid = np.linspace(-3, 3, 25)
        self.yaw_curve = [self.overall(measure(self.prep, perturb_extrinsic(self.fr["calib"], yaw_deg=y), self.shape))
                          for y in self.yaw_grid]
        c = self.ax_curve
        c.clear()
        c.plot(self.yaw_grid, 100 * np.array(self.yaw_curve), color=BLUE, linewidth=2, label="chỉ lệch yaw")
        c.axhline(100 * THRESHOLD, color=BAD, linestyle="--", linewidth=1.2, label=f"ngưỡng {100 * THRESHOLD:.0f} %")
        self.cur_line = c.axvline(0, color=INK_2, linewidth=1)
        (self.cur_dot,) = c.plot([], [], "o", color=INK, markersize=8, label="hiện tại")
        c.set_xlim(-3.1, 3.1)
        c.set_ylim(0, 105)
        c.set_xlabel("yaw (°)")
        c.set_ylabel("hit_ratio (%)")
        c.set_title(f"hit_ratio của frame {frame} theo yaw", loc="left", fontsize=10)
        c.grid(color=GRID)
        c.legend(loc="lower center", fontsize=8, frameon=False)
        self.update()

    @staticmethod
    def overall(rows: list[dict], group: str | None = None) -> float:
        rows = [r for r in rows if group is None or r["group"] == group]
        n = sum(r["n_points_fixed"] for r in rows)
        return sum(r["hits_fixed"] for r in rows) / n if n else float("nan")

    # ---------- vẽ ----------
    def update(self) -> None:
        v = {k: s.val for k, s in self.sliders.items()}
        calib = perturb_extrinsic(self.fr["calib"], v["roll"], v["pitch"], v["yaw"], (v["tx"], v["ty"], v["tz"]))
        uv, depth, mask = project_velo_to_image(self.pts, calib, self.shape)
        rows = measure(self.prep, calib, self.shape)

        if self.mode == "depth":
            self.scat.set_offsets(uv)
            self.scat.set_array(depth)
            self.scat.set_alpha(1.0)
            self.scat_obj.set_offsets(np.empty((0, 2)))
        else:                                                  # điểm nền mờ, điểm của vật: xanh = trong box, đỏ = trượt
            self.scat.set_offsets(uv)
            self.scat.set_array(depth)
            self.scat.set_alpha(0.15)
            uv_all = np.full((len(self.pts), 2), np.nan)
            uv_all[mask] = uv
            xy, col = [], []
            for o in self.prep["objects"]:
                p = uv_all[o["idx_visible"]]
                p = p[np.isfinite(p).all(axis=1)]
                x1, y1, x2, y2 = o["bbox"]
                hit = (p[:, 0] >= x1) & (p[:, 0] <= x2) & (p[:, 1] >= y1) & (p[:, 1] <= y2)
                xy.append(p)
                col += [GOOD if h else BAD for h in hit]
            self.scat_obj.set_offsets(np.vstack(xy) if xy else np.empty((0, 2)))
            self.scat_obj.set_facecolors(col)

        for a in self.box_artists:
            a.remove()
        self.box_artists = []
        for o, r in zip(self.prep["objects"], rows):
            ratio = r["hits_fixed"] / r["n_points_fixed"] if r["n_points_fixed"] else float("nan")
            x1, y1, x2, y2 = o["bbox"]
            color = ratio_color(ratio)
            rect = self.plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.8)
            self.ax_img.add_patch(rect)
            txt = self.ax_img.text(x1, y1 - 3, f"{o['type']} {o['distance_m']:.0f}m {pct(ratio)}", fontsize=7,
                                   color="white", va="bottom", bbox={"facecolor": color, "edgecolor": "none", "pad": 1})
            self.box_artists += [rect, txt]

        total = self.overall(rows)
        f = calib.P2[0, 0]
        ok = np.isnan(total) or total >= THRESHOLD
        lines = [
            f"{Path(self.root).name} / {self.frames[self.i]}   ({self.i + 1}/{len(self.frames)})",
            f"Lệch: yaw {v['yaw']:+.2f}°  pitch {v['pitch']:+.2f}°  roll {v['roll']:+.2f}°",
            f"      t = ({v['tx']:+.2f}, {v['ty']:+.2f}, {v['tz']:+.2f}) m",
            f"Điểm trong ảnh: {int(mask.sum()):,} / {len(self.pts):,}".replace(",", " "),
            f"Dời do yaw: f·tan(yaw) = {f * np.tan(np.deg2rad(abs(v['yaw']))):.1f} px",
            "",
            f"hit_ratio cả frame: {pct(total)}",
        ] + [f"  {GROUP_NAME[g]:<12}{pct(self.overall(rows, g))}" for g in GROUP_NAME]
        if dataset_type(self.root) == "nuscenes":
            dt = (self.fr["timestamp_camera_us"] - self.fr["timestamp_lidar_us"]) / 1000
            lines.append(f"camera - LiDAR = {dt:+.1f} ms, bù chuyển động: {'BẬT' if self.ego else 'TẮT'}")
        self.ax_text.clear()
        self.ax_text.axis("off")
        self.ax_text.text(0, 1, "\n".join(lines), va="top", family="DejaVu Sans Mono", fontsize=9.5, color=INK,
                          linespacing=1.5)
        status = "✓ Calibration ổn" if ok else f"⚠ CẢNH BÁO: hit_ratio < {100 * THRESHOLD:.0f} % → calib lệch?"
        self.ax_text.text(0, -0.02, status, va="top", fontsize=10.5, color=INK, weight="bold",
                          bbox={"facecolor": GOOD if ok else BAD, "alpha": 0.25, "edgecolor": "none", "pad": 5})
        self.cur_line.set_xdata([v["yaw"], v["yaw"]])
        self.cur_dot.set_data([v["yaw"]], [100 * total] if not np.isnan(total) else [np.nan])
        self.ax_img.set_title(f"{'Màu theo độ sâu (đỏ = gần, xanh = xa)' if self.mode == 'depth' else 'Điểm của vật: xanh = trong 2D box, đỏ = trượt ra ngoài'}"
                              "   |   khung: xanh ≥ 90 %, vàng 50–90 %, đỏ < 50 %", fontsize=9, color=INK_2, loc="left")
        self.fig.canvas.draw_idle()

    # ---------- điều khiển ----------
    def step(self, d: int) -> None:
        self.i = (self.i + d) % len(self.frames)
        self.load()

    def reset(self) -> None:
        for s in self.sliders.values():
            s.eventson = False
            s.set_val(0.0)
            s.eventson = True
        self.update()

    def change_dataset(self, label: str) -> None:
        self.root = DATASETS[label]
        self.frames, self.i = list_frames(self.root), 0
        self.load()

    def toggle(self, label: str) -> None:
        if label.startswith("Tô"):
            self.mode = "hit" if self.mode == "depth" else "depth"
            self.update()
        else:
            self.ego = not self.ego
            if dataset_type(self.root) == "nuscenes":
                self.load()

    def on_key(self, event) -> None:
        if event.key == "left":
            self.step(-1)
        elif event.key == "right":
            self.step(1)
        elif event.key == "r":
            self.reset()
        elif event.key == "m":
            self.check.set_active(0)                       # gọi lại toggle() qua callback của CheckButtons


def main() -> None:
    ap = argparse.ArgumentParser(description="UI tương tác: làm lệch calibration LiDAR-camera và xem hit_ratio thay đổi")
    ap.add_argument("--data-root", default="data/kitti_mini", choices=list(DATASETS.values()), help="dataset mở đầu tiên")
    ap.add_argument("--frame", default="000011", help="frame mở đầu tiên (không có trong dataset -> frame đầu)")
    for name, lo, hi, _, unit in SLIDERS:
        ap.add_argument(f"--{name}", type=float, default=0.0, help=f"giá trị ban đầu của {name} ({unit}, {lo}..{hi})")
    ap.add_argument("--mode", choices=["depth", "hit"], default="depth", help="tô màu theo độ sâu hay trúng/trượt box")
    ap.add_argument("--snapshot", default=None, help="lưu 1 ảnh PNG rồi thoát, không mở cửa sổ")
    args = ap.parse_args()

    if args.snapshot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    init = {name: getattr(args, name) for name, *_ in SLIDERS}
    app = CalibLab(plt, args.data_root, args.frame, init, args.mode)
    if args.snapshot:
        Path(args.snapshot).parent.mkdir(parents=True, exist_ok=True)
        app.fig.savefig(args.snapshot, dpi=110, facecolor=SURFACE)
        print(f"-> {args.snapshot}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
