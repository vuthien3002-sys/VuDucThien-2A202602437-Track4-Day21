# Báo cáo Day 6: Calibration lệch xoay và lệch dịch ảnh hưởng thế nào tới projection LiDAR → camera

- **Họ tên:** Vũ Đức Thiện
- **MSSV:** 2A202602437
- **Lớp:** VinUni AI20K K4 · Track 4 (Computer Vision and Robotics)
- **Link repo:** https://github.com/vuthien3002-sys/VuDucThien-2A202602437-Track4-Day21
- **Topic:** A — LiDAR-camera projection QA
- **Dataset:** data/kitti_mini (chính), data/nuscenes_mini_subset (so sánh + lỗi thời gian), data/synthetic (debug)
- **Các frame đã dùng:** KITTI 000008, 000011, 000049 (thí nghiệm chính); 000004, 000019 (demo xa/gần); cả 20 frame kitti_mini cho bảng theo class/khoảng cách. nuScenes: cả 80 keyframe scene-0103_000…039 (ngày) và scene-1094_000…039 (đêm), failure Time ở scene-0103_010

## 1. Claim

**Claim nháp (CP1):** Lệch yaw 1° làm tỉ lệ điểm LiDAR của người đi bộ rơi đúng vào 2D box giảm hơn 20 điểm phần trăm, trong khi với xe con chỉ giảm dưới 5 điểm phần trăm. Ngược lại, lệch dịch 10 cm chủ yếu làm giảm tỉ lệ này ở vật gần (< 15 m), vật xa hơn 30 m gần như không đổi.

## 2. Evidence

**Demo (CP2).** Overlay điểm LiDAR (màu theo độ sâu) + 2D box của label ở 3 frame gần / trung bình / xa. Hàm chiếu tự viết khớp đúng số kỳ vọng: `inside_image` = 3910 / 19946 / 3120 điểm (synthetic / KITTI 000011 / nuScenes scene-0103_010).

![demo](../results/figures/demo_overlay_3_distances.png)

**Metric.** `hit_ratio` = (số điểm trong 3D box GT, chọn bằng calib **gốc**) rơi vào 2D box của vật khi chiếu bằng calib **đã lệch** ÷ (số điểm đó nằm trong ảnh theo calib gốc). Mỗi lần chỉ lệch **1 trục** (`perturb_extrinsic`, quay/dịch trong velodyne frame), giữ nguyên frame, class (Car, Van, Pedestrian, Cyclist), range. Không có phép ngẫu nhiên → chạy lại 2 lần ra CSV giống từng byte (đã kiểm `filecmp`).

**Kiểm tra cài đặt:** chạy script mẫu `src/exp_yaw_sweep.py` trên 000008 / 000011 / 000049 ra **đúng 15/15 số** của bảng kỳ vọng (vd 000011: 99.45 → 91.88 → 77.44 → 45.44 → 21.23 %) → `results/yaw_perturb_sweep.csv`, [yaw_sweep.png](../results/figures/yaw_sweep.png).

**Kết quả chính (KITTI, 20 frame, 113 vật)** — `results/calib_sweep_summary_kitti_mini.csv`:

| Lệch yaw | Xe (Car/Van, 89 vật) | Người đi bộ (18 vật) | Tất cả |
|---|---|---|---|
| 0° | 99.9 % | 95.7 % | 99.6 % |
| 0.5° | 98.4 % | 81.6 % | 97.2 % |
| **1°** | **95.4 %** (−4.5 điểm) | **59.5 %** (−36.2 điểm) | 92.9 % |
| 2° | 87.9 % | 30.2 % | 84.0 % |
| 3° | 80.1 % | 18.6 % | 76.0 % |

| Theo khoảng cách (tất cả vật) | 0–15 m (32 vật) | 15–30 m (41) | > 30 m (40) |
|---|---|---|---|
| calib đúng | 99.6 % | 99.5 % | 99.7 % |
| xoay yaw 1° | 95.8 % | 88.4 % | **70.0 %** |
| xoay pitch 1° | 94.3 % | 89.3 % | **65.6 %** |
| dịch ngang ty 10 cm | 97.8 % | 98.4 % | 98.9 % |
| dịch đứng tz 10 cm | 97.2 % | 98.1 % | 97.7 % |
| dịch ngang ty 10 cm, chỉ người đi bộ | 89.1 % | 83.6 % | 85.4 % |

![claim](../results/figures/claim_rot_vs_trans.png)

- Lệch xoay càng hại khi vật càng **xa**: độ trượt `f·tanθ` ≈ 12.6 px không đổi theo khoảng cách, còn bề rộng vật `f·W/z` co lại → tỉ lệ trượt/bề rộng = `z·tanθ/W` tăng tuyến tính theo z. Người đi bộ (W ≈ 0.6 m, box trung vị 28 px) hỏng nặng hơn xe (W ≈ 1.6 m + chiều dài).
- Lệch dịch **không** chủ yếu hại vật gần như claim nháp dự đoán: cả độ trượt `f·d/z` lẫn bề rộng `f·W/z` đều tỉ lệ 1/z → tỉ lệ trượt = `d/W`, **không phụ thuộc khoảng cách** (người đi bộ: 89 / 84 / 85 %). Pixel trượt lớn ở vật gần là đúng, nhưng vật gần cũng to tương ứng.
- Yaw hại vật **hẹp**, pitch hại vật **thấp**: pitch 1° làm xe giảm còn 92.2 % (nặng hơn yaw), người chỉ còn 88.0 %. Roll 1° (98.3 %) và dịch dọc tx (99.7 %) gần như vô hại.

**Ngưỡng phát hiện drift (Advanced)** — `results/drift_detection_kitti.csv`, [drift_detection.png](../results/figures/drift_detection.png). Mỗi frame tính 1 `hit_ratio`, cảnh báo khi < ngưỡng:

| Ngưỡng | Báo nhầm ở 0° | Phát hiện yaw 0.5° | yaw 1° | yaw 2° | yaw 3° |
|---|---|---|---|---|---|
| 95 % | 1/20 (frame 000048, mức sàn 92.0 %) | 10/20 | 14/20 | 20/20 | 20/20 |
| 90 % | 0/20 | 6/20 | 10/20 | 16/20 | 20/20 |

6 frame **không phát hiện được yaw 1°** ở ngưỡng 95 % (000008, 000010, 000019, 000021, 000031, 000032) đều là frame chỉ có xe gần/rộng (đông xe, vật < 6 m, van/truck): tỉ lệ gộp theo điểm bị xe gần chi phối. Mức sàn ở 0° chưa phải 100 % vì 2D box do người vẽ không khớp tuyệt đối với 3D box (người đi bộ 95.7 %).

**[B5] So sánh KITTI và nuScenes** (cùng code, cùng metric, 80 keyframe nuScenes, 1146 vật) — `results/calib_sweep_summary_nusc.csv`, [kitti_vs_nuscenes.png](../results/figures/kitti_vs_nuscenes.png):

| | KITTI | nuScenes |
|---|---|---|
| Tiêu cự / ảnh | 721.5 px, 1242×375 | 1253–1266 px, 1600×900 |
| Điểm LiDAR / vật có điểm (trung vị) | 113 (64 beam) | 4 (32 beam) |
| calib đúng: tất cả / người đi bộ | 99.6 % / 95.7 % | 99.9 % / 99.9 % |
| yaw 1°: tất cả / xe / người | 92.9 / 95.4 / 59.5 % | 92.2 / 95.2 / 61.5 % |
| xoay quanh trục ngang 1° (KITTI `pitch` = nuScenes `roll`) | 91.8 % | 95.1 % |
| dịch ngang 10 cm, người đi bộ (KITTI `ty` = nuScenes `tx`) | 87.2 % | 97.4 % |
| yaw 1° người: ngày (scene-0103) / đêm (scene-1094) | — | 48.6 % / 68.6 % |

Vì sao giống/khác: (1) yaw 1° gần như **trùng nhau** dù nuScenes trượt 1253·tan1° ≈ 21.9 px (KITTI 12.6 px): ảnh nuScenes phóng vật lớn theo đúng tỉ lệ f, nên f **bị triệt tiêu** (`z·tanθ/W`). (2) Mức sàn nuScenes ≈ 100 % và người đi bộ chịu dịch tốt hơn (97.4 % so với 87.2 %) vì 2D box nuScenes **không do người vẽ** mà là bao của 8 góc 3D box chiếu lên ảnh → rộng hơn dáng người, có lề. (3) Trục LiDAR nuScenes là x-phải, y-trước nên `pitch` trong code của nuScenes thực chất là roll của xe; phải ghép đúng trục vật lý mới so sánh được. (4) Đêm/ngày không làm metric kém đi (LiDAR là cảm biến chủ động); khác biệt 48.6 / 68.6 % đến từ khoảng cách: người đi bộ ban đêm gần hơn (khoảng cách trung bình theo số điểm 19.4 m so với 25.5 m ban ngày). (5) nuScenes chỉ 4 điểm/vật → tỉ lệ theo từng vật rất nhiễu, phải gộp nhiều vật.

**[B2] Stress test suy giảm dữ liệu** (2 loại × 3 mức, seed = 0) — `results/degradation_summary.csv`, [degradation_sweep.png](../results/figures/degradation_sweep.png): giữ 30 % điểm (`random_dropout`) hay thêm nhiễu σ = 10 cm (`gaussian_noise`), `hit_ratio` gộp vẫn 99.6 / 99.4 % ở 0° và 93.1 / 92.8 % ở yaw 1°; tỉ lệ frame bị cờ không đổi (5 % ở 0°, 70 % ở 1°). Metric là **tỉ lệ** nên bỏ điểm không làm lệch nó; nhiễu chỉ làm mẫu số giảm 40 105 → 35 642 điểm. Kết luận: bài kiểm tra calibration bền với mưa/bụi/nhiễu, giới hạn thật nằm ở **nội dung frame** (có vật hẹp/xa hay không).

**[B3] Latency** (CPU Intel i7-8850H, 12 luồng, 15.8 GB RAM, không GPU; 50 lần đo, bỏ lần đầu) — `results/latency_qa.csv`, `results/latency_summary.csv`, `results/latency_hardware.txt`:

| Frame | Số điểm | Chỉ projection p50 / p95 | Cả bài kiểm tra 1 frame p50 / p95 |
|---|---|---|---|
| KITTI 000011 | 108 004 | 12.3 / 18.6 ms | 54.4 / 61.7 ms |
| nuScenes scene-0103_010 | 34 720 | 2.6 / 3.0 ms | 21.4 / 24.7 ms |

**[B4] Tool dùng lại được:** `python -m src.exp_calib_sweep --help` — quét 6 trục (`--axes yaw pitch roll tx ty tz`), mức lệch (`--rot-levels`, `--trans-levels`), dataset bất kỳ (`--data-root`), tắt bù chuyển động (`--ignore-ego-motion`); mặc định chạy được ngay. `src/exp_degradation.py`, `src/bench_latency.py`, `src/demo_overlays.py` cũng có `--help`.

## 3. Failure case

Nêu khi nào hệ thống hoặc phương pháp fail, vì sao fail, và liên hệ tới lớp nào trong 6 lớp debug: I/O, Geometry, Time, Preprocess, Model, Metric.

![failure](../results/figures/fail_[ĐIỀN].png)

[ĐIỀN]

## 4. Khuyến nghị nếu triển khai thật

Use-case cụ thể (ADAS / robot / drone), trade-off và bước tiếp theo.

[ĐIỀN]

## 5. Cách chạy lại

Các lệnh tái tạo lại toàn bộ kết quả từ repo sạch.

```bash
# 0. Môi trường (Windows máy GBK cần bật UTF-8 mode, nếu không pip đọc requirements.txt bị lỗi)
python -m venv .venv && .venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
set PYTHONUTF8=1                                         # PowerShell: $env:PYTHONUTF8=1 ; bash: export PYTHONUTF8=1
pip install -r requirements.txt

# 1. CP2: tự kiểm 2 hàm TODO + overlay (in đúng inside_image = 3910 / 19946 / 3120)
python -m src.test_projection
python -m starter.projection --data-root data/synthetic --frame 000000
python -m starter.projection --data-root data/kitti_mini --frame 000011
python -m starter.projection --data-root data/nuscenes_mini_subset --frame scene-0103_010
python -m src.demo_overlays          # -> results/figures/demo_overlay_3_distances.png (000019 gần, 000011, 000004 xa)

# 2. CP3: thí nghiệm chính (tất định, chạy lại ra cùng CSV)
python -m src.exp_yaw_sweep --data-root data/kitti_mini --frames 000008 000011 000049   # script mẫu -> yaw_perturb_sweep.csv
python -m src.exp_calib_sweep                                                         # KITTI 20 frame × 6 trục (~15 s)
python -m src.exp_calib_sweep --data-root data/nuscenes_mini_subset --tag nusc        # [B5] nuScenes 80 frame (~15 s)
python -m src.plot_results           # -> yaw_sweep.png, claim_rot_vs_trans.png, kitti_vs_nuscenes.png, drift_detection.png
python -m src.exp_degradation        # [B2] -> degradation_summary.csv, degradation_sweep.png (seed = 0)
python -m src.bench_latency          # [B3] -> latency_qa.csv, latency_summary.csv (số ms thay đổi nhẹ theo máy)
```

## 6. Khai báo sử dụng AI

Ghi rõ đã dùng công cụ AI nào, dùng vào việc gì, và bạn đã tự kiểm chứng kết quả đó bằng cách nào. Nếu không dùng AI, ghi "Không sử dụng". Xem quy định ở `RULES.md` mục 2.

| Công cụ | Dùng cho việc gì | Bạn đã kiểm chứng thế nào |
|---|---|---|
| [ĐIỀN] | | |
