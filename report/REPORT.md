# Báo cáo Day 6: Calibration lệch xoay và lệch dịch ảnh hưởng thế nào tới projection LiDAR → camera

- **Họ tên:** Vũ Đức Thiện
- **MSSV:** 2A202602437
- **Lớp:** VinUni AI20K K4 · Track 4 (Computer Vision and Robotics)
- **Link repo:** https://github.com/vuthien3002-sys/VuDucThien-2A202602437-Track4-Day21
- **Topic:** A — LiDAR-camera projection QA
- **Dataset:** data/kitti_mini (chính), data/nuscenes_mini_subset (so sánh + lỗi thời gian), data/synthetic (debug)
- **Các frame đã dùng:** KITTI 000008, 000011, 000049 (thí nghiệm chính); 000004, 000019 (demo xa/gần); cả 20 frame kitti_mini cho bảng theo khoảng cách. nuScenes scene-0103_000…039 và scene-1094_000…039 (mỗi frame thứ 5)

## 1. Claim

**Claim nháp (CP1):** Lệch yaw 1° làm tỉ lệ điểm LiDAR của người đi bộ rơi đúng vào 2D box giảm hơn 20 điểm phần trăm, trong khi với xe con chỉ giảm dưới 5 điểm phần trăm. Ngược lại, lệch dịch 10 cm chủ yếu làm giảm tỉ lệ này ở vật gần (< 15 m), vật xa hơn 30 m gần như không đổi.

## 2. Evidence

Bảng hoặc plot số liệu, kèm ảnh/video demo. Ghi rõ đường dẫn file trong `results/`.

| Cấu hình / mức perturb | Metric 1 | Metric 2 | Ghi chú |
|---|---|---|---|
| [ĐIỀN] | | | |

![demo](../results/figures/[ĐIỀN].png)

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
[ĐIỀN]
```

## 6. Khai báo sử dụng AI

Ghi rõ đã dùng công cụ AI nào, dùng vào việc gì, và bạn đã tự kiểm chứng kết quả đó bằng cách nào. Nếu không dùng AI, ghi "Không sử dụng". Xem quy định ở `RULES.md` mục 2.

| Công cụ | Dùng cho việc gì | Bạn đã kiểm chứng thế nào |
|---|---|---|
| [ĐIỀN] | | |
