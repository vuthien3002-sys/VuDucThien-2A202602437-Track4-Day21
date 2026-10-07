"""Tự kiểm tra 2 hàm TODO(CP2). Chạy từ gốc repo: python -m src.test_projection

Nguồn: script mẫu trong Lab Guide Day 6, Phần 04 mục 4.3 (giữ nguyên 4 điểm test và các assert).
"""
import numpy as np

from starter.datasets import load_frame
from starter.projection import cam_to_image, velo_to_cam

np.set_printoptions(suppress=True, precision=2)   # in số thường, không in dạng 1.0e+01
fr = load_frame("data/synthetic", "000000")
calib, shape = fr["calib"], fr["image"].shape

pts = np.array([
    [10.0, 0.0, 0.0],       # 10 m phía trước      -> phải hợp lệ
    [np.nan, 0.0, 0.0],     # điểm lỗi NaN         -> phải bị loại
    [-10.0, 0.0, 0.0],      # 10 m phía sau xe     -> phải bị loại (z_cam < 0)
    [10.0, 50.0, 0.0],      # 50 m bên trái        -> phải bị loại (ngoài khung hình)
])
cam = velo_to_cam(pts, calib)
uv, depth, mask = cam_to_image(cam, calib.P2, shape)

print("camera frame:\n", np.round(cam, 2))
print("uv:", np.round(uv, 1), " depth:", np.round(depth, 2), " mask:", mask)

assert cam.shape == (4, 3), f"velo_to_cam phai tra ve (N, 3), dang la {cam.shape}"
assert abs(cam[0, 2] - 9.73) < 0.01, f"z_cam cua (10,0,0) phai ~ 9.73, dang la {cam[0, 2]}"
assert mask.tolist() == [True, False, False, False], f"mask sai: {mask}"
assert uv.shape == (1, 2) and depth.shape == (1,), "uv phai (M, 2), depth phai (M,)"
assert np.allclose(uv[0], [614, 175], atol=1), f"pixel cua (10,0,0) phai ~ (614, 175), dang la {uv[0]}"
print("CP2 self-test passed")
