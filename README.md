# Keyframe Extraction — HCM AI Challenge 2026

Trích keyframe từ video, sinh ảnh và metadata. Chạy local trên GPU, **không gọi API**.

## Cài đặt

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```

Cần GPU NVIDIA và `ffmpeg` trong PATH. Trọng số AutoShot đặt ở `models/autoshot_ckpt_0_200_0.pth`.

## Chạy

Đặt video vào `data/video/<thư mục>/` rồi:

```bash
python -u scripts/run_keyframes_only.py data/video/Videos_L26_a > logs/L26_a.log 2>&1
```

Đổi tên thư mục là chạy được bộ khác. Tham số đã đặt sẵn trong `config.yaml`, không cần chỉnh.

| Cờ | Tác dụng |
|---|---|
| `--only L26_V001` | chỉ chạy một video |
| `--limit 5` | chỉ chạy 5 video đầu |
| `--force` | chạy lại cả video đã xong |
| `--progress` | hiện thanh tiến trình |

Có resume: video đã xong tự bỏ qua, ngắt giữa chừng chạy lại được.

## Sau khi chạy

```bash
python -u scripts/export_keyframe_meta.py --merged data/output/keyframe_meta_all.jsonl
python -u scripts/group_keyframes_by_collection.py
python -u scripts/package_btc_format.py
```

Lần lượt: xuất metadata → gom thư mục ảnh theo bộ → đóng gói ra cấu trúc BTC.

## Đầu ra

```
data/output/
├── keyframes/<bộ>/<video_id>/*.jpg
└── <video_id>/
    ├── keyframes.jsonl        1 dòng = 1 keyframe
    ├── keyframe_meta.jsonl    schema chung
    ├── keyframe_meta.csv
    ├── shots.jsonl            mốc shot
    ├── events.jsonl           chuỗi sự kiện TRAKE
    └── video_meta.json

data/btc_format/               cấu trúc BTC
├── Keyframes_<bộ>/<video_id>/001.jpg
└── map-keyframes/<video_id>.csv    n, pts_time, fps, frame_idx
```

Schema metadata:

```json
{"video_id": "L26_V001", "keyframe_id": "L26_V001_kf000001",
 "shot_id": "L26_V001_shot00000", "frame_idx": 10, "fps": 25.0,
 "pts_time": 0.4, "keyframe_path": "..."}
```

`frame_idx` là vị trí thật trong video (khớp `cv2.CAP_PROP_POS_FRAMES`) — **đây là số nộp
cho ban tổ chức**, khác với `keyframe_id` vốn chỉ là số thứ tự.

Chi tiết phương pháp và số liệu đo: `docs/METHODS.md`.
