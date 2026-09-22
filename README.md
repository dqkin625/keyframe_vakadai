# Keyframe Extraction — HCM AI Challenge 2026

Trích keyframe từ video, sinh ảnh và metadata. Chạy local trên GPU, **không gọi API**.

## Cài đặt

```bash
git clone https://github.com/dqkin625/keyframe_vakadai.git
cd keyframe_vakadai
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```

**GPU NVIDIA**: bắt buộc. `config.yaml` để `gpu_decode: true` và AutoShot chạy trên CUDA.
Driver cũ (vd 528.x, chỉ tới CUDA 12.1) phải cài đúng bản torch cu121, nếu không sẽ lỗi
`cudaErrorDevicesUnavailable`:

```bash
.venv/Scripts/pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

**ffmpeg**: KHÔNG cần cài vào máy. `imageio-ffmpeg` trong `requirements.txt` đã đóng gói sẵn
ffmpeg trong venv.

**Trọng số AutoShot** (`models/autoshot_ckpt_0_200_0.pth`, 57 MB): KHÔNG cần tải thủ công.
Lần chạy đầu tiên `src/frame_sampling/shot_detector.py` tự tải về `models/` từ HuggingFace
và in ra `[AutoShot] tải weights lần đầu (~57MB)`. Thư mục `models/` nằm trong `.gitignore`
nên không có trong repo — đó là lý do phải tải.

Chỉ khi máy chặn mạng ra HuggingFace thì tải tay rồi đặt vào `models/`:

```bash
mkdir models
curl -L -o models/autoshot_ckpt_0_200_0.pth https://huggingface.co/backseollgi/AutoShot/resolve/main/ckpt_0_200_0.pth
```

File đúng phải nặng **57.243.097 byte**. Tải thiếu sẽ lỗi `PytorchStreamReader failed`.

**API key**: KHÔNG cần cho bước keyframe. `run_keyframes_only.py` dừng trước bước caption,
nên `.env` chỉ cần khi chạy caption Gemma (xem `.env.example`).

## Chạy

Đặt video vào `data/video/<thư mục>/` (thư mục này nằm trong `.gitignore`, phải tự tạo), rồi:

```bash
mkdir logs
python -u scripts/run_keyframes_only.py data/video/Videos_L26_a > logs/L26_a.log 2>&1
```

`mkdir logs` chỉ cần làm 1 lần — shell tạo được file log nhưng không tạo được thư mục chứa nó.

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
