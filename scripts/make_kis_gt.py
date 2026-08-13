"""
Lấy mẫu 100 keyframe BTC -> khung ground truth cho truy vấn KIS (tự chấm).

    python scripts/make_kis_gt.py                    # 100 mẫu, seed 42
    python scripts/make_kis_gt.py --n 60 --seed 7

Ý tưởng: LÀM NGƯỢC CHIỀU. Chọn đáp án (một keyframe cụ thể) TRƯỚC, viết truy vấn SAU
(bước 2, `gen_kis_queries.py`). Nhờ vậy ground truth có sẵn theo cách xây dựng, không phải
đi dò 873 video để tìm đáp án.

Ràng buộc khi chọn mẫu:
  - MỖI VIDEO TỐI ĐA 1 MẪU  -> hai truy vấn không bao giờ đụng nhau, giảm ca "trúng đoạn khác
    cũng đúng" mà bộ chấm tự động không xử lý được.
  - BỎ 10% đầu và 10% cuối mỗi video -> né intro/outro/logo/màn hình đen của bản tin.
  - LỌC ẢNH XẤU: quá tối, quá đơn điệu (std thấp) -> không viết nổi truy vấn phân biệt được.

Khoảng đáp án [s,e]: BTC KHÔNG công bố con số cố định (xem docs/METHODS.md §KIS-GT).
Ví dụ trong thể lệ: KIS [500,510] = 11 frame, Q&A [800,900] = 101 frame -> lệch 10 lần.
=> Ghi ra NHIỀU mức dung sai, để bộ chấm báo cáo song song thay vì cược vào một mức.

Output: data/eval/kis_gt.json  (trường `query` để rỗng, bước 2 điền)
"""
import argparse
import csv
import json
import os
import random
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
try:                                   # log tiếng Việt ra file/console Windows (cp1252)
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

KEYFRAME_DIR = "data/keyframes"
MAP_DIR = "data/map-keyframes-aic25-b1/map-keyframes"
VIDEO_ROOT = "data/video"
OUT_PATH = "data/eval/kis_gt.json"

TOLERANCES = {"tol5": 5, "tol30": 30, "tol90": 90}

# Ảnh bị loại nếu độ lệch chuẩn mức xám dưới ngưỡng này (gần như một màu: đen, trắng, fade).
MIN_STD = 28.0
# ...hoặc trung bình mức xám quá tối / quá sáng.
MIN_MEAN, MAX_MEAN = 25.0, 235.0
MAX_TRY_PER_VIDEO = 10


def _log(msg: str) -> None:
    print(msg, flush=True)


def _read_map(video_id: str):
    """Đọc map-keyframes/<video_id>.csv -> list dict(n, pts_time, fps, frame_idx)."""
    path = os.path.join(MAP_DIR, f"{video_id}.csv")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "n": int(r["n"]),
                    "pts_time": float(r["pts_time"]),
                    "fps": float(r["fps"]),
                    "frame_idx": int(r["frame_idx"]),
                })
            except (KeyError, ValueError):
                continue
    return rows or None


def _video_path(video_id: str):
    """Đường dẫn .mp4 nếu đã tải về (chỉ một phần kho có sẵn video)."""
    batch = f"Videos_{video_id.split('_')[0]}_a"
    p = os.path.join(VIDEO_ROOT, batch, f"{video_id}.mp4")
    return p if os.path.exists(p) else None


def _usable(img_path: str) -> bool:
    """Ảnh có đủ nội dung để viết truy vấn phân biệt được không."""
    import cv2
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        return False
    m, s = float(img.mean()), float(img.std())
    return s >= MIN_STD and MIN_MEAN <= m <= MAX_MEAN


def _ranges(frame_idx: int, rows, pos: int):
    """Sinh các khoảng đáp án [s,e] cho keyframe ở vị trí `pos` trong `rows`."""
    last = rows[-1]["frame_idx"]
    out = {}
    for name, half in TOLERANCES.items():
        out[name] = [max(0, frame_idx - half), min(last, frame_idx + half)]
    # Khoảng "đoạn thuộc về keyframe này": nửa đường tới keyframe trước và sau.
    # Xấp xỉ cách hiểu "chỉ ra một khung hình BẤT KỲ thuộc đoạn video đó" (thể lệ, trang 1).
    prev_f = rows[pos - 1]["frame_idx"] if pos > 0 else 0
    next_f = rows[pos + 1]["frame_idx"] if pos + 1 < len(rows) else last
    out["kf_span"] = [(prev_f + frame_idx) // 2, (frame_idx + next_f) // 2]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="số mẫu cần lấy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    videos = sorted(d for d in os.listdir(KEYFRAME_DIR)
                    if os.path.isdir(os.path.join(KEYFRAME_DIR, d)))
    _log(f"[1/3] {len(videos)} video có keyframe trong {KEYFRAME_DIR}")

    rng.shuffle(videos)
    samples, skipped = [], {"no_map": 0, "few_kf": 0, "bad_img": 0, "no_file": 0}

    for vid in videos:
        if len(samples) >= args.n:
            break
        rows = _read_map(vid)
        if rows is None:
            skipped["no_map"] += 1
            continue
        if len(rows) < 20:
            skipped["few_kf"] += 1
            continue

        # bỏ 10% đầu/cuối: intro, outro, logo, hình đen
        lo, hi = int(len(rows) * 0.10), int(len(rows) * 0.90)
        cand = list(range(lo, hi))
        rng.shuffle(cand)

        picked = None
        for pos in cand[:MAX_TRY_PER_VIDEO]:
            row = rows[pos]
            jpg = os.path.join(KEYFRAME_DIR, vid, f"{row['n']:03d}.jpg")
            if not os.path.exists(jpg):
                skipped["no_file"] += 1
                continue
            if not _usable(jpg):
                skipped["bad_img"] += 1
                continue
            picked = (pos, row, jpg)
            break
        if picked is None:
            continue

        pos, row, jpg = picked
        vpath = _video_path(vid)
        samples.append({
            "query_id": f"kis_{len(samples) + 1:03d}",
            "type": "KIS",
            "query": "",                      # bước 2 điền
            "source": "",                     # "llm" | "manual" — bước 2 điền
            "gt_video": vid,
            "gt_frame": row["frame_idx"],
            "gt_ranges": _ranges(row["frame_idx"], rows, pos),
            "keyframe_n": row["n"],
            "keyframe_file": jpg.replace("\\", "/"),
            "pts_time": row["pts_time"],
            "fps": row["fps"],
            "n_keyframes": len(rows),
            "video_path": vpath.replace("\\", "/") if vpath else None,
        })

    _log(f"[2/3] lấy được {len(samples)}/{args.n} mẫu; bỏ qua: {skipped}")
    if len(samples) < args.n:
        _log(f"      ⚠ THIẾU {args.n - len(samples)} mẫu — hết video hợp lệ.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    meta = {
        "note": "Ground truth KIS tự tạo. `query` do bước 2 sinh từ ẢNH (không nhìn caption).",
        "n_samples": len(samples),
        "seed": args.seed,
        "tolerances": TOLERANCES,
        "gt_range_caveat": (
            "BTC không công bố bề rộng [s,e] cố định. Ví dụ trong thể lệ: KIS [500,510]=11 frame, "
            "Q&A [800,900]=101 frame. Luôn báo cáo điểm ở CẢ các mức, đừng chốt một mức."
        ),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "samples": samples}, f, ensure_ascii=False, indent=2)

    have_video = sum(1 for s in samples if s["video_path"])
    _log(f"[3/3] ghi {args.out} — {len(samples)} mẫu, {have_video} mẫu có sẵn file .mp4")
    _log("      Bước tiếp: python -u scripts/gen_kis_queries.py > logs/gen_kis.log 2>&1")


if __name__ == "__main__":
    main()
