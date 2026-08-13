"""
Xuất metadata keyframe ra schema thống nhất cho cả nhóm.

    python -u scripts/export_keyframe_meta.py                      # tất cả video trong data/output
    python -u scripts/export_keyframe_meta.py --only L21_V001
    python -u scripts/export_keyframe_meta.py --id-style numeric   # id là SỐ theo từng video
    python -u scripts/export_keyframe_meta.py --format csv

KHÔNG chạy lại pipeline — chỉ đọc `data/output/<video>/keyframes.jsonl` đã có rồi ghi lại
theo tên trường nhóm yêu cầu. Vài giây cho toàn kho.

Trường xuất ra:
    video_id      "L21_V001"
    keyframe_id   khoá keyframe
    shot_id       khoá shot
    frame_idx     CHỈ SỐ FRAME GỐC trong video (0-based, khớp cv2.CAP_PROP_POS_FRAMES)
    fps           fps của video, LẶP Ở MỌI DÒNG như map-keyframes của BTC
    pts_time      frame_idx / fps
    keyframe_path đường dẫn ảnh

⚠️ `keyframe_id` và `frame_idx` là HAI THỨ KHÁC NHAU:
    keyframe_id = số thứ tự keyframe (0,1,2,... liên tục)
    frame_idx   = vị trí THẬT trong video (0,10,25,... nhảy cóc) — đây mới là số nộp BTC

--id-style (mặc định `global`):
    global  -> "L21_V001_kf000001" / "L21_V001_shot00000"  — DUY NHẤT TOÀN KHO
    numeric -> 1 / 0                                        — số, chỉ duy nhất trong 1 video
Chọn `global` làm mặc định vì `shot_index` của pipeline chỉ chạy 0,1,2... theo TỪNG video:
mọi video đều có shot 0, nên dùng làm khoá chính trong DB sẽ đụng nhau.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

FIELDS = ["video_id", "keyframe_id", "shot_id", "frame_idx", "fps", "pts_time", "keyframe_path"]


def _fps_of(video_dir: str, video_id: str) -> float:
    """fps lấy theo thứ tự tin cậy giảm dần: events.jsonl -> map-keyframes BTC -> đọc video."""
    p = os.path.join(video_dir, "events.jsonl")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    v = json.loads(line).get("fps")
                    if v:
                        return float(v)
                    break
    p = f"data/map-keyframes-aic25-b1/map-keyframes/{video_id}.csv"
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("fps"):
                    return float(row["fps"])
                break
    import cv2
    for folder in os.listdir("data/video"):
        cand = os.path.join("data/video", folder, f"{video_id}.mp4")
        if os.path.exists(cand):
            cap = cv2.VideoCapture(cand)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            cap.release()
            if fps > 0:
                return fps
    raise RuntimeError(f"Không xác định được fps cho {video_id}")


def convert(video_dir: str, video_id: str, id_style: str) -> list:
    src = os.path.join(video_dir, "keyframes.jsonl")
    if not os.path.exists(src):
        return []
    fps = _fps_of(video_dir, video_id)
    out = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            kf_n = int(r.get("n", r["id"] + 1)) - 1
            shot_n = int(r["shot_index"])
            # schema mới dùng `frame_idx`; bản cũ dùng `frame_index` -> nhận cả hai
            frame_idx = int(r["frame_idx"] if "frame_idx" in r else r["frame_index"])
            if id_style == "global":
                kf_id = f"{video_id}_kf{kf_n:06d}"
                shot_id = f"{video_id}_shot{shot_n:05d}"
            else:
                kf_id, shot_id = kf_n, shot_n
            out.append({
                "video_id": video_id,
                "keyframe_id": kf_id,
                "shot_id": shot_id,
                "frame_idx": frame_idx,
                "fps": fps,
                # tính lại từ frame_idx thay vì dùng time_sec có sẵn -> đảm bảo LUÔN nhất quán
                # với frame_idx dù pipeline có làm tròn khác đi.
                "pts_time": round(frame_idx / fps, 6),
                "keyframe_path": r.get("keyframe_path", "").replace("\\", "/"),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="data/output")
    ap.add_argument("--only", default=None, help="chỉ 1 video, vd L21_V001")
    ap.add_argument("--prefix", default=None, help="lọc theo tiền tố, vd L21")
    ap.add_argument("--id-style", choices=("global", "numeric"), default="global")
    ap.add_argument("--format", choices=("jsonl", "csv", "both"), default="both")
    ap.add_argument("--merged", default=None,
                    help="gộp TẤT CẢ video vào 1 file, vd data/output/keyframe_meta_all.jsonl")
    args = ap.parse_args()

    vids = sorted(d for d in os.listdir(args.out_root)
                  if os.path.isdir(os.path.join(args.out_root, d))
                  and os.path.exists(os.path.join(args.out_root, d, "keyframes.jsonl")))
    if args.only:
        vids = [v for v in vids if v == args.only]
    if args.prefix:
        vids = [v for v in vids if v.startswith(args.prefix)]
    if not vids:
        print("Không tìm thấy video nào có keyframes.jsonl", flush=True)
        return

    print(f"{len(vids)} video | id-style={args.id_style} | format={args.format}", flush=True)
    total, merged = 0, []
    for v in vids:
        d = os.path.join(args.out_root, v)
        rows = convert(d, v, args.id_style)
        if not rows:
            continue
        total += len(rows)
        merged.extend(rows)
        if args.format in ("jsonl", "both"):
            with open(os.path.join(d, "keyframe_meta.jsonl"), "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if args.format in ("csv", "both"):
            with open(os.path.join(d, "keyframe_meta.csv"), "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rows)

    print(f"xuất {total} keyframe -> <video>/keyframe_meta.jsonl + .csv", flush=True)

    if args.merged:
        os.makedirs(os.path.dirname(args.merged) or ".", exist_ok=True)
        with open(args.merged, "w", encoding="utf-8") as f:
            for r in merged:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"gộp toàn bộ -> {args.merged} ({len(merged)} dòng, "
              f"{os.path.getsize(args.merged)/1e6:.0f} MB)", flush=True)

    if merged:
        print("\nví dụ 2 dòng đầu:", flush=True)
        for r in merged[:2]:
            print("   " + json.dumps(r, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
