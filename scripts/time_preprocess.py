"""Đo THỜI GIAN phần TIỀN XỬ LÝ (trước caption): decode + AutoShot + keyframe + cắt sự kiện.
Không caption, không embedding. So với độ dài video.

    python scripts/time_preprocess.py [video.mp4]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import yaml

from src.frame_sampling.keyframe_extractor import combined_autoshot_action

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "data/video/Videos_L21_a/L21_V001.mp4"
cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
cfg["shot_detection"]["detector"] = "autoshot"
cfg["keyframe"]["strategy"] = "action"

cap = cv2.VideoCapture(VIDEO)
fps = float(cap.get(cv2.CAP_PROP_FPS) or 30)
nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
cap.release()
dur = nframe / fps
print(f"video: {os.path.basename(VIDEO)}  —  {dur:.0f}s ({dur/60:.1f} phút), "
      f"{nframe} frame @ {fps:.0f}fps\n", flush=True)

t0 = time.time()
res = combined_autoshot_action(VIDEO, "L21_V001", cfg["shot_detection"], cfg["keyframe"])
dt = time.time() - t0

if res is None:
    print("  (decode gộp thiếu frame — sẽ fallback tách rời khi chạy thật)")
    sys.exit(1)
shots, keyframes = res
n_ev = sum(len(getattr(s, "events", None) or []) for s in shots)

print("\n" + "=" * 54)
print(f"  TIỀN XỬ LÝ (decode+shot+keyframe+sự kiện): {dt:.1f}s")
print(f"  → NHANH HƠN REALTIME {dur/dt:.1f}×   (video {dur/60:.1f} phút xử lý trong {dt:.0f}s)")
print(f"  → {dt/(dur/60):.1f}s xử lý cho mỗi PHÚT video")
print("-" * 54)
print(f"  ra: {len(shots)} shot, {len(keyframes)} keyframe, {n_ev} pha")
print(f"  (config: action_max_gap_sec = {cfg['keyframe'].get('action_max_gap_sec')})")
print("=" * 54)
