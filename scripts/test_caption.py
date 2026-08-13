"""
test_caption.py — TEST NHANH khâu caption, BỎ QUA shot detection + keyframe extraction.

Đọc spans (start/end mỗi shot) + ảnh keyframe fallback từ CACHE (data/cache/<video_id>/)
đã tạo sẵn từ 1 lần chạy full trước đó -> chỉ chạy khâu GỌI API caption. Dùng để thử nghiệm
config caption (số luồng, giãn cách, retry...) nhanh mà không phải chờ shot detection mỗi lần.

VÍ DỤ:
    python scripts/test_caption.py                       # full 331 shot, config trong config.yaml
    python scripts/test_caption.py --shots 40            # chỉ 40 shot đầu (test nhanh)
    python scripts/test_caption.py --per-account 2 --gap 15   # override config để sweep
    python scripts/test_caption.py L21_V002 --shots 30

Tạo/cập nhật cache: chạy 1 lần   python scripts/save_shot_cache.py L21_V001
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VENV_PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
if not os.path.exists(_VENV_PY):
    _VENV_PY = os.path.join(ROOT, ".venv", "bin", "python")
if (os.path.exists(_VENV_PY)
        and os.path.normcase(sys.executable) != os.path.normcase(_VENV_PY)
        and not os.environ.get("_HCMAI_RELAUNCHED")):
    env = dict(os.environ, _HCMAI_RELAUNCHED="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    sys.exit(subprocess.call([_VENV_PY, os.path.abspath(__file__)] + sys.argv[1:], env=env))

os.chdir(ROOT)
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
import warnings; warnings.filterwarnings("ignore")

import argparse
import glob
import json
import time

import cv2
import numpy as np
import yaml

from src.captioning.captioner import build_captioner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id", nargs="?", default="L21_V001")
    ap.add_argument("--shots", type=int, default=0, help="chỉ caption N shot đầu (0 = tất cả)")
    ap.add_argument("--backend", default="gemma")
    # override config caption cho tiện sweep (None = giữ nguyên config.yaml)
    ap.add_argument("--inflight", type=int, default=None)
    ap.add_argument("--per-account", type=int, default=None)
    ap.add_argument("--gap", type=float, default=None)
    ap.add_argument("--retries", type=int, default=None)
    ap.add_argument("--cooldown", type=float, default=None)
    ap.add_argument("--rpm", type=int, default=None)
    args = ap.parse_args()

    vid = args.video_id
    cache = os.path.join("data", "cache", vid)
    if not os.path.exists(os.path.join(cache, "shots.jsonl")):
        print(f"[LỖI] chưa có cache {cache}. Tạo bằng:  python scripts/save_shot_cache.py {vid}")
        sys.exit(1)

    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ccfg = cfg["caption"]
    ccfg["backend"] = args.backend
    # áp override
    ov = {"api_max_inflight": args.inflight, "api_per_account_inflight": args.per_account,
          "api_per_account_min_gap_sec": args.gap, "api_max_retries": args.retries,
          "api_cooldown_sec": args.cooldown, "api_rpm_per_model": args.rpm}
    for k, v in ov.items():
        if v is not None:
            ccfg[k] = v

    # video path (cho cắt clip)
    vids = glob.glob(os.path.join(cfg["paths"]["video_dir"], "**", vid + ".*"), recursive=True)
    if not vids:
        print(f"[LỖI] không tìm thấy file video {vid} trong {cfg['paths']['video_dir']}")
        sys.exit(1)
    video_path = vids[0]

    # spans từ shots.jsonl (sắp theo shot_index)
    shots = [json.loads(l) for l in open(os.path.join(cache, "shots.jsonl"), encoding="utf-8")]
    shots.sort(key=lambda s: s["shot_index"])
    if args.shots > 0:
        shots = shots[:args.shots]
    spans = [(s["start_time"], s["end_time"]) for s in shots]

    # ảnh keyframe fallback: gom theo shot_index từ keyframes.jsonl, đọc từ cache/keyframes/
    kfs = [json.loads(l) for l in open(os.path.join(cache, "keyframes.jsonl"), encoding="utf-8")]
    by_shot = {}
    for k in kfs:
        by_shot.setdefault(k["shot_index"], []).append(k)
    kf_dir = os.path.join(cache, "keyframes")
    frames_fb = []
    for s in shots:
        imgs = []
        for k in sorted(by_shot.get(s["shot_index"], []), key=lambda x: x.get("frame_index", 0)):
            p = os.path.join(kf_dir, os.path.basename(k["keyframe_path"].replace("\\", "/")))
            im = cv2.imread(p)
            if im is not None:
                imgs.append(im)
        frames_fb.append(imgs if imgs else [np.zeros((64, 64, 3), np.uint8)])

    cap = build_captioner(ccfg)
    n = len(spans)
    print(f"== TEST CAPTION {vid}: {n} shot (cache, bỏ qua shot detection) ==")
    print(f"   inflight={ccfg.get('api_max_inflight')} per_account={ccfg.get('api_per_account_inflight')} "
          f"gap={ccfg.get('api_per_account_min_gap_sec')} retries={ccfg.get('api_max_retries')} "
          f"rpm={ccfg.get('api_rpm_per_model')} cooldown={ccfg.get('api_cooldown_sec')}", flush=True)
    t0 = time.time()
    caps, n_vid, n_fb = cap.caption_shots_video(video_path, spans, frames_fb, hints=[""] * n)
    dt = time.time() - t0

    ok = sum(1 for c in caps if c and not c.startswith("["))
    print(f"\n===== KẾT QUẢ =====")
    print(f"  THÀNH CÔNG : {ok}/{n} = {100*ok/n:.0f}%")
    print(f"  video/fallback: {n_vid}/{n_fb}")
    print(f"  thời gian  : {dt:.0f}s = {n/dt*60:.1f} shot/phút")
    if hasattr(cap, "_log_dist"):
        cap._log_dist()
    # lưu caption để xem
    outp = os.path.join(cache, "test_captions.jsonl")
    with open(outp, "w", encoding="utf-8") as f:
        for s, c in zip(shots, caps):
            f.write(json.dumps({"shot_index": s["shot_index"], "caption": c}, ensure_ascii=False) + "\n")
    print(f"  -> caption lưu ở {outp}")


if __name__ == "__main__":
    main()
