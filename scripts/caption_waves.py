"""
caption_waves.py — Caption cả video trên FREE-TIER bằng cách gọi theo ĐỢT + RESUME.

Vì sao: đo được (2026-08-03) free-tier Gemma siết throttle theo ACCOUNT khi gọi multimodal
LIÊN TỤC BỀN (331 shot 1 lượt -> ~29% thành công), nhưng gọi ĐỢT NGẮN + NGHỈ thì throttle
nguội lại giữa đợt (40 shot -> ~62%). Không phải do IP (đã thử 4G + proxy, không đổi).

Cách chạy:
- Mỗi ĐỢT caption tối đa `--shots-per-wave` shot CÒN LỖI, rồi NGHỈ `--rest` giây.
- Chỉ làm lại shot còn `[hết quota]`/`[rỗng]`/thiếu -> KHÔNG phí quota làm lại shot đã xong.
- Lặp tới khi phủ hết hoặc đủ `--max-waves` -> cộng dồn ~95%+ hoàn toàn FREE.
- Kết quả lưu bền ở data/cache/<video_id>/captions.jsonl (chạy lại là RESUME tiếp).

Cần cache trước (spans + keyframe). Nếu chưa có: chạy full 1 lần rồi
    python scripts/save_shot_cache.py <video_id>

VÍ DỤ:
    python scripts/caption_waves.py L21_V001 --shots-per-wave 40 --rest 60 --max-waves 20
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


def is_bad(cap: str) -> bool:
    """Caption LỖI/thiếu -> cần làm lại."""
    return (not cap) or (not str(cap).strip()) or str(cap).startswith("[")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id", nargs="?", default="L21_V001")
    ap.add_argument("--shots-per-wave", type=int, default=40, help="số shot tối đa/đợt")
    ap.add_argument("--rest", type=float, default=60, help="nghỉ bao nhiêu giây giữa 2 đợt")
    ap.add_argument("--max-waves", type=int, default=20)
    ap.add_argument("--reset", action="store_true", help="xoá caption đã lưu, làm lại từ đầu")
    # override config caption
    ap.add_argument("--inflight", type=int, default=None)
    ap.add_argument("--per-account", type=int, default=None)
    ap.add_argument("--gap", type=float, default=None)
    ap.add_argument("--retries", type=int, default=None)
    args = ap.parse_args()

    vid = args.video_id
    cache = os.path.join("data", "cache", vid)
    if not os.path.exists(os.path.join(cache, "shots.jsonl")):
        print(f"[LỖI] chưa có cache {cache}. Tạo bằng:  python scripts/save_shot_cache.py {vid}")
        sys.exit(1)

    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    ccfg = cfg["caption"]; ccfg["backend"] = "gemma"
    for k, v in {"api_max_inflight": args.inflight, "api_per_account_inflight": args.per_account,
                 "api_per_account_min_gap_sec": args.gap, "api_max_retries": args.retries}.items():
        if v is not None:
            ccfg[k] = v

    vids = glob.glob(os.path.join(cfg["paths"]["video_dir"], "**", vid + ".*"), recursive=True)
    if not vids:
        print(f"[LỖI] không tìm thấy video {vid}"); sys.exit(1)
    video_path = vids[0]

    # spans + ảnh fallback từ cache
    shots = [json.loads(l) for l in open(os.path.join(cache, "shots.jsonl"), encoding="utf-8")]
    shots.sort(key=lambda s: s["shot_index"])
    spans = [(s["start_time"], s["end_time"]) for s in shots]
    idx_list = [s["shot_index"] for s in shots]

    kfs = [json.loads(l) for l in open(os.path.join(cache, "keyframes.jsonl"), encoding="utf-8")]
    by_shot = {}
    for k in kfs:
        by_shot.setdefault(k["shot_index"], []).append(k)
    kf_dir = os.path.join(cache, "keyframes")
    frames_fb = []
    for s in shots:
        imgs = []
        for k in sorted(by_shot.get(s["shot_index"], []), key=lambda x: x.get("frame_index", 0)):
            im = cv2.imread(os.path.join(kf_dir, os.path.basename(k["keyframe_path"].replace("\\", "/"))))
            if im is not None:
                imgs.append(im)
        frames_fb.append(imgs if imgs else [np.zeros((64, 64, 3), np.uint8)])

    # RESUME: nạp caption đã lưu
    resfile = os.path.join(cache, "captions.jsonl")
    results = {}
    if os.path.exists(resfile) and not args.reset:
        for l in open(resfile, encoding="utf-8"):
            d = json.loads(l); results[d["shot_index"]] = d.get("caption", "")

    def save():
        with open(resfile, "w", encoding="utf-8") as f:
            for si in idx_list:
                f.write(json.dumps({"shot_index": si, "caption": results.get(si, "")}, ensure_ascii=False) + "\n")

    n_total = len(shots)
    cap = build_captioner(ccfg)
    print(f"== CAPTION WAVES {vid}: {n_total} shot | {args.shots_per_wave}/đợt, nghỉ {args.rest:g}s, "
          f"tối đa {args.max_waves} đợt ==")
    done0 = sum(1 for si in idx_list if not is_bad(results.get(si, "")))
    if done0:
        print(f"   RESUME: đã có {done0}/{n_total} caption tốt từ trước.")

    spw = max(1, args.shots_per_wave)
    base = float(args.rest)
    warm_rest = base            # nghỉ sau đợt PRODUCTIVE (throttle vừa bị hâm nóng) — TỰ HỌC
    rest_cap = base * 6         # trần nghỉ
    prev_pct = 1.0
    t0 = time.time()
    for wave in range(1, args.max_waves + 1):
        failed = [j for j, si in enumerate(idx_list) if is_bad(results.get(si, ""))]
        if not failed:
            print(f"\n✅ ĐỦ HẾT {n_total} shot — dừng."); break
        # cửa sổ xoay để mọi shot lỗi đều được thử (không kẹt ở nhóm đầu)
        off = ((wave - 1) * spw) % len(failed)
        batch = (failed[off:] + failed[:off])[:spw]
        sub_spans = [spans[j] for j in batch]
        sub_fb = [frames_fb[j] for j in batch]

        tw = time.time()
        caps, n_vid, n_fb = cap.caption_shots_video(video_path, sub_spans, sub_fb, hints=[""] * len(batch))
        got = 0
        for j, c in zip(batch, caps):
            if not is_bad(c):
                results[idx_list[j]] = c; got += 1
            else:
                results.setdefault(idx_list[j], c)   # giữ marker lỗi để lần sau retry
        save()
        pct = got / max(1, len(batch))
        done = sum(1 for si in idx_list if not is_bad(results.get(si, "")))
        # HỌC nhịp nguội: nếu đợt TRƯỚC đậu nhiều (hâm nóng) mà đợt NÀY crash -> warm_rest CHƯA đủ -> tăng.
        if prev_pct >= 0.5 and pct < 0.3:
            warm_rest = min(rest_cap, warm_rest * 1.5)
        # Nghỉ cho đợt KẾ: đợt vừa rồi productive -> throttle nóng -> nghỉ warm_rest (dài);
        #                  đợt vừa rồi crash -> ít request đậu, đã nguội -> nghỉ NGẮN.
        next_rest = warm_rest if pct >= 0.5 else base * 0.35
        prev_pct = pct
        print(f"  [đợt {wave}] caption {len(batch)} shot lỗi -> +{got} tốt ({100*pct:.0f}%) "
              f"| video {n_vid}/fallback {n_fb} | TỔNG {done}/{n_total} ({100*done/n_total:.0f}%) "
              f"| {time.time()-tw:.0f}s | warm_rest={warm_rest:.0f}s", flush=True)
        if hasattr(cap, "_log_dist"):
            cap._log_dist()
        remaining = sum(1 for si in idx_list if is_bad(results.get(si, "")))
        if remaining and wave < args.max_waves:
            print(f"     nghỉ {next_rest:.0f}s cho throttle nguội...", flush=True)
            time.sleep(next_rest)

    done = sum(1 for si in idx_list if not is_bad(results.get(si, "")))
    print(f"\n===== XONG {time.time()-t0:.0f}s =====")
    print(f"  THÀNH CÔNG: {done}/{n_total} ({100*done/n_total:.0f}%)")
    print(f"  caption lưu ở: {resfile}  (chạy lại script này = RESUME làm nốt phần còn lỗi)")


if __name__ == "__main__":
    main()
