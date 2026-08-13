"""
main.py — Chạy pipeline chỉ với một lệnh:

    python main.py

Muốn đổi video test: sửa dòng VIDEO ở phần "CHỈNH Ở ĐÂY" bên dưới.
Hoặc truyền thẳng tên file:   python main.py tenvideo.mp4

Script tự tìm và dùng Python trong .venv, nên chạy bằng python nào cũng được.
"""

# ══════════════════════ CHỈNH Ở ĐÂY ══════════════════════
VIDEO = "POV/POV3.mp4"      # đường dẫn trong data/videos/ (hỗ trợ thư mục con)
                           # vd: "POV/POV.mp4" | "test/testVTV.mp4" | "CCTV/..."
                           # ""  = chạy TẤT CẢ video (quét cả thư mục con)

DETECTOR = "autoshot"      # cắt shot: autoshot (2 đội vô địch dùng, chính xác nhất)
                           #           | transnetv2 | pyscenedetect (nhẹ, không cần model)
CAPTION = "mock"           # ⚠ TẠM: chỉ chạy tới TRƯỚC caption (mock = không gọi API). Trả về "gemma" khi cần caption thật.
                           # caption: gemma (API free, nhanh, song song, ko cần GPU) | qwen3-vl (local) | blip | mock
DEVICE = "cuda"            # cuda (GPU) | cpu

CAPTION_LEVEL = "shot"     # shot: 1 caption/shot mô tả HÀNH ĐỘNG (gửi nhiều keyframe/1 request)
                           #        | keyframe: 1 caption/ảnh tĩnh | auto: tự chọn theo mật độ shot

STRATEGY = "action"        # ← caption theo shot mô tả diễn biến; đổi "dake"/"adaptive" nếu cần
                           #    (dake = 1 keyframe/đoạn cho video quay liên tục)
# ═════════════════════════════════════════════════════════

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---- Tự chuyển sang Python của .venv (để chạy `python main.py` bằng python nào cũng được) ----
_VENV_PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")      # Windows
if not os.path.exists(_VENV_PY):
    _VENV_PY = os.path.join(ROOT, ".venv", "bin", "python")          # Linux/macOS

if (os.path.exists(_VENV_PY)
        and os.path.normcase(sys.executable) != os.path.normcase(_VENV_PY)
        and not os.environ.get("_HCMAI_RELAUNCHED")):
    env = dict(os.environ, _HCMAI_RELAUNCHED="1", PYTHONIOENCODING="utf-8")
    sys.exit(subprocess.call([_VENV_PY, os.path.abspath(__file__)] + sys.argv[1:], env=env))

# ---- Chạy được từ bất kỳ thư mục nào ----
os.chdir(ROOT)
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import logging                                    # noqa: E402
import warnings                                   # noqa: E402
warnings.filterwarnings("ignore")                 # bớt cảnh báo rác của torch/transformers
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


import time                                       # noqa: E402
import yaml                                       # noqa: E402

from src.pipeline import process_video, run, finalize_output   # noqa: E402


VIDEO_EXT = (".mp4", ".mkv", ".avi", ".mov", ".webm")


def list_videos(root: str):
    """Liệt kê mọi video trong root, KỂ CẢ thư mục con. Trả về đường dẫn tương đối."""
    out = []
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            if f.lower().endswith(VIDEO_EXT):
                rel = os.path.relpath(os.path.join(dirpath, f), root)
                out.append(rel.replace("\\", "/"))
    return sorted(out)


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else VIDEO

    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["shot_detection"]["detector"] = DETECTOR
    cfg["keyframe"]["strategy"] = STRATEGY

    out_dir = cfg["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 58)
    print(f"  Video      : {video or '(tất cả trong ' + cfg['paths']['video_dir'] + ')'}")
    print(f"  Cắt shot   : {DETECTOR}")
    print(f"  Sample kf  : {STRATEGY}")
    print(f"  Caption    : {CAPTION}  ({DEVICE})")
    print("=" * 58)

    t0 = time.time()

    if video:
        path = os.path.join(cfg["paths"]["video_dir"], video)
        if not os.path.exists(path):
            print(f"\n[LỖI] Không tìm thấy: {path}\n")
            have = list_videos(cfg["paths"]["video_dir"])
            if have:
                print("Các video đang có (copy nguyên dòng vào VIDEO):")
                for v in have:
                    print(f"   {v}")
            else:
                print("(chưa có video nào trong " + cfg["paths"]["video_dir"] + ")")
            sys.exit(1)

        recs, embs, evs = process_video(path, cfg)
        video_id = os.path.splitext(os.path.basename(path))[0]   # lưu riêng theo tên video
        out = finalize_output(recs, {},
                              out_dir, subdir=video_id, all_events=evs)
        out_dir = os.path.join(out_dir, video_id)                # để in đường dẫn đúng bên dưới
        n_shot = sum(1 for _ in open(out["shots"], encoding="utf-8"))
        n_kf = len(recs)
        n_ev = len(evs)
    else:
        res = run(cfg)
        n_shot = sum(1 for _ in open(res["shots"], encoding="utf-8"))
        n_kf = sum(1 for _ in open(res["keyframes"], encoding="utf-8"))
        n_ev = (sum(1 for _ in open(res["events"], encoding="utf-8"))
                if res.get("events") else 0)

    el = time.time() - t0
    print("\n" + "=" * 58)
    print(f"  XONG trong {el:.1f}s  →  {n_shot} shot, {n_kf} keyframe, {n_ev} sự kiện")
    print(f"  {out_dir}\\shots.jsonl      (mốc shot)")
    print(f"  {out_dir}\\keyframes.jsonl  (từng keyframe)")
    print(f"  {out_dir}\\events.jsonl     (chuỗi sự kiện có thứ tự - TRAKE)")
    print(f"  {out_dir}\\keyframes\\       (ảnh)")
    print("=" * 58)


if __name__ == "__main__":
    main()
