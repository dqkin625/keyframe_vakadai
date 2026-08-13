"""
CLI chạy pipeline sample-frame + caption.

Ví dụ:
    python -m scripts.run_pipeline --config config.yaml
    python -m scripts.run_pipeline --config config.yaml --caption.backend mock   # test không GPU
    python -m scripts.run_pipeline --video data/videos/L01_V001.mp4               # 1 video
"""
import argparse
import os
import sys

import yaml

# Console Windows mặc định cp1252 -> ép UTF-8 để in caption tiếng Việt không lỗi.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import run, process_video, finalize_output  # noqa: E402
from src.captioning import build_captioner            # noqa: E402


def _apply_overrides(cfg: dict, pairs):
    """--a.b value  ->  cfg['a']['b'] = value (ép kiểu theo giá trị cũ)."""
    for key, val in pairs:
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        old = node.get(parts[-1])
        if isinstance(old, bool):
            val = val.lower() in ("1", "true", "yes")
        elif isinstance(old, int):
            val = int(val)
        elif isinstance(old, float):
            val = float(val)
        node[parts[-1]] = val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video", help="chạy 1 video duy nhất (bỏ qua video_dir)")
    ap.add_argument("--set", nargs=2, action="append", default=[], metavar=("KEY", "VALUE"),
                    help="override config, vd: --set caption.backend mock")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _apply_overrides(cfg, args.set)

    if args.video:
        captioner = build_captioner(cfg["caption"])
        recs, embs, evs = process_video(args.video, cfg, captioner)   # (records, embeddings, events)
        out_dir = cfg["paths"]["output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        video_id = os.path.splitext(os.path.basename(args.video))[0]   # lưu riêng theo tên video
        out = finalize_output(recs, {k: [embs.get(k)] for k in ("clip", "siglip", "caption")},
                              out_dir, subdir=video_id, all_events=evs)
        print(f"Xong: {len(recs)} keyframe, {len(evs)} sự kiện -> " + ", ".join(out.values()))
    else:
        run(cfg)


if __name__ == "__main__":
    main()
