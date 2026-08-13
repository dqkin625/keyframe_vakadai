"""CÁCH 2 (TRAKE): gán nhãn ACTION cho từng SỰ KIỆN trong events.jsonl — CHẠY RIÊNG.

    python scripts/label_events.py [data/output/L21_V001] [--multi-only] [--limit N] [--dry]

Vì sao chạy riêng (không qua full pipeline): events.jsonl + ảnh keyframe ĐÃ có sẵn trong
data/output/<vid>/. Script này chỉ nạp lại ảnh đại diện mỗi sự kiện rồi gọi VLM gán nhãn ->
CHỈ tốn quota cho NHÃN, không caption lại cả video. Ghi `action` ngược vào events.jsonl.

Phân công lao động (khác hẳn "caption kể chuyện"):
  - "KHI NÀO"  (start/end/anchor_frame từng pha)  <- đã do event_segmentation (optical flow) cắt.
  - "LÀM GÌ"   (nhãn ngữ nghĩa từng pha)          <- VLM gán ở đây, 1 ảnh/sự kiện, ngắn (<25 từ).

  --multi-only : chỉ gán cho shot bị cắt >1 sự kiện (chỗ TRAKE THẬT SỰ cần phân biệt pha);
                 shot 1-sự-kiện đã có caption shot bao trùm -> bỏ để tiết kiệm quota.
  --limit N    : chỉ gán N shot đầu (verify chất lượng trước khi chạy cả video dưới throttle).
  --dry        : không gọi API, chỉ in xem sẽ gán bao nhiêu shot/sự kiện.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import yaml

from src.captioning import build_captioner
from src.pipeline import label_event_actions


class _KF:
    """Keyframe tối giản cho label_event_actions (chỉ cần frame_index + image)."""
    __slots__ = ("frame_index", "image")

    def __init__(self, frame_index, image):
        self.frame_index = frame_index
        self.image = image


def main():
    args = [a for a in sys.argv[1:]]
    outdir = next((a for a in args if not a.startswith("--")), "data/output/L21_V001")
    multi_only = "--multi-only" in args
    dry = "--dry" in args
    limit = None
    for a in args:
        if a.startswith("--limit"):
            limit = int(a.split("=")[1]) if "=" in a else int(args[args.index(a) + 1])

    ev_p = os.path.join(outdir, "events.jsonl")
    kf_p = os.path.join(outdir, "keyframes.jsonl")
    for p in (ev_p, kf_p):
        if not os.path.exists(p):
            print(f"THIẾU {p}"); sys.exit(2)

    events = [json.loads(l) for l in open(ev_p, encoding="utf-8") if l.strip()]
    kfs_meta = [json.loads(l) for l in open(kf_p, encoding="utf-8") if l.strip()]

    # chọn SHOT cần gán
    from collections import Counter, defaultdict as _dd
    n_ev_of = Counter(e["shot_index"] for e in events)
    target_shots = set(n_ev_of)
    if multi_only:
        target_shots = {si for si, c in n_ev_of.items() if c > 1}
    # RESUME: chỉ giữ shot còn ÍT NHẤT 1 sự kiện CHƯA có nhãn -> chạy lại nhiều lượt lấp dần
    # (throttle Gemma multimodal chỉ cho ~57%/lượt; giống caption_waves).
    resume = "--no-resume" not in args
    if resume:
        unlabeled_shot = {e["shot_index"] for e in events if not (e.get("action") or "").strip()}
        target_shots &= unlabeled_shot
    if limit is not None:
        target_shots = set(sorted(target_shots)[:limit])

    sel_events = [e for e in events if e["shot_index"] in target_shots]
    n_sel_ev = len(sel_events)
    print(f"=== {outdir} ===")
    print(f"  {len(events)} sự kiện / {len(n_ev_of)} shot | sẽ gán: {len(target_shots)} shot, "
          f"{n_sel_ev} sự kiện"
          + ("  [--multi-only]" if multi_only else "")
          + (f"  [--limit {limit}]" if limit else ""))
    if dry:
        print("  (--dry: không gọi API)"); return

    # nạp ảnh keyframe cần dùng (chỉ những frame là rep của sự kiện được chọn -> đỡ RAM)
    need_frames = set()
    for e in sel_events:
        need_frames.update(e.get("keyframe_frames") or [])
    kfs = []
    miss = 0
    for k in kfs_meta:
        if k["frame_index"] in need_frames:
            img = cv2.imread(k["keyframe_path"])
            if img is None:
                miss += 1
                continue
            kfs.append(_KF(k["frame_index"], img))
    print(f"  nạp {len(kfs)} ảnh keyframe" + (f"  ⚠️ thiếu {miss} ảnh" if miss else ""))

    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    cfg["caption"]["backend"] = cfg["caption"].get("backend", "gemma")
    captioner = build_captioner(cfg["caption"])
    if not hasattr(captioner, "label_events_batch"):
        print(f"  backend '{cfg['caption']['backend']}' KHÔNG hỗ trợ label_events_batch"); sys.exit(2)

    n_ok = label_event_actions(sel_events, kfs, captioner, log=print)

    # ghi ngược action vào events.jsonl (theo id)
    action_of = {e["id"]: e.get("action", "") for e in sel_events}
    for e in events:
        if e["id"] in action_of and action_of[e["id"]]:
            e["action"] = action_of[e["id"]]
    with open(ev_p, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # thống kê + vài mẫu
    labeled = [e for e in events if (e.get("action") or "").strip()]
    print(f"\n  ĐÃ GÁN: {len(labeled)}/{len(events)} sự kiện có nhãn action")
    print(f"  --- vài chuỗi đã gán (shot nhiều pha) ---")
    from collections import defaultdict
    by_shot = defaultdict(list)
    for e in labeled:
        by_shot[e["shot_index"]].append(e)
    shown = 0
    for si in sorted(by_shot):
        evs = sorted(by_shot[si], key=lambda e: e["start_frame"])
        if len(evs) < 2:
            continue
        print(f"  shot {si} ({len(evs)} pha):")
        for e in evs:
            print(f"     f{e['anchor_frame']:>6} [{e['motion']}]  ->  {e['action']}")
        shown += 1
        if shown >= 6:
            break


if __name__ == "__main__":
    main()
