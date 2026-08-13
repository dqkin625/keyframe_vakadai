"""A/B FIX điểm yếu TRÌNH TỰ: caption CÙNG shot bằng config CŨ (0.5fps/4f) vs MỚI (1.5fps/8f).

    python scripts/test_temporal_caption.py [data/output/L21_V001] [--n 6]

Đo xem nâng fps/frames + siết prompt có làm caption CÓ TRÌNH TỰ thời gian không (điểm yếu:
caption tĩnh, chỉ 6% có trình tự). Chọn shot GIÀU CHUYỂN ĐỘNG (nhiều pha + có người + có
động từ) — nơi trình tự MỚI có ý nghĩa. Cùng prompt (mới) cho cả 2, chỉ khác fps/frames ->
cô lập đúng tác động của việc "model THẤY được nhiều khung hơn".

KHÔNG commit config này — chạy test xong xem có lợi thật rồi mới quyết giữ hay revert.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import yaml

from src.captioning import build_captioner

VIDEO = "data/video/Videos_L21_a/L21_V001.mp4"
SEQ = re.compile(r"(ban đầu|lúc đầu|đầu tiên|rồi |sau đó|tiếp theo|tiếp đó|cuối cùng|"
                 r"trước khi|sau khi|khi đó|dần dần|liền |ngay sau|đang.{0,30}thì)")


def has_seq(c):
    return bool(SEQ.search((c or "").lower()))


def main():
    args = sys.argv[1:]
    n_want = 6
    # tách --n [val] ra khỏi args trước khi tìm outdir (tránh nuốt nhầm val làm outdir)
    rest = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--n":
            n_want = int(args[i + 1]); i += 2; continue
        if a.startswith("--n="):
            n_want = int(a.split("=")[1]); i += 1; continue
        rest.append(a); i += 1
    outdir = next((a for a in rest if not a.startswith("--")), "data/output/L21_V001")

    shots = [json.loads(l) for l in open(os.path.join(outdir, "shots.jsonl"), encoding="utf-8") if l.strip()]
    events = [json.loads(l) for l in open(os.path.join(outdir, "events.jsonl"), encoding="utf-8") if l.strip()]
    from collections import Counter
    n_ev = Counter(e["shot_index"] for e in events)

    # chọn shot GIÀU CHUYỂN ĐỘNG: nhiều pha + caption cũ có người + có động từ hành động
    person = re.compile(r"(người|đàn ông|phụ nữ|nam|nữ|nhóm|cô|anh|chị|em|nhân viên|công nhân)")
    verb = re.compile(r"(đi |chạy|cúi|cầm|mang|đẩy|kéo|bước|trao|đưa|đặt|nâng|mở|đóng|quay|"
                      r"dừng|lên|xuống|băng qua|bắt tay|nhìn|vẫy|bê|khiêng|di chuyển)")
    cand = []
    for s in shots:
        c = (s.get("caption") or "")
        if c.strip().startswith("[") or not c.strip():
            continue
        if n_ev[s["shot_index"]] >= 2 and person.search(c.lower()) and verb.search(c.lower()):
            cand.append(s)
    cand.sort(key=lambda s: -n_ev[s["shot_index"]])         # nhiều pha nhất trước
    picked = cand[:n_want]
    print(f"=== A/B TRÌNH TỰ — {len(picked)} shot giàu chuyển động (L21_V001) ===\n")

    # ảnh fallback / shot (các keyframe của shot)
    def fb_imgs(s):
        out = []
        for kf in s.get("keyframes", []):
            im = cv2.imread(kf["keyframe_path"])
            if im is not None:
                out.append(im)
        return out

    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    cfg["caption"]["backend"] = "gemma"
    cfg["caption"]["shot_video"] = True
    cfg["caption"]["recap_memory"] = False
    cap = build_captioner(cfg["caption"])
    fps = cfg["caption"].get("video_fps"); mf = cfg["caption"].get("video_max_frames")
    print(f"config hiện tại: video_fps={fps}, video_max_frames={mf} (prompt MỚI)\n")

    spans = [(s["start_time"], s["end_time"]) for s in picked]
    frames_fb = [fb_imgs(s) for s in picked]

    # GỐC = caption ĐÃ LƯU trong shots.jsonl (prompt CŨ, sinh trước khi sửa) — "before" thật.
    old = [(s.get("caption") or "") for s in picked]
    # MỚI = caption lại bằng config HIỆN TẠI (prompt mới, fps/frames như config.yaml).
    print(f"Đang caption lại {len(picked)} shot bằng config hiện tại...", flush=True)
    new, nv, nfb = cap.caption_shots_video(VIDEO, spans, frames_fb, hints=[""] * len(spans))
    print(f"  {nv} shot dùng VIDEO, {nfb} fallback ảnh\n")

    n_old = n_new = ok = 0
    for s, co, cn in zip(picked, old, new):
        si = s["shot_index"]
        dur = s["end_time"] - s["start_time"]
        so, sn = has_seq(co), has_seq(cn)
        n_old += so; n_new += sn; ok += 1
        print(f"--- shot {si} ({dur:.0f}s, {n_ev[si]} pha) ---")
        print(f"  GỐC (prompt cũ) {'[trình tự]' if so else '[tĩnh]  ':<10}: {(co or '')[:220]}")
        print(f"  MỚI (prompt mới){'[trình tự]' if sn else '[tĩnh]  ':<10}: {(cn or '')[:220]}")
        print()

    d = max(1, ok)
    print("=" * 60)
    print(f"CÓ NGÔN NGỮ TRÌNH TỰ:  GỐC {n_old}/{ok} ({100*n_old/d:.0f}%)  ->  "
          f"MỚI {n_new}/{ok} ({100*n_new/d:.0f}%)   [fps={fps}/{mf}f, 0 token thêm]")
    print(f"Độ dài TB (ký tự):     GỐC {sum(len(c or '') for c in old)//d}  ->  "
          f"MỚI {sum(len(c or '') for c in new)//d}")
    print("=" * 60)
    print("=> MỚI >> GỐC: giữ prompt+0.5/4. MỚI ~ GỐC (không hơn): cân nhắc lên fps 1.0/6.")


if __name__ == "__main__":
    main()
