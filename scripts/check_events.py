"""Kiểm tra tính đúng đắn của `events.jsonl` (nguyên liệu TRAKE).

Chạy:  python scripts/check_events.py data/output/L21_V001/events.jsonl
       python scripts/check_events.py data/output            (quét đệ quy mọi events.jsonl)

Kiểm 9 điều — mỗi điều đều là một cách hỏng ĐÃ TỪNG hoặc CÓ THỂ làm mất trắng điểm TRAKE:

  1. frame_id là frame THẬT     — bẫy lớn nhất của kho này: `001.jpg` KHÔNG phải frame 1.
                                   Nộp số thứ tự thay vì frame thật -> R-Score = 0 dù tìm đúng cảnh.
  2. anchor nằm trong đoạn      — anchor là frame ta NỘP; nằm ngoài đoạn là vô nghĩa.
  3. thứ tự thời gian           — TRAKE khớp chuỗi CÓ THỨ TỰ; đảo thứ tự là hỏng bài toán.
  4. phủ kín, không hở/chồng    — hở = có khoảnh khắc không sự kiện nào chứa -> không bao giờ trúng.
  5. event_ord liên tục / scene — chuỗi phải đánh số liên tục 0..n-1 trong mỗi scene.
  6. fps hợp lệ & nhất quán     — kho có cả 25 và 30fps; quy đổi sai giây<->frame là lệch hết.
  7. keyframe_frames trong đoạn — ảnh đại diện phải thuộc đúng đoạn của nó.
  8. nối được sang shots.jsonl  — (video_id, shot_index) phải tồn tại bên shots.jsonl.
  9. thống kê phân bố           — để thấy có bị cắt vụn / không cắt gì không.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict


def _load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _min_sec_from_config() -> float:
    """Đọc `keyframe.event_min_sec` từ config.yaml (mặc định 0.4 nếu không đọc được)."""
    cfgp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "config.yaml")
    try:
        import yaml
        with open(cfgp, encoding="utf-8") as f:
            return float(yaml.safe_load(f)["keyframe"].get("event_min_sec", 0.4))
    except Exception:
        return 0.4


MIN_SEC = _min_sec_from_config()


def check_file(ev_path: str) -> int:
    """Trả về SỐ LỖI tìm được (0 = sạch)."""
    rows = _load(ev_path)
    print(f"\n=== {ev_path} ===")
    if not rows:
        print("  ⚠️  FILE RỖNG")
        return 1

    errs = []

    def bad(msg, sample=None):
        errs.append(msg)
        print(f"  ✗ {msg}" + (f"\n      vd: {sample}" if sample else ""))

    # --- 1. frame thật (heuristic: nếu MỌI frame đều < số dòng thì rất giống 'số thứ tự') ---
    max_f = max(r["end_frame"] for r in rows)
    if max_f < len(rows):
        bad(f"NGHI VẤN frame là SỐ THỨ TỰ, không phải frame thật "
            f"(end_frame lớn nhất={max_f} < số sự kiện={len(rows)})")

    # --- 2/3/4/6/7 theo từng video ---
    by_vid = defaultdict(list)
    for r in rows:
        by_vid[r["video_id"]].append(r)

    for vid, rs in sorted(by_vid.items()):
        rs = sorted(rs, key=lambda r: (r["start_frame"], r["shot_index"], r["ord_in_shot"]))

        for r in rs:
            if not (r["start_frame"] <= r["anchor_frame"] <= r["end_frame"]):
                bad(f"[{vid}] anchor NGOÀI đoạn", r); break
            if r["start_frame"] > r["end_frame"]:
                bad(f"[{vid}] start_frame > end_frame", r); break
            for kf in r.get("keyframe_frames", []):
                if not (r["start_frame"] <= kf <= r["end_frame"]):
                    bad(f"[{vid}] keyframe {kf} NGOÀI đoạn", r); break

        # fps
        fpss = {round(float(r["fps"]), 3) for r in rs}
        if len(fpss) > 1:
            bad(f"[{vid}] fps KHÔNG nhất quán trong cùng video: {fpss}")
        for f in fpss:
            if not (1.0 < f < 240.0):
                bad(f"[{vid}] fps vô lý: {f}")
        # time khớp frame/fps
        f0 = list(fpss)[0]
        for r in rs[:200]:
            if abs(r["start_time"] - r["start_frame"] / f0) > 0.01:
                bad(f"[{vid}] start_time KHÔNG khớp start_frame/fps", r); break

        # thứ tự + phủ kín TRONG TỪNG SHOT (giữa 2 shot có thể hở do shot detection)
        by_shot = defaultdict(list)
        for r in rs:
            by_shot[r["shot_index"]].append(r)
        n_gap = n_ovl = n_ord = 0
        for si, evs in by_shot.items():
            evs.sort(key=lambda r: r["start_frame"])
            if [e["ord_in_shot"] for e in evs] != list(range(len(evs))):
                n_ord += 1
            for a, b in zip(evs, evs[1:]):
                if b["start_frame"] > a["end_frame"] + 1:
                    n_gap += 1
                elif b["start_frame"] <= a["end_frame"]:
                    n_ovl += 1
        if n_ord:
            bad(f"[{vid}] {n_ord} shot có ord_in_shot KHÔNG liên tục 0..n-1")
        if n_gap:
            bad(f"[{vid}] {n_gap} chỗ HỞ giữa 2 sự kiện liền nhau trong cùng shot")
        if n_ovl:
            bad(f"[{vid}] {n_ovl} chỗ CHỒNG LẤN giữa 2 sự kiện trong cùng shot")

        # đoạn ngắn bất thường: CHỈ chấp nhận khi cả shot vốn đã ngắn (không cắt được dài hơn).
        # Nếu là đoạn do CẮT ra mà ngắn hơn min -> min_gap đang tính sai (vd round thay vì ceil).
        n_ev_of = {si: len(evs) for si, evs in by_shot.items()}
        cut_short = [r for r in rs
                     if r["duration"] < MIN_SEC - 1e-6 and n_ev_of[r["shot_index"]] > 1]
        if cut_short:
            bad(f"[{vid}] {len(cut_short)} đoạn do CẮT ra ngắn hơn event_min_sec={MIN_SEC}s "
                f"(shot 1-sự-kiện ngắn thì KHÔNG tính) -> nghi min_gap làm tròn xuống",
                cut_short[0])

        # event_ord liên tục trong từng SCENE
        by_scene = defaultdict(list)
        for r in rs:
            by_scene[r["scene_id"]].append(r)
        n_sc = sum(1 for evs in by_scene.values()
                   if sorted(e["event_ord"] for e in evs) != list(range(len(evs))))
        if n_sc:
            bad(f"[{vid}] {n_sc} scene có event_ord KHÔNG liên tục 0..n-1")

    # --- 8. nối sang shots.jsonl ---
    shot_path = os.path.join(os.path.dirname(ev_path), "shots.jsonl")
    if os.path.exists(shot_path):
        keys = {(s["video_id"], s["shot_index"]) for s in _load(shot_path)}
        miss = {(r["video_id"], r["shot_index"]) for r in rows} - keys
        if miss:
            bad(f"{len(miss)} (video_id, shot_index) KHÔNG có trong shots.jsonl",
                list(miss)[:3])
    else:
        print(f"  … bỏ qua kiểm nối: không thấy {shot_path}")

    # --- 9. thống kê ---
    per_shot = Counter()
    for r in rows:
        per_shot[(r["video_id"], r["shot_index"])] += 1
    dist = Counter(per_shot.values())
    n_shot = len(per_shot)
    n_split = sum(c for k, c in dist.items() if k > 1)
    durs = sorted(r["duration"] for r in rows)
    n_act = sum(1 for r in rows if (r.get("action") or "").strip())

    print(f"  video={len(by_vid)}  shot={n_shot}  sự kiện={len(rows)}  "
          f"scene={len({(r['video_id'], r['scene_id']) for r in rows})}")
    print(f"  shot bị cắt >1 sự kiện: {n_split}/{n_shot} ({100*n_split/max(n_shot,1):.1f}%)")
    print("  phân bố sự kiện/shot: " +
          ", ".join(f"{k}→{dist[k]}" for k in sorted(dist)))
    if durs:
        print(f"  thời lượng sự kiện (giây): min={durs[0]:.2f}  "
              f"trung vị={durs[len(durs)//2]:.2f}  max={durs[-1]:.2f}")
    print(f"  đã gán nhãn `action`: {n_act}/{len(rows)}"
          + ("   (caption.event_labels đang TẮT — bình thường)" if n_act == 0 else ""))

    if not errs:
        print("  ✓ TẤT CẢ 8 KIỂM TRA ĐẠT")
    return len(errs)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    target = sys.argv[1]
    paths = []
    if os.path.isdir(target):
        for dp, _, fs in os.walk(target):
            if "events.jsonl" in fs:
                paths.append(os.path.join(dp, "events.jsonl"))
    else:
        paths = [target]
    if not paths:
        print(f"Không tìm thấy events.jsonl trong {target}")
        sys.exit(2)

    total = sum(check_file(p) for p in sorted(paths))
    print("\n" + "=" * 60)
    print(f"{len(paths)} file — {'SẠCH' if total == 0 else str(total) + ' LỖI'}")
    print("=" * 60)
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
