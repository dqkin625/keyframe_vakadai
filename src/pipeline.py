"""
Pipeline: video -> detect_shots -> extract_keyframes -> ghi .jpg + metadata .jsonl.

DỪNG Ở BƯỚC KEYFRAME. Không caption, không embedding.

Xuất 3 file: keyframes.jsonl (1 dòng = 1 keyframe), shots.jsonl (1 dòng = 1 shot),
events.jsonl (chuỗi sự kiện TRAKE).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List

import cv2
from tqdm import tqdm

from .frame_sampling import detect_shots, extract_keyframes

CAPTION_LEVEL = "shot"


def _write_keyframe(kf, out_dir: str, keyframe_subdir: str) -> str:
    d = os.path.join(out_dir, keyframe_subdir, kf.video_id)
    os.makedirs(d, exist_ok=True)
    fname = f"shot{kf.shot_index:04d}_f{kf.frame_index:07d}.jpg"
    path = os.path.join(d, fname)
    cv2.imwrite(path, kf.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


def build_event_records(video_id: str, shots, by_shot: Dict[int, list],
                        shot_caption_level: str) -> List[dict]:
    """Dựng record cấp SỰ KIỆN (`events.jsonl`) — nguyên liệu cho TRAKE.
    Shot chưa cắt được sự kiện thì HẠ VỀ 1 sự kiện phủ trọn shot (events phủ 100% thời
    lượng). KHÔNG chép caption shot vào sự kiện (tránh nhân N lần); nối bằng `shot_index`."""
    from .frame_sampling.event_segmenter import Event, events_to_records

    ev_by_shot: Dict[int, list] = {}
    scene_of: Dict[int, int] = {}
    for s in shots:
        evs = list(getattr(s, "events", None) or [])
        if not evs:
            evs = [Event(video_id=video_id, shot_index=s.index, ord_in_shot=0,
                         start_frame=s.start_frame, end_frame=s.end_frame,
                         anchor_frame=(s.start_frame + s.end_frame) // 2,
                         fps=s.fps, motion=getattr(s, "motion_hint", ""))]
        ev_by_shot[s.index] = evs
        sid = getattr(s, "scene_id", -1)
        scene_of[s.index] = sid if sid >= 0 else s.index

    rows = events_to_records(ev_by_shot, scene_of)

    # gắn các keyframe THẬT rơi vào từng sự kiện (sau dedup) -> tầng sau có ảnh để re-rank
    kf_of_shot = {si: sorted(k.frame_index for k in kfs) for si, kfs in by_shot.items()}
    for r in rows:
        r["keyframe_frames"] = [f for f in kf_of_shot.get(r["shot_index"], [])
                                if r["start_frame"] <= f <= r["end_frame"]]
        r["caption_level"] = shot_caption_level
    return rows


def process_video(video_path: str, cfg: dict,
                  verbose: bool = True) -> "tuple[List[dict], Dict[str, object], List[dict]]":
    """Trả về (records_keyframe, embeddings, records_sự_kiện).
    ⚠️ Trước 2026-08-05 hàm này trả 2 giá trị; đã thêm `events` -> mọi call-site phải mở 3."""
    import time
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    def _say(msg):
        if verbose:
            print(msg, flush=True)

    det = cfg["shot_detection"].get("detector", "pyscenedetect")
    strat = cfg["keyframe"].get("strategy", "adaptive")
    # TỐI ƯU: AutoShot + (DAKE|action) -> GỘP detect shot + trích keyframe vào 1 lượt decode.
    combined = (det == "autoshot" and strat in ("dake", "action")
                and cfg.get("fast_combined_decode", True))
    shots = keyframes = None
    if combined:
        from .frame_sampling.keyframe_extractor import (
            combined_autoshot_dake, combined_autoshot_action)
        fn = combined_autoshot_action if strat == "action" else combined_autoshot_dake
        _say(f"[1-2/4] Cắt shot + {strat} (gộp 1 lượt decode)...")
        t = time.time()
        res = fn(video_path, video_id, cfg["shot_detection"], cfg["keyframe"])
        if res is not None:
            shots, keyframes = res
            _say(f"      → {len(shots)} shot, {len(keyframes)} keyframe   ({time.time()-t:.1f}s)")
        else:
            _say("      (decode gộp thiếu frame -> quay về tách rời)")

    if shots is None:      # đường tách rời (fallback hoặc detector/strategy khác)
        _say(f"[1/4] Cắt shot ({det})... (video dài có thể mất vài phút)")
        t = time.time()
        shots = detect_shots(video_path, cfg["shot_detection"])
        _say(f"      → {len(shots)} shot   ({time.time()-t:.1f}s)")
        _say(f"[2/4] Sample keyframe ({strat})...")
        t = time.time()
        keyframes = extract_keyframes(video_path, video_id, shots, cfg["keyframe"])
        _say(f"      → {len(keyframes)} keyframe   ({time.time()-t:.1f}s)")

    # GUARD: video hỏng/rỗng -> BỎ QUA, không crash cả batch.
    if not shots or not keyframes:
        _say(f"      ⚠️  '{video_id}' không đọc được shot/keyframe nào (video hỏng?) -> BỎ QUA")
        return [], {"clip": None, "siglip": None, "caption": None}, []

    # LỌC keyframe THẬT SỰ KHÁC NHAU (threshold lớn) -> ảnh lưu không trùng.
    if cfg["keyframe"].get("keyframe_dedup", True):
        from .frame_sampling.keyframe_extractor import dedup_keyframes
        n0 = len(keyframes)
        # Truyền TRẦN LỖ HỔNG xuống dedup: nếu không, dedup xoá sạch keyframe mà
        # `action_max_gap_sec` vừa chèn vào shot TĨNH (giống nhau -> bị coi là trùng).
        _fps = shots[0].fps if shots else 25.0
        _max_gap = int(float(cfg["keyframe"].get("action_max_gap_sec", 7.0)) * _fps)
        keyframes = dedup_keyframes(keyframes, float(cfg["keyframe"].get("keyframe_dedup_threshold", 0.35)),
                                    max_gap_frames=_max_gap,
                                    min_fill_diff=float(cfg["keyframe"].get("action_max_gap_min_diff", 0.0)))
        _say(f"      [lọc keyframe khác nhau] {n0} -> {len(keyframes)} (threshold "
             f"{cfg['keyframe'].get('keyframe_dedup_threshold', 0.35)})")

    # gom keyframe theo shot để sinh 1 caption / shot
    by_shot: Dict[int, list] = defaultdict(list)
    for kf in keyframes:
        by_shot[kf.shot_index].append(kf)

    # ---- 3/3: ghi ảnh + metadata ----
    _say(f"[3/3] Ghi {len(keyframes)} ảnh keyframe...")
    # XOÁ ảnh keyframe CŨ của video này trước khi ghi -> tránh rác orphan tích luỹ.
    _kf_dir = os.path.join(cfg["paths"]["output_dir"], cfg["paths"]["keyframe_subdir"], video_id)
    if os.path.isdir(_kf_dir):
        import shutil
        shutil.rmtree(_kf_dir, ignore_errors=True)
    shot_by_idx = {s.index: s for s in shots}
    records: List[dict] = []
    for kf in tqdm(keyframes, desc="      ghi ảnh", unit="ảnh", disable=not verbose):
        path = _write_keyframe(kf, cfg["paths"]["output_dir"], cfg["paths"]["keyframe_subdir"])
        shot = shot_by_idx[kf.shot_index]
        records.append({
            "video_id": kf.video_id,
            "shot_index": kf.shot_index,
            "frame_idx": kf.frame_index,
            "pts_time": round(kf.frame_index / shot.fps, 6),
            "fps": float(shot.fps),
            "shot_start_frame": shot.start_frame,
            "shot_end_frame": shot.end_frame,
            "shot_start_time": round(shot.start_time, 3),
            "shot_end_time": round(shot.end_time, 3),
            "keyframe_path": path,
        })

    embs = {"clip": None, "siglip": None, "caption": None}

    # ---- CHUỖI SỰ KIỆN (TRAKE) — dựng SAU dedup để keyframe gắn vào đúng sự kiện ----
    event_rows = build_event_records(video_id, shots, by_shot, CAPTION_LEVEL)
    n_multi = sum(1 for s in shots if len(getattr(s, "events", None) or []) > 1)
    n_scene = len({r["scene_id"] for r in event_rows})
    _say(f"      [sự kiện] {len(event_rows)} sự kiện / {len(shots)} shot "
         f"({n_multi} shot bị cắt >1 sự kiện) -> {n_scene} scene")

    return records, embs, event_rows


def shots_from_keyframes(records: List[dict]) -> List[dict]:
    """Dựng record cấp SHOT từ record keyframe (gom theo video_id + shot_index)."""
    by_shot: "Dict[tuple, list]" = defaultdict(list)
    for r in records:
        by_shot[(r["video_id"], r["shot_index"])].append(r)

    shots: List[dict] = []
    for (video_id, shot_index), kfs in sorted(by_shot.items()):
        kfs = sorted(kfs, key=lambda r: r["frame_idx"])
        head = kfs[0]
        shots.append({
            "video_id": video_id,
            "shot_id": f"{video_id}_shot{shot_index:05d}",
            "shot_index": shot_index,
            "start_frame": head["shot_start_frame"],
            "end_frame": head["shot_end_frame"],
            "start_time": head["shot_start_time"],
            "end_time": head["shot_end_time"],
            "duration": round(head["shot_end_time"] - head["shot_start_time"], 3),
            "fps": head["fps"],
            "n_keyframes": len(kfs),
            "keyframes": [
                {"frame_idx": k["frame_idx"], "pts_time": k["pts_time"],
                 "keyframe_path": k["keyframe_path"]}
                for k in kfs
            ],
        })
    return shots


def _write_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def finalize_output(all_kf: List[dict], emb_parts: Dict[str, list],
                    out_dir: str, subdir: str = None,
                    all_events: List[dict] = None) -> Dict[str, str]:
    """Ghi keyframes.jsonl + shots.jsonl + events.jsonl + map-keyframes.csv (chuẩn BTC).

    `n` và `frame_idx` là HAI thứ khác nhau: `n` là số thứ tự keyframe (1,2,3...),
    `frame_idx` là vị trí THẬT trong video (khớp cv2.CAP_PROP_POS_FRAMES) — số nộp BTC.
    """
    if subdir:
        out_dir = os.path.join(out_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)

    all_kf.sort(key=lambda r: (r["video_id"], r["frame_idx"]))
    n_of_video: Dict[str, int] = defaultdict(int)
    for i, r in enumerate(all_kf):
        r["id"] = i
        n_of_video[r["video_id"]] += 1
        r["n"] = n_of_video[r["video_id"]]
        r["keyframe_id"] = f"{r['video_id']}_kf{r['n'] - 1:06d}"
        r["shot_id"] = f"{r['video_id']}_shot{r['shot_index']:05d}"

    all_shots = shots_from_keyframes(all_kf)
    for j, s in enumerate(all_shots):
        s["id"] = j

    KEEP = ("id", "keyframe_id", "video_id", "shot_id", "shot_index",
            "n", "frame_idx", "pts_time", "fps", "keyframe_path")

    kf_path = os.path.join(out_dir, "keyframes.jsonl")
    shot_path = os.path.join(out_dir, "shots.jsonl")
    _write_jsonl(kf_path, [{k: r[k] for k in KEEP if k in r} for r in all_kf])
    _write_jsonl(shot_path, all_shots)

    out = {"shots": shot_path, "keyframes": kf_path}

    # map-keyframes.csv — ĐÚNG 4 cột BTC phát, để dùng thay thế được cho nhau
    import csv as _csv
    map_path = os.path.join(out_dir, "map-keyframes.csv")
    with open(map_path, "w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["n", "pts_time", "fps", "frame_idx"])
        w.writeheader()
        for r in all_kf:
            w.writerow({"n": r["n"], "pts_time": r["pts_time"],
                        "fps": r["fps"], "frame_idx": r["frame_idx"]})
    out["map_keyframes"] = map_path

    if all_events:
        kf_by_video: Dict[str, list] = defaultdict(list)   # video_id -> [(frame, kf_id)]
        for r in all_kf:
            kf_by_video[r["video_id"]].append((r["frame_idx"], r["id"]))
        for v in kf_by_video:
            kf_by_video[v].sort()
        for k, e in enumerate(all_events):
            e["id"] = k
            pool = e.get("keyframe_frames") or [f for f, _ in kf_by_video.get(e["video_id"], [])]
            if pool:
                rep_frame = min(pool, key=lambda f: abs(f - e["anchor_frame"]))
                vid_kfs = kf_by_video.get(e["video_id"], [])
                rep_id = min(vid_kfs, key=lambda fk: abs(fk[0] - rep_frame))[1] if vid_kfs else -1
                e["rep_frame"] = int(rep_frame)
                e["rep_keyframe_id"] = int(rep_id)
            else:
                e["rep_frame"] = int(e["anchor_frame"])
                e["rep_keyframe_id"] = -1
        ev_path = os.path.join(out_dir, "events.jsonl")
        _write_jsonl(ev_path, all_events)
        out["events"] = ev_path

    return out


def run(cfg: dict) -> Dict[str, str]:
    video_dir = cfg["paths"]["video_dir"]
    out_dir = cfg["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # quét đệ quy để hỗ trợ thư mục con (vd data/videos/POV/, data/videos/CCTV/)
    videos = []
    for dirpath, _, files in os.walk(video_dir):
        for f in sorted(files):
            if f.lower().endswith((".mp4", ".mkv", ".avi", ".mov", ".webm")):
                videos.append(os.path.join(dirpath, f))
    videos.sort()
    if not videos:
        raise SystemExit(f"Không tìm thấy video trong {video_dir}")

    all_kf: List[dict] = []
    all_events: List[dict] = []
    emb_parts = {"clip": [], "siglip": [], "caption": []}
    failed = []
    for vp in tqdm(videos, desc="videos"):
        try:
            recs, embs, evs = process_video(vp, cfg)
        except Exception as e:   # 1 video hỏng KHÔNG được làm chết cả batch 100GB
            failed.append((vp, str(e)[:80]))
            print(f"\n  ⚠️  LỖI xử lý '{vp}': {str(e)[:120]} -> BỎ QUA, chạy tiếp", flush=True)
            continue
        if not recs:             # video rỗng/hỏng đã được process_video bỏ qua
            failed.append((vp, "0 keyframe"))
            continue
        all_kf.extend(recs)
        all_events.extend(evs)
        for k in emb_parts:
            emb_parts[k].append(embs.get(k))

    out = finalize_output(all_kf, emb_parts, out_dir, all_events=all_events)
    ok_n = len(videos) - len(failed)
    print(f"\nXong: {ok_n}/{len(videos)} video OK -> {len(all_kf)} keyframe, "
          f"{len(all_events)} sự kiện")
    if failed:
        print(f"  ⚠️  {len(failed)} video BỎ QUA (hỏng/rỗng):")
        for vp, why in failed[:10]:
            print(f"     - {os.path.basename(vp)}: {why}")
        if len(failed) > 10:
            print(f"     ... và {len(failed)-10} video khác")
    for k, v in out.items():
        print(f"  - {v}")
    return out
