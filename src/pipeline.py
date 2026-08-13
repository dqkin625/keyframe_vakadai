"""
Pipeline end-to-end: video -> detect_shots -> extract_keyframes -> caption_shot
-> ghi keyframe .jpg + metadata .jsonl.

Xuất 2 file bổ sung nhau: keyframes.jsonl (1 dòng = 1 keyframe, đơn vị EMBEDDING)
và shots.jsonl (1 dòng = 1 shot, đơn vị TRUY XUẤT). Caption gắn theo SHOT.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List

import cv2
from tqdm import tqdm

from .frame_sampling import detect_shots, extract_keyframes
from .captioning import build_captioner


import re as _re
# mốc "TEXT:" phải đứng ĐẦU DÒNG hoặc sau khoảng trắng (tránh cắt nhầm "context:", "subtext:")
_OCR_MARKER = _re.compile(r"(?:^|\s)(?:TEXT|Text|text)\s*:", _re.MULTILINE)


def split_caption_ocr(text: str):
    """Tách chuỗi model trả về thành (caption, ocr) theo mốc 'TEXT:' (đứng riêng)."""
    if not text:
        return "", ""
    m = _OCR_MARKER.search(text)
    if m is None:
        return text.strip(), ""
    cap, ocr = text[:m.start()], text[m.end():]
    ocr = ocr.strip()
    if ocr.lower() in ("(không có)", "không có", "(none)", "none", "(no text)"):
        ocr = ""
    return cap.strip(), ocr


def decide_caption_level(ccfg: dict, shots, keyframes) -> str:
    """Quyết định caption theo 'shot' hay 'keyframe' (cfg level = shot|keyframe|auto).
    "auto": mật độ shot thấp (CCTV/POV) -> keyframe, cao (đã dựng) -> shot (U-CESE tr.6)."""
    level = str(ccfg.get("level", "auto")).lower()
    if level in ("shot", "keyframe"):
        return level

    if not shots or not keyframes:
        return "shot"
    duration_min = max(1e-6, (shots[-1].end_time - shots[0].start_time) / 60.0)
    shots_per_min = len(shots) / duration_min
    thr = float(ccfg.get("auto_shots_per_min", 2.0))
    chosen = "keyframe" if shots_per_min < thr else "shot"
    print(f"      [auto] {len(shots)} shot / {duration_min:.1f} phút "
          f"= {shots_per_min:.1f} shot/phút (ngưỡng {thr}) -> caption theo {chosen.upper()}",
          flush=True)
    return chosen


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


def label_event_actions(event_rows: List[dict], keyframes: list, captioner, log=print) -> int:
    """Gán nhãn hành động cho từng SỰ KIỆN bằng VLM. Sửa `event_rows` tại chỗ, trả số nhãn.
    Gửi ĐÚNG 1 ảnh đại diện/sự kiện theo thứ tự thời gian (keyframe gần `anchor_frame` nhất);
    gom theo SHOT -> 1 request/shot có ngữ cảnh, số request = số shot (không nở theo sự kiện)."""
    img_of = {kf.frame_index: kf.image for kf in keyframes}

    by_shot_ev: Dict[int, List[dict]] = defaultdict(list)
    for r in event_rows:
        by_shot_ev[r["shot_index"]].append(r)

    order = sorted(by_shot_ev)
    groups, picked = [], []
    for si in order:
        evs = sorted(by_shot_ev[si], key=lambda r: r["start_frame"])
        imgs, rows = [], []
        for r in evs:
            cands = r.get("keyframe_frames") or []
            if not cands:
                continue
            best = min(cands, key=lambda f: abs(f - r["anchor_frame"]))
            im = img_of.get(best)
            if im is None:
                continue
            imgs.append(im); rows.append(r)
        groups.append(imgs); picked.append(rows)

    n_req = sum(1 for g in groups if g)
    if not n_req:
        log("      [nhãn sự kiện] không sự kiện nào có keyframe -> BỎ QUA")
        return 0
    log(f"      [nhãn sự kiện] {n_req} request (1 ảnh/sự kiện, gom theo shot)...")
    labels = captioner.label_events_batch(groups)

    n_ok = 0
    for rows, labs in zip(picked, labels):
        for r, lab in zip(rows, labs or []):
            if lab:
                r["action"] = lab
                n_ok += 1
    log(f"      [nhãn sự kiện] gán được {n_ok}/{len(event_rows)} sự kiện")
    return n_ok


def process_video(video_path: str, cfg: dict, captioner,
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

    # ---- 3/4: caption (theo shot hoặc theo keyframe) ----
    ccfg = cfg["caption"]
    level = decide_caption_level(ccfg, shots, keyframes)

    kf_caption: Dict[int, str] = {}      # frame_index -> caption
    shot_caption: Dict[int, str] = {}    # shot_index  -> caption
    clip_embs = None                     # CLIP keyframe emb (tái dùng từ dedup nếu có)

    if level == "keyframe":
        if ccfg.get("dedup", True):
            # TỐI ƯU: dedup nội dung (Vortex) -> chỉ caption đại diện mỗi cụm rồi gán lại.
            from .captioning.reduce import caption_keyframes_dedup
            _say(f"[3/4] Sinh caption ({ccfg.get('backend')}) cho {len(keyframes)} KEYFRAME "
                 f"(dedup + batch)...")
            caps, clip_embs, n_calls = caption_keyframes_dedup(keyframes, captioner, ccfg, log=_say)
            for kf, c in zip(keyframes, caps):
                kf_caption[kf.frame_index] = c
        else:
            # U-CESE tr.6: mô tả TỪNG keyframe kèm cửa sổ ngữ cảnh (chậm)
            k = int(ccfg.get("context_window", 2))
            _say(f"[3/4] Sinh caption ({ccfg.get('backend')}) cho {len(keyframes)} KEYFRAME "
                 f"(ngữ cảnh ±{k})...")
            for i, kf in enumerate(tqdm(keyframes, desc="      caption", unit="kf",
                                        disable=not verbose)):
                ctx = [keyframes[j].image
                       for j in range(max(0, i - k), min(len(keyframes), i + k + 1)) if j != i]
                kf_caption[kf.frame_index] = captioner.caption_keyframe(kf.image, ctx)
    else:
        recap = ccfg.get("recap_memory", True)
        mem_len = int(ccfg.get("recap_memory_chars", 400))
        order = sorted(by_shot)
        hint_of = {s.index: getattr(s, "motion_hint", "") for s in shots}   # CHUYỂN ĐỘNG đo được / shot

        shot_by_idx0 = {s.index: s for s in shots}

        if not recap and ccfg.get("shot_video") and hasattr(captioner, "caption_shots_video"):
            # THỬ NGHIỆM: caption bằng VIDEO CLIP mỗi shot (hiểu motion thật) + fallback ảnh.
            _say(f"[3/4] Sinh caption ({ccfg.get('backend')}) cho {len(by_shot)} SHOT (VIDEO clip, fallback ảnh)...")
            spans = [(shot_by_idx0[si].start_time, shot_by_idx0[si].end_time) for si in order]
            frames_fb = [[kf.image for kf in by_shot[si]] for si in order]
            caps, n_vid, n_fb = captioner.caption_shots_video(
                video_path, spans, frames_fb, hints=[hint_of.get(si, "") for si in order])
            _say(f"      [video] {n_vid} shot dùng VIDEO, {n_fb} shot fallback về ẢNH")
            for si, c in zip(order, caps):
                shot_caption[si] = c
        elif not recap and hasattr(captioner, "caption_shots"):
            # KHÔNG ReCap -> shot ĐỘC LẬP -> caption SONG SONG (nhiều key).
            _say(f"[3/4] Sinh caption ({ccfg.get('backend')}) cho {len(by_shot)} SHOT (song song, +motion hint)...")
            caps = captioner.caption_shots([[kf.image for kf in by_shot[si]] for si in order],
                                           hints=[hint_of.get(si, "") for si in order])
            for si, c in zip(order, caps):
                shot_caption[si] = c
        else:
            # ReCap (U-CESE tr.7) hoặc backend local -> TUẦN TỰ theo thời gian để bộ nhớ
            # hồi quy có nghĩa (mô tả shot trước làm bối cảnh shot sau).
            _say(f"[3/4] Sinh caption ({ccfg.get('backend')}) cho {len(by_shot)} SHOT"
                 f"{' (ReCap memory)' if recap else ''}...")
            memory = ""
            for shot_idx in tqdm(order, total=len(order),
                                 desc="      caption", unit="shot", disable=not verbose):
                kfs = by_shot[shot_idx]
                cap = captioner.caption_shot([kf.image for kf in kfs], memory=memory if recap else "",
                                             hint=hint_of.get(shot_idx, ""))
                shot_caption[shot_idx] = cap
                if recap:   # memory = phần MÔ TẢ (bỏ OCR), cắt gọn
                    desc = split_caption_ocr(cap)[0]
                    memory = (memory + " " + desc).strip()[-mem_len:]

    # ---- 4/4: ghi ảnh + metadata ----
    _say(f"[4/4] Ghi {len(keyframes)} ảnh keyframe...")
    # XOÁ ảnh keyframe CŨ của video này trước khi ghi -> tránh RÁC orphan của settings cũ
    # tích luỹ qua các lần chạy (folder không tự dọn).
    _kf_dir = os.path.join(cfg["paths"]["output_dir"], cfg["paths"]["keyframe_subdir"], video_id)
    if os.path.isdir(_kf_dir):
        import shutil
        shutil.rmtree(_kf_dir, ignore_errors=True)
    shot_by_idx = {s.index: s for s in shots}
    records: List[dict] = []
    for kf in tqdm(keyframes, desc="      ghi ảnh", unit="ảnh", disable=not verbose):
        path = _write_keyframe(kf, cfg["paths"]["output_dir"], cfg["paths"]["keyframe_subdir"])
        shot = shot_by_idx[kf.shot_index]
        raw = (kf_caption.get(kf.frame_index) if level == "keyframe"
               else shot_caption[kf.shot_index])
        caption, ocr = split_caption_ocr(raw) if ccfg.get("ocr", True) else (raw, "")
        records.append({
            "video_id": kf.video_id,
            "shot_index": kf.shot_index,
            "frame_index": kf.frame_index,
            "time_sec": round(kf.time_sec, 3),
            "shot_start_frame": shot.start_frame,
            "shot_end_frame": shot.end_frame,
            "shot_start_time": round(shot.start_time, 3),
            "shot_end_time": round(shot.end_time, 3),
            "keyframe_path": path,
            "caption": caption,
            "ocr": ocr,                       # chữ trên hình (song ngữ Việt-Anh), OCR qua Qwen
            "caption_level": level,
        })

    # CẢNH BÁO caption lỗi API bị ghi vào output (tránh nhiễm dữ liệu âm thầm)
    n_err = sum(1 for r in records if r["caption"].startswith(("[lỗi", "[hết quota")))
    if n_err:
        _say(f"      ⚠️  {n_err}/{len(records)} caption LỖI API (bị '[lỗi...]'/'[hết quota]') "
             f"- kiểm tra quota/key rồi chạy lại video này")

    # ---- (tuỳ chọn) EMBEDDING: CLIP keyframe (VisualDB) + caption embedding (TextualDB) ----
    embs = {"clip": None, "siglip": None, "caption": None}
    ecfg = cfg.get("embedding", {})
    if ecfg.get("enabled", False):
        from .embedding import embed_keyframes_clip, embed_keyframes_siglip, CaptionEmbedder
        # Thiết bị: caption API/mock -> cuda (GPU rảnh); Qwen local -> cpu (tránh OOM 6GB).
        edev = ecfg.get("device", "auto")
        if edev == "auto":
            frees_gpu = ccfg.get("backend", "").lower() in ("gemma", "gemini", "api", "mock")
            edev = "cuda" if frees_gpu else "cpu"
        e = {**ecfg, "device": edev}
        imgs = [kf.image for kf in keyframes]
        _say(f"      [embed] thiết bị: {edev}")
        # CLIP: tái dùng từ dedup nếu có, không thì tính
        if clip_embs is not None and len(clip_embs) == len(records):
            embs["clip"] = clip_embs
        else:
            _say("      [embed] CLIP keyframe...")
            embs["clip"] = embed_keyframes_clip(imgs, e)
        if ecfg.get("siglip", False):
            _say("      [embed] SigLIP2 keyframe...")
            embs["siglip"] = embed_keyframes_siglip(imgs, e)
        _say("      [embed] caption (e5 đa ngôn ngữ)...")
        embedder = CaptionEmbedder(e)
        embs["caption"] = embedder.encode_captions([r["caption"] for r in records])

    # ---- CHUỖI SỰ KIỆN (TRAKE) — dựng SAU dedup để keyframe gắn vào đúng sự kiện ----
    event_rows = build_event_records(video_id, shots, by_shot, level)
    n_multi = sum(1 for s in shots if len(getattr(s, "events", None) or []) > 1)
    n_scene = len({r["scene_id"] for r in event_rows})
    _say(f"      [sự kiện] {len(event_rows)} sự kiện / {len(shots)} shot "
         f"({n_multi} shot bị cắt >1 sự kiện) -> {n_scene} scene")

    # ---- (tuỳ chọn) GÁN NHÃN hành động / sự kiện bằng VLM (MẶC ĐỊNH TẮT: request phụ thêm) ----
    if ccfg.get("event_labels", False) and hasattr(captioner, "label_events_batch"):
        label_event_actions(event_rows, keyframes, captioner, log=_say)

    return records, embs, event_rows


def shots_from_keyframes(records: List[dict]) -> List[dict]:
    """Dựng record cấp SHOT từ record keyframe (gom theo video_id + shot_index)."""
    by_shot: "Dict[tuple, list]" = defaultdict(list)
    for r in records:
        by_shot[(r["video_id"], r["shot_index"])].append(r)

    shots: List[dict] = []
    for (video_id, shot_index), kfs in sorted(by_shot.items()):
        kfs = sorted(kfs, key=lambda r: r["frame_index"])
        head = kfs[0]
        # caption cấp shot: "shot" -> 1 câu chung; "keyframe" -> ghép các câu khác nhau lại.
        uniq, seen = [], set()
        ocr_uniq, ocr_seen = [], set()
        for r in kfs:
            c = (r.get("caption") or "").strip()
            if c and c not in seen:
                seen.add(c); uniq.append(c)
            o = (r.get("ocr") or "").strip()
            if o and o not in ocr_seen:
                ocr_seen.add(o); ocr_uniq.append(o)
        shots.append({
            "video_id": video_id,
            "shot_index": shot_index,
            "start_frame": head["shot_start_frame"],
            "end_frame": head["shot_end_frame"],
            "start_time": head["shot_start_time"],
            "end_time": head["shot_end_time"],
            "duration": round(head["shot_end_time"] - head["shot_start_time"], 3),
            "caption": " | ".join(uniq),
            "ocr": " | ".join(ocr_uniq),               # chữ trên hình gộp cả shot
            "caption_level": head.get("caption_level", "shot"),
            "n_keyframes": len(kfs),
            # "shot": dùng chung caption -> KHÔNG lặp caption/ocr vào keyframe; "keyframe": đính riêng.
            "keyframes": [
                {"frame_index": k["frame_index"], "time_sec": k["time_sec"],
                 "keyframe_path": k["keyframe_path"],
                 **({} if head.get("caption_level") == "shot"
                    else {"caption": k.get("caption", ""), "ocr": k.get("ocr", "")})}
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
    """Gán `id` toàn cục (khoá nối Milvus↔Elasticsearch), ghi 3 file jsonl + embedding .npy
    CÙNG THỨ TỰ keyframes.jsonl. subdir: ghi vào out_dir/subdir/ để KHÔNG đè video khác."""
    import numpy as np
    if subdir:
        out_dir = os.path.join(out_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)
    for i, r in enumerate(all_kf):
        r["id"] = i                                # khoá KEYFRAME (Milvus PK, khớp clip/siglip .npy)
    all_shots = shots_from_keyframes(all_kf)
    for j, s in enumerate(all_shots):
        s["id"] = j                                # khoá SHOT (khớp caption_emb.npy khi caption theo shot)

    shot_level = bool(all_kf) and all_kf[0].get("caption_level") == "shot"

    # keyframes.jsonl GỌN: chế độ "shot", caption/ocr đã ở shots.jsonl -> KHÔNG lặp vào keyframe.
    KEEP = ("id", "video_id", "shot_index", "frame_index", "time_sec", "keyframe_path")

    def _slim(r):
        row = {k: r[k] for k in KEEP if k in r}
        if not shot_level:            # chế độ "keyframe": mỗi ảnh 1 caption RIÊNG -> phải giữ
            row["caption"] = r.get("caption", "")
            row["ocr"] = r.get("ocr", "")
        return row

    kf_path = os.path.join(out_dir, "keyframes.jsonl")
    shot_path = os.path.join(out_dir, "shots.jsonl")
    _write_jsonl(kf_path, [_slim(r) for r in all_kf])
    _write_jsonl(shot_path, all_shots)

    out = {"shots": shot_path, "keyframes": kf_path}

    if all_events:
        # CON TRỎ event -> keyframe (TRAKE tra thẳng embedding sự kiện). rep = keyframe gần
        # `anchor_frame` nhất (ưu tiên `keyframe_frames`, không có thì gần nhất cùng video).
        # Xem docs/METHODS.md mục J.
        kf_by_video: Dict[str, list] = defaultdict(list)   # video_id -> [(frame, kf_id)]
        for r in all_kf:
            kf_by_video[r["video_id"]].append((r["frame_index"], r["id"]))
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

    def _save(name, arr, n_expect, fname):
        if len(arr) == n_expect:
            p = os.path.join(out_dir, fname)
            np.save(p, arr)
            out[name] = p
        else:   # lệch số dòng -> KHÔNG lưu (tránh sai alignment), nhưng BÁO rõ
            print(f"  ⚠️  embedding '{name}' lệch ({len(arr)} vs {n_expect}) - BỎ lưu {fname}")

    # CLIP + SigLIP: THEO KEYFRAME (mỗi ảnh 1 vector, đều khác nhau) -> khớp keyframes.jsonl `id`.
    for name, fn in (("clip", "clip_keyframe.npy"), ("siglip", "siglip2_keyframe.npy")):
        good = [p for p in emb_parts.get(name, []) if p is not None]
        if good:
            _save(name, np.concatenate(good), len(all_kf), fn)

    # CAPTION embedding: chế độ "shot" -> 1 vector / SHOT (bỏ trùng, khớp shots.jsonl `id`);
    # chế độ "keyframe" -> 1 vector / keyframe.
    good = [p for p in emb_parts.get("caption", []) if p is not None]
    if good:
        arr = np.concatenate(good)
        if shot_level and len(arr) == len(all_kf):
            first_id = {}                          # keyframe ĐẦU mỗi shot (các kf/shot có caption giống nhau)
            for r in all_kf:
                key = (r["video_id"], r["shot_index"])
                if key not in first_id or r["id"] < first_id[key]:
                    first_id[key] = r["id"]
            rows = [arr[first_id[(s["video_id"], s["shot_index"])]] for s in all_shots]
            _save("caption", np.stack(rows), len(all_shots), "caption_emb.npy")
        else:
            _save("caption", arr, len(all_kf), "caption_emb.npy")
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

    captioner = build_captioner(cfg["caption"])

    all_kf: List[dict] = []
    all_events: List[dict] = []
    emb_parts = {"clip": [], "siglip": [], "caption": []}
    failed = []
    for vp in tqdm(videos, desc="videos"):
        try:
            recs, embs, evs = process_video(vp, cfg, captioner)
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
