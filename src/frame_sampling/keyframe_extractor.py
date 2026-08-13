"""
Trích keyframe trong mỗi shot. Chiến lược (config.yaml -> keyframe.strategy):
middle | uniform | adaptive | dake (+ action, dake_per_shot, clip_reldiff).
Trả về danh sách Keyframe kèm ảnh (numpy BGR) để tầng caption/CLIP dùng lại.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np

from .shot_detector import Shot


@dataclass
class Keyframe:
    video_id: str
    shot_index: int
    frame_index: int          # chỉ số frame trong toàn video
    time_sec: float
    image: np.ndarray = field(repr=False)   # BGR uint8
    path: str = ""            # điền sau khi ghi ra đĩa


# ---- Helpers ----
def _resize_long_side(img: np.ndarray, long_side: int) -> np.ndarray:
    if long_side <= 0:
        return img
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _hist(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(h, h, 0, 1, cv2.NORM_MINMAX)
    return h.flatten()


def _jpeg_size(img: np.ndarray, quality: int) -> int:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return int(buf.nbytes) if ok else 0


def _read_shot_frames(cap: cv2.VideoCapture, shot: Shot, step: int = 1):
    """Yield (frame_index, image) cho các frame trong shot, giãn cách `step`."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, shot.start_frame)
    idx = shot.start_frame
    while idx <= shot.end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - shot.start_frame) % step == 0:
            yield idx, frame
        idx += 1


# ---- Strategies ----
def _pick_middle(cap, shot) -> List[int]:
    return [(shot.start_frame + shot.end_frame) // 2]


def _pick_uniform(cap, shot, step) -> List[int]:
    frames = list(range(shot.start_frame, shot.end_frame + 1, max(1, step)))
    return frames or [shot.start_frame]


def _hist_diff(a: np.ndarray, b: np.ndarray) -> float:
    """0 = giống hệt, 1 = khác hoàn toàn (dựa trên tương quan histogram)."""
    return 1.0 - cv2.compareHist(a.reshape(-1, 1), b.reshape(-1, 1), cv2.HISTCMP_CORREL)


def _pick_adaptive(cap, shot, max_per_shot, hist_threshold, min_gap_frames) -> List[int]:
    """Chọn keyframe ĐA DẠNG trong shot: hạt giống = frame giữa; thêm frame chỉ khi
    VỪA cách mọi keyframe đã giữ >= min_gap_frames VỪA khác > hist_threshold."""
    cands = [(idx, _hist(img)) for idx, img in _read_shot_frames(cap, shot, step=6)]
    if not cands:
        return [(shot.start_frame + shot.end_frame) // 2]

    # hạt giống: ứng viên gần giữa shot nhất
    mid_frame = (shot.start_frame + shot.end_frame) // 2
    seed = min(range(len(cands)), key=lambda i: abs(cands[i][0] - mid_frame))
    kept = [cands[seed]]

    # tham lam: mỗi vòng thêm ứng viên "mới lạ" nhất so với những cái đã giữ
    while len(kept) < max_per_shot:
        best, best_novelty = None, 0.0
        for idx, h in cands:
            if any(abs(idx - k[0]) < min_gap_frames for k in kept):
                continue                              # quá sát về thời gian -> bỏ
            novelty = min(_hist_diff(h, k[1]) for k in kept)   # khác cái GẦN nhất bao nhiêu
            if novelty > best_novelty:
                best, best_novelty = (idx, h), novelty
        if best is None or best_novelty < hist_threshold:
            break                                     # không còn gì đủ khác -> dừng
        kept.append(best)

    return sorted(k[0] for k in kept)


def _pick_action(cap, shot, n_per_shot, change_threshold, min_gap_frames) -> List[int]:
    """Chọn <= n_per_shot keyframe cho caption theo HÀNH ĐỘNG: rải đều theo thời gian,
    chỉ giữ frame khác frame giữ trước > change_threshold. Luôn giữ frame đầu, cố lấy
    thêm frame cuối. So với frame GIỮ TRƯỚC (không phải trung bình) để bám chuyển động."""
    length = shot.end_frame - shot.start_frame + 1
    if length <= 1:
        return [shot.start_frame]
    # lưới ứng viên dày ~3x số cần, rải đều -> có cái để lọc mà vẫn phủ thời gian
    n_cand = max(3 * n_per_shot, n_per_shot)
    step = max(1, length // n_cand)
    cands = [(idx, _hist(img)) for idx, img in _read_shot_frames(cap, shot, step=step)]
    if not cands:
        return [(shot.start_frame + shot.end_frame) // 2]

    kept = [cands[0]]                                   # luôn giữ frame đầu shot
    for idx, h in cands[1:]:
        if len(kept) >= n_per_shot:
            break
        if abs(idx - kept[-1][0]) < min_gap_frames:     # quá sát về thời gian -> bỏ
            continue
        if _hist_diff(h, kept[-1][1]) > change_threshold:   # THỰC SỰ khác mới lưu
            kept.append((idx, h))
    if len(kept) < n_per_shot and cands[-1][0] != kept[-1][0] \
            and _hist_diff(cands[-1][1], kept[-1][1]) > change_threshold * 0.5:
        kept.append(cands[-1])
    return sorted(k[0] for k in kept)


def _pick_dake(cap, shot, jpeg_quality, max_per_shot=1) -> List[int]:
    """[CŨ - giữ để tương thích] Chọn frame có JPEG size lớn nhất trong shot.
    Đây KHÔNG phải DAKE gốc; DAKE thật là `dake_global()` bên dưới."""
    best = []  # (size, idx)
    for idx, img in _read_shot_frames(cap, shot, step=4):
        best.append((_jpeg_size(img, jpeg_quality), idx))
    if not best:
        return [(shot.start_frame + shot.end_frame) // 2]
    best.sort(reverse=True)
    return sorted(idx for _, idx in best[:max_per_shot])


# ---- DAKE — Dynamic-Aware Keyframe Extraction (U-CESE, arxiv 2605.23274 tr.5-6) ----
# Algorithm 1 + steepness của paper, chạy TOÀN VIDEO (không cần shot detection).
def _jpeg_sizes_chunk(args):
    """[worker đa tiến trình] Đo JPEG size cho một ĐOẠN frame liên tiếp.
    Phải là hàm cấp module để pickle được trên Windows (spawn)."""
    video_path, start, count, long_side, quality = args
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    for _ in range(count):
        ok, frame = cap.read()
        if not ok:
            break
        out.append(_jpeg_size(_resize_long_side(frame, long_side), quality))
    cap.release()
    return start, out


def _video_dims(video_path):
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return W, H, n


def _scaled_wh(W, H, long_side):
    s = long_side / max(W, H) if max(W, H) > 0 else 1.0
    return max(2, int(round(W * s))), max(2, int(round(H * s)))


def _measure_jpeg_sizes(video_path: str, long_side: int, quality: int,
                        workers: int) -> List[int]:
    """Đo JPEG size của MỌI frame qua GPU decode STREAMING (tuần tự, KHÔNG seek — đọc
    đủ cả video long-GOP mà cv2 seek hay trượt). Nén JPEG song song bằng thread."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    from .video_io import stream_frames

    W, H, n_total = _video_dims(video_path)
    w, h = _scaled_wh(W, H, long_side)

    def enc(fr):
        ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return int(buf.nbytes) if ok else 0

    def _measure(use_gpu):
        out: List[int] = []
        bar = tqdm(total=n_total or None,
                   desc=f"      đo JPEG ({'GPU' if use_gpu else 'CPU'} stream)", unit="frame")
        with ThreadPoolExecutor(max_workers=nthreads) as ex:
            inflight = deque()
            for _, fr in stream_frames(video_path, w, h, use_gpu=use_gpu):
                inflight.append(ex.submit(enc, fr))
                if len(inflight) >= nthreads * 2:      # giới hạn frame giữ trong RAM
                    out.append(inflight.popleft().result()); bar.update(1)
            while inflight:
                out.append(inflight.popleft().result()); bar.update(1)
        bar.close()
        return out

    nthreads = max(2, workers)
    # Ưu tiên GPU; nếu GPU trả thiếu frame thì đọc lại CPU streaming (chắc chắn đủ).
    sizes = _measure(use_gpu=True)
    if n_total and len(sizes) < 0.9 * n_total:
        sizes = _measure(use_gpu=False)
    return sizes


def _read_frames_chunk(args):
    """[worker] Đọc các frame cần lấy trong MỘT đoạn, quét TUẦN TỰ (không seek lẻ).
    Trả về JPEG bytes thay vì numpy để giảm chi phí truyền giữa tiến trình."""
    video_path, start, count, wanted, long_side, quality = args
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    wanted = set(wanted)
    out, idx, remaining = [], start, len(wanted)
    for _ in range(count):
        if remaining <= 0:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            ok2, buf = cv2.imencode(".jpg", _resize_long_side(frame, long_side),
                                    [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok2:
                out.append((idx, buf.tobytes()))
            remaining -= 1
        idx += 1
    cap.release()
    return out


def _read_frames_parallel(video_path: str, indices: List[int], long_side: int,
                          workers: int, quality: int = 95) -> dict:
    """Đọc nhiều frame rải rác -> {frame_index: ảnh BGR}. Chia video thành đoạn,
    mỗi nhân quét tuần tự đoạn của mình (nhanh hơn nhiều so với seek từng frame)."""
    from tqdm import tqdm

    if not indices:
        return {}
    indices = sorted(set(indices))

    cap = cv2.VideoCapture(video_path)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    # Ít frame hoặc 1 nhân -> seek trực tiếp cho đơn giản
    if workers <= 1 or len(indices) < 20 or n_total < 500:
        cap = cv2.VideoCapture(video_path)
        out = {}
        for fidx in tqdm(indices, desc="      đọc keyframe", unit="frame"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            if ok:
                out[fidx] = _resize_long_side(frame, long_side)
        cap.release()
        return out

    chunk = math.ceil(n_total / workers)
    tasks = []
    for s in range(0, n_total, chunk):
        e = min(s + chunk, n_total)
        want = [i for i in indices if s <= i < e]
        if want:
            tasks.append((video_path, s, e - s, want, long_side, quality))

    from concurrent.futures import ProcessPoolExecutor, as_completed
    out = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_read_frames_chunk, t) for t in tasks]
        for f in tqdm(as_completed(futs), total=len(futs),
                      desc=f"      đọc keyframe ({workers} nhân)", unit="đoạn"):
            for idx, jpg in f.result():
                out[idx] = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    return out


def _steepness(s_i: float, s_j: float, i: int, j: int, s_max: float) -> float:
    """S(i,j) = A / sqrt((j-i)^2 + A^2),  A = 100*(s_j - s_i)/s_max   [paper tr.5]"""
    if s_max <= 0:
        return 0.0
    a = 100.0 * (s_j - s_i) / s_max
    return a / math.sqrt((j - i) ** 2 + a ** 2 + 1e-12)


def dake_global(video_path: str, fps: float, cfg: dict) -> List[int]:
    """DAKE gốc: trả về danh sách frame_index được chọn trên toàn video.

    cfg (block `keyframe`):
      dake_ratio           - tỉ lệ frame giữ làm keyframe (paper tr.10: 0.02)
      dake_min_window_sec  - ép mỗi cửa sổ phải có >=1 keyframe (paper tr.10: 2s)
      dake_size_long_side  - thu nhỏ trước khi đo JPEG cho nhanh
      dake_use_abs         - False = bám paper (steepness có dấu); True = dùng |steepness|
    """
    rho = float(cfg.get("dake_ratio", 0.02))
    win_sec = float(cfg.get("dake_min_window_sec", 2.0))
    small = int(cfg.get("dake_size_long_side", 320))
    use_abs = bool(cfg.get("dake_use_abs", False))
    quality = int(cfg.get("jpeg_quality", 90))

    # B1: đo kích thước JPEG của MỌI frame (đa nhân)
    workers = int(cfg.get("dake_workers", 0))
    if workers <= 0:                                   # 0 = tự động
        workers = max(1, (os.cpu_count() or 4) - 2)    # chừa 2 nhân cho hệ thống
    sizes = _measure_jpeg_sizes(video_path, small, quality, workers)
    return dake_select_from_sizes(sizes, fps, cfg)


def dake_select_from_sizes(sizes, fps: float, cfg: dict) -> List[int]:
    """Chọn keyframe từ danh sách JPEG size ĐÃ có (tách khỏi decode để GỘP với shot detect).
    Đúng Algorithm 1 (U-CESE tr.6): steepness + top-rho + ép phủ cửa sổ."""
    rho = float(cfg.get("dake_ratio", 0.02))
    win_sec = float(cfg.get("dake_min_window_sec", 2.0))
    use_abs = bool(cfg.get("dake_use_abs", False))

    n = len(sizes)
    if n == 0:
        return []
    if n <= 2:
        return [0]
    s_max = float(max(sizes)) or 1.0

    # B2: Algorithm 1 — steepness trung bình trong cửa sổ j = i+1..i+3
    s_list: List[tuple] = []
    for i in range(n - 1):
        total, count = 0.0, 0
        for j in range(i + 1, min(n, i + 4)):        # paper: min(n, i+3) inclusive
            total += _steepness(sizes[i], sizes[j], i, j, s_max)
            count += 1
        val = total / max(1, count)
        s_list.append((i, abs(val) if use_abs else val))

    # B3: lấy top-k theo steepness giảm dần, k = floor(rho * |S_list|)
    k = max(1, int(rho * len(s_list)))
    ranked = sorted(s_list, key=lambda x: x[1], reverse=True)
    selected = {idx for idx, _ in ranked[:k]}

    # B4: ép phủ — mỗi cửa sổ win_sec phải có >= 1 keyframe (paper tr.10)
    win = max(1, int(win_sec * fps))
    best_in_win = {}
    for idx, val in s_list:
        w = idx // win
        if w not in best_in_win or val > best_in_win[w][1]:
            best_in_win[w] = (idx, val)
    for w in range(0, (n - 1) // win + 1):
        if not any(w * win <= s < (w + 1) * win for s in selected):
            if w in best_in_win:
                selected.add(best_in_win[w][0])

    return sorted(selected)


# ---- VORTEX two-stage keyframe selection (arxiv 2606.19682 tr.5) ----
# Trong shot lấy embedding CLIP mỗi 8 frame, giữ khi rel_diff > 0.4.
_CLIP_CACHE = {}


def _get_clip(model_name: str, pretrained: str, device: str):
    """Nạp CLIP một lần (lazy) — chỉ cần khi dùng strategy 'clip_reldiff'."""
    key = (model_name, pretrained, device)
    if key not in _CLIP_CACHE:
        import open_clip          # pip install open_clip_torch
        import torch
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device)
        model.eval()
        _CLIP_CACHE[key] = (model, preprocess, torch)
    return _CLIP_CACHE[key]


def _pick_clip_reldiff(cap, shot, cfg) -> List[int]:
    """Vortex tr.5: sample mỗi `step` frame trong shot, giữ khi rel_diff > ngưỡng."""
    from PIL import Image

    step = int(cfg.get("clip_step", 8))                  # paper: every eighth frame
    thr = float(cfg.get("clip_rel_diff_threshold", 0.4))  # paper: 0.4
    device = cfg.get("clip_device", "cuda")
    model, preprocess, torch = _get_clip(
        cfg.get("clip_model", "ViT-L-14-quickgelu"),
        cfg.get("clip_pretrained", "dfn2b"), device)

    picks: List[int] = []
    e_prev = None
    for idx, img in _read_shot_frames(cap, shot, step=step):
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            e = model.encode_image(preprocess(pil).unsqueeze(0).to(device))[0].float()
        if e_prev is None:
            picks.append(idx)                 # frame đầu shot luôn là keyframe
            e_prev = e
            continue
        rel = (torch.norm(e - e_prev) / (torch.norm(e_prev) + 1e-8)).item()
        if rel > thr:
            picks.append(idx)
            e_prev = e
    return picks or [(shot.start_frame + shot.end_frame) // 2]


# ---- Public API ----
def _dake_read_and_map(video_path, video_id, idxs, shots, long_side):
    """ĐỌC các keyframe đã chọn (GPU streaming, fallback CPU) rồi GÁN vào shot gần nhất."""
    from .video_io import stream_frames
    idxs = set(idxs)
    W, H, _ = _video_dims(video_path)
    w, h = _scaled_wh(W, H, long_side) if long_side > 0 else (W, H)

    def _read(use_gpu):
        got = {}
        rem = len(idxs)
        for i, fr in stream_frames(video_path, w, h, use_gpu=use_gpu):
            if i in idxs:
                got[i] = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)   # về BGR để đồng nhất
                rem -= 1
                if rem <= 0:
                    break
        return got

    frames = _read(use_gpu=True)
    if len(frames) < 0.9 * len(idxs):      # GPU thiếu -> đọc lại CPU
        frames = _read(use_gpu=False)

    keyframes: List[Keyframe] = []
    si = 0
    for fidx in sorted(frames):
        while si < len(shots) - 1 and fidx > shots[si].end_frame:
            si += 1
        shot = shots[si]
        keyframes.append(Keyframe(
            video_id=video_id, shot_index=shot.index, frame_index=fidx,
            time_sec=fidx / shot.fps, image=frames[fidx],
        ))
    return keyframes


def action_global(video_path, shots, cfg, video_id: str = "") -> List[int]:
    """FAST cho strategy 'action': giải mã video 1 LƯỢT bằng GPU streaming ở res NHỎ,
    tính histogram theo `stride`, rồi chọn keyframe cho từng shot. Trả về danh sách
    global frame index (sắp xếp); rỗng nếu stream lỗi/thiếu (caller fallback)."""
    from .video_io import stream_frames
    if not shots:
        return []
    stride = max(1, int(cfg.get("action_candidate_stride", 5)))
    need_hist = str(cfg.get("keyframe_signal", "motion")).lower() != "motion"
    W, H, n_total = _video_dims(video_path)
    w, h = _scaled_wh(W, H, int(cfg.get("action_hist_long_side", 192)))

    def _collect(use_gpu):
        raw, seen = [], 0
        for fidx, frame in stream_frames(video_path, w, h, use_gpu=use_gpu):
            seen = fidx + 1
            if fidx % stride:
                continue
            s48 = cv2.resize(frame, (48, 27), interpolation=cv2.INTER_AREA)
            raw.append((fidx, _hist(frame) if need_hist else None,
                        cv2.cvtColor(s48, cv2.COLOR_RGB2GRAY)))
        return raw, seen

    raw, seen = _collect(use_gpu=True)
    if n_total and seen < 0.9 * n_total:            # GPU thiếu frame -> đọc lại CPU
        raw, seen = _collect(use_gpu=False)

    return _dispatch_action(raw, shots, cfg, video_id=video_id)


def _action_select_all(cand_by_shot, shots, cfg) -> List[int]:
    """Chọn keyframe action từ các ỨNG VIÊN (fidx, histogram) đã gom sẵn theo shot.
    Tách khỏi decode để dùng chung cho cả action_global lẫn combined_autoshot_action."""
    n_per = int(cfg.get("keyframes_per_shot", 10))
    thr = float(cfg.get("action_change_threshold", 0.30))
    fps = shots[0].fps if shots else 25.0
    min_gap = max(1, int(cfg.get("action_min_gap_sec", 0.4) * fps))
    selected: List[int] = []
    for shot in shots:
        cands = cand_by_shot.get(shot.index) or []
        if not cands:                               # shot không có ứng viên -> lấy frame giữa
            selected.append((shot.start_frame + shot.end_frame) // 2)
            continue
        kept = [cands[0]]                           # luôn giữ frame đầu shot
        for idx, hh in cands[1:]:
            if len(kept) >= n_per:
                break
            if abs(idx - kept[-1][0]) < min_gap:    # quá sát thời gian -> bỏ
                continue
            if _hist_diff(hh, kept[-1][1]) > thr:   # THỰC SỰ khác mới giữ
                kept.append((idx, hh))
        if len(kept) < n_per and cands[-1][0] != kept[-1][0] \
                and _hist_diff(cands[-1][1], kept[-1][1]) > thr * 0.5:
            kept.append(cands[-1])                  # cố lấy frame cuối shot (kết diễn biến)
        selected.extend(k[0] for k in kept)
    return sorted(set(selected))


# ---- A + B: chọn keyframe theo CHUYỂN ĐỘNG (optical flow) + đo HƯỚNG -> motion hint ----
def _flow_vec(g0: np.ndarray, g1: np.ndarray):
    """Optical flow Farneback trung bình -> (dx, dy, magnitude, coherence).
    dx>0: NỘI DUNG dịch PHẢI; dy>0: dịch XUỐNG. coherence ~1 = cả khung dịch đều
    (camera lia); ~0 = chuyển động cục bộ (chủ thể di chuyển)."""
    f = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 2, 9, 2, 5, 1.1, 0)
    dx, dy = float(f[..., 0].mean()), float(f[..., 1].mean())
    mag = float(np.hypot(f[..., 0], f[..., 1]).mean())
    coh = (dx * dx + dy * dy) ** 0.5 / (mag + 1e-6)
    return dx, dy, mag, coh


def _dir_phrase(dx: float, dy: float, mag: float, coh: float, still: float) -> str:
    """Diễn giải flow -> câu tả. ⚠️ flow đo chuyển động của NỘI DUNG; nếu cả khung dịch
    đều (coherence cao) là CAMERA lia -> camera đi NGƯỢC chiều nội dung."""
    if mag < still:
        return "gần như tĩnh"
    axis = []
    if abs(dx) >= 0.1 and abs(dx) >= abs(dy) * 0.6:
        axis.append("phải" if dx > 0 else "trái")
    if abs(dy) >= 0.1 and abs(dy) >= abs(dx) * 0.6:
        axis.append("xuống" if dy > 0 else "lên")
    lvl = "mạnh" if mag > 1.6 else ("vừa" if mag > 0.8 else "nhẹ")
    if not axis:
        return f"chuyển động tại chỗ ({lvl})"
    if coh > 0.5:                     # cả khung dịch đều -> CAMERA lia (ngược chiều nội dung)
        flip = {"phải": "trái", "trái": "phải", "lên": "xuống", "xuống": "lên"}
        return f"camera lia sang {'/'.join(flip[a] for a in axis)} ({lvl})"
    return f"chủ thể/vật di chuyển sang {'/'.join(axis)} ({lvl})"


def _action_select_motion(cand_by_shot, shots, cfg):
    """Chọn keyframe THEO CHUYỂN ĐỘNG + sinh motion hint cho từng shot.
    cand_by_shot: {shot_index: [(fidx, gray48), ...]}.
    Trả về (selected_idxs, {shot_index: hint_text}, {shot_index: per_flow}).
    `per_flow` = list (dx,dy,mag,coh) mỗi bước -> TRẢ RA để event_segmenter dùng lại
    (khỏi tính lại optical flow, phần đắt nhất).

    (A) Số keyframe theo NGÂN SÁCH motion (tĩnh -> 2), đặt theo ARC-LENGTH motion tích
    lũy -> dồn vào đoạn nhiều action. (B) chia shot thành đoạn đều -> tóm tắt hướng."""
    n_per = int(cfg.get("keyframes_per_shot", 10))
    fps = shots[0].fps if shots else 25.0
    min_gap = max(1, int(cfg.get("action_min_gap_sec", 0.4) * fps))
    step_thr = float(cfg.get("action_motion_step", 5.0))     # tổng motion để CÓ THÊM 1 keyframe
    still = float(cfg.get("action_still_level", 0.5))         # mức trung bình dưới đây = tĩnh
    n_hint = max(1, int(cfg.get("action_hint_segments", 3)))  # số đoạn tóm tắt trong hint
    # TRẦN LỖ HỔNG: không để trống quá ngần này giây trong 1 shot (0 = tắt). Bù cho việc
    # số keyframe chỉ nhìn LƯỢNG CHUYỂN ĐỘNG, không nhìn THỜI LƯỢNG (shot tĩnh dài vẫn chỉ
    # 2 keyframe). Đo keyframe BTC (L21_V001/2/3): trần 7.00-7.04s -> mặc định 7.0.
    max_gap = int(float(cfg.get("action_max_gap_sec", 7.0)) * fps)

    selected, hints, per_by_shot = [], {}, {}
    for shot in shots:
        cands = cand_by_shot.get(shot.index) or []
        if len(cands) < 2:
            selected.append((shot.start_frame + shot.end_frame) // 2)
            hints[shot.index] = "gần như tĩnh"
            per_by_shot[shot.index] = [(0.0, 0.0, 0.0, 0.0)] * len(cands)
            continue
        fidxs = [c[0] for c in cands]
        grays = [c[1] for c in cands]
        per = [(0.0, 0.0, 0.0, 0.0)]        # (dx,dy,mag,coh) cho mỗi bước
        cum = [0.0]                         # motion tích lũy
        for i in range(1, len(cands)):
            v = _flow_vec(grays[i - 1], grays[i])
            per.append(v); cum.append(cum[-1] + v[2])
        per_by_shot[shot.index] = per        # GIỮ LẠI cho event_segmenter (khỏi tính lại flow)
        total = cum[-1]
        mean_mag = total / (len(cands) - 1)

        # (A) chọn keyframe
        if mean_mag < still:                # cả shot gần như tĩnh -> chỉ 2 keyframe (đầu+cuối)
            kept = [0, len(cands) - 1]
        else:
            n_kf = max(2, min(n_per, int(round(1 + total / step_thr))))
            kept = []
            for k in range(n_kf):           # đặt theo arc-length: cum = total*k/(n_kf-1)
                tgt = total * k / (n_kf - 1)
                j = 0
                while j < len(cum) - 1 and cum[j] < tgt:
                    j += 1
                if not kept or fidxs[j] - fidxs[kept[-1]] >= min_gap:
                    kept.append(j)
            if kept[-1] != len(cands) - 1:
                kept.append(len(cands) - 1)
        kept = sorted(set(kept))

        # áp TRẦN LỖ HỔNG: chèn ứng viên vào mọi khoảng hở > max_gap.
        # CỐ Ý cho phép vượt `keyframes_per_shot` (shot dài cần nhiều keyframe hơn).
        if max_gap > 0 and len(kept) >= 2:
            filled = [kept[0]]
            for j in kept[1:]:
                while fidxs[j] - fidxs[filled[-1]] > max_gap:
                    tgt = fidxs[filled[-1]] + max_gap
                    # ứng viên gần mốc `tgt` nhất, nằm giữa cái vừa giữ và cái kế tiếp
                    nxt = min(range(filled[-1] + 1, j + 1),
                              key=lambda k: abs(fidxs[k] - tgt))
                    if nxt <= filled[-1]:      # không tiến được nữa (ứng viên quá thưa) -> dừng
                        break
                    filled.append(nxt)
                filled.append(j)
            kept = sorted(set(filled))

        selected.extend(fidxs[j] for j in kept)

        # (B) motion hint: chia shot thành n_hint đoạn ĐỀU THỜI GIAN, tóm tắt hướng mỗi đoạn
        m = len(cands)
        lines = []
        for s in range(n_hint):
            a = 1 + s * (m - 1) // n_hint
            b = 1 + (s + 1) * (m - 1) // n_hint
            seg = [i for i in range(a, b) if i < m]
            if not seg:
                continue
            sdx = sum(per[i][0] for i in seg); sdy = sum(per[i][1] for i in seg)
            smag = sum(per[i][2] for i in seg) / len(seg)
            scoh = sum(per[i][3] for i in seg) / len(seg)
            t0, t1 = fidxs[seg[0] - 1] / fps, fidxs[seg[-1]] / fps
            lines.append(f"{t0:.0f}-{t1:.0f}s: {_dir_phrase(sdx, sdy, smag, scoh, still)}")
        hints[shot.index] = " | ".join(lines) if lines else "gần như tĩnh"
    return sorted(set(selected)), hints, per_by_shot


def _dispatch_action(cand_raw, shots, cfg, video_id: str = "") -> List[int]:
    """cand_raw = [(fidx, hist_or_None, gray48), ...] toàn video theo stride. Gom theo shot,
    chọn keyframe theo `keyframe_signal` (motion | histogram) và gán motion_hint cho shots.
    Khi `event_segmentation` (mặc định BẬT): cắt mỗi shot thành CHUỖI SỰ KIỆN + gom shot
    liền kề thành SCENE (dùng lại optical flow đã tính, không decode thêm)."""
    signal = str(cfg.get("keyframe_signal", "motion")).lower()
    shots_sorted = sorted(shots, key=lambda s: s.start_frame)

    def _group(pick):
        cbs = {s.index: [] for s in shots}
        si = 0
        for fidx, hist, gray in cand_raw:
            while si < len(shots_sorted) and fidx > shots_sorted[si].end_frame:
                si += 1
            if si >= len(shots_sorted):
                break
            if fidx >= shots_sorted[si].start_frame:
                cbs[shots_sorted[si].index].append((fidx, pick(hist, gray)))
        return cbs

    if signal == "histogram":
        # nhánh histogram KHÔNG có optical flow -> không cắt được sự kiện theo động học
        return _action_select_all(_group(lambda h, g: h), shots, cfg)

    gray_by_shot = _group(lambda h, g: g)
    idxs, hints, per_by_shot = _action_select_motion(gray_by_shot, shots, cfg)
    for s in shots:                      # gắn hint chuyển động vào shot -> dùng cho prompt caption
        s.motion_hint = hints.get(s.index, "")

    if cfg.get("event_segmentation", True):
        from .event_segmenter import segment_events, group_scenes
        ev = segment_events(video_id, gray_by_shot, shots, per_by_shot, cfg)
        scene_of = group_scenes(shots, gray_by_shot, cfg)
        for s in shots:
            s.events = ev.get(s.index, [])
            s.scene_id = scene_of.get(s.index, s.index)
    return idxs


def combined_autoshot_dake(video_path, video_id, sd_cfg, kf_cfg):
    """GỘP shot detection (AutoShot) + DAKE bằng 1 LƯỢT decode chung: mỗi frame vừa đo
    JPEG (DAKE) vừa thu nhỏ 48x27 (AutoShot). Trả về (shots, keyframes); None nếu decode thiếu."""
    import cv2
    from tqdm import tqdm
    from .video_io import stream_frames
    from .shot_detector import autoshot_infer

    W, H, n_total = _video_dims(video_path)
    long_side = int(kf_cfg.get("dake_size_long_side", 192))
    w, h = _scaled_wh(W, H, long_side)
    quality = int(kf_cfg.get("jpeg_quality", 90))
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()
    use_gpu = bool(sd_cfg.get("gpu_decode", True))
    quiet = bool(sd_cfg.get("quiet", False))

    sizes, small = [], []
    bar = None if quiet else tqdm(total=n_total or None, desc="[gộp] decode 1 lượt", unit="frame")
    for _, fr in stream_frames(video_path, w, h, use_gpu=use_gpu):   # fr = RGB
        ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        sizes.append(int(buf.nbytes) if ok else 0)
        small.append(cv2.resize(fr, (48, 27), interpolation=cv2.INTER_AREA))
        if bar:
            bar.update(1)
    if bar:
        bar.close()

    if n_total and len(sizes) < 0.9 * n_total:      # GPU decode thiếu -> để caller fallback tách rời
        return None

    import numpy as np
    shots = autoshot_infer(np.asarray(small, dtype=np.uint8),
                           sd_cfg.get("autoshot_prob_threshold", 0.5), fps, quiet)
    idxs = dake_select_from_sizes(sizes, fps, kf_cfg)
    keyframes = _dake_read_and_map(video_path, video_id, idxs, shots,
                                   kf_cfg.get("resize_long_side", 720))
    return shots, keyframes


def _merge_adjacent_similar_shots(shots, small, rms_thr: float, max_sec: float):
    """Gộp các shot LIỀN KỀ gần-giống nhau (RMS mean ảnh 48x27 < rms_thr) thành 1 shot,
    vá việc AutoShot cắt vụn một cảnh liên tục -> nhiều keyframe nhìn như trùng.
    Guard `max_sec`: KHÔNG gộp nếu shot kết quả dài quá ngần này giây (tránh nuốt cả đoạn
    tin dài). Chỉ so shot LIỀN KỀ nên 2 cảnh giống nhau ở 2 thời điểm XA KHÔNG bị gộp nhầm."""
    from .shot_detector import Shot
    if len(shots) < 2:
        return shots
    small = np.asarray(small, dtype=np.float32) / 255.0
    n = len(small)
    order = sorted(shots, key=lambda s: s.start_frame)

    def mean_frame(s):
        a, b = max(0, s.start_frame), min(n, s.end_frame + 1)
        return small[a:b].mean(axis=0) if b > a else None

    means = [mean_frame(s) for s in order]
    groups = [[0]]
    for i in range(1, len(order)):
        prev = groups[-1][-1]                         # shot LIỀN TRƯỚC trong nhóm
        g_start = order[groups[-1][0]].start_frame
        dur = (order[i].end_frame - g_start + 1) / (order[i].fps or 25.0)
        rms = (float(np.sqrt(np.mean((means[prev] - means[i]) ** 2)))
               if means[prev] is not None and means[i] is not None else 1.0)
        if rms < rms_thr and dur <= max_sec:
            groups[-1].append(i)
        else:
            groups.append([i])
    return [Shot(index=k, start_frame=order[g[0]].start_frame,
                 end_frame=order[g[-1]].end_frame, fps=order[g[0]].fps)
            for k, g in enumerate(groups)]


def combined_autoshot_action(video_path, video_id, sd_cfg, kf_cfg):
    """GỘP shot detection (AutoShot) + trích keyframe "action" bằng 1 LƯỢT decode chung:
    mỗi frame vừa thu nhỏ 48x27 (AutoShot) vừa tính histogram theo stride (action).
    Trả về (shots, keyframes); None nếu decode thiếu (caller fallback tách rời)."""
    import cv2
    import numpy as np
    from tqdm import tqdm
    from .video_io import stream_frames
    from .shot_detector import autoshot_infer

    W, H, n_total = _video_dims(video_path)
    w, h = _scaled_wh(W, H, int(kf_cfg.get("action_hist_long_side", 192)))
    stride = max(1, int(kf_cfg.get("action_candidate_stride", 5)))
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()
    use_gpu = bool(sd_cfg.get("gpu_decode", True))
    quiet = bool(sd_cfg.get("quiet", False))

    need_hist = str(kf_cfg.get("keyframe_signal", "motion")).lower() != "motion"
    small, cand_raw = [], []    # small: 48x27 cho AutoShot; cand_raw: (fidx, hist|None, gray48) theo stride
    bar = None if quiet else tqdm(total=n_total or None, desc="[gộp] decode 1 lượt", unit="frame")
    for fidx, fr in stream_frames(video_path, w, h, use_gpu=use_gpu):    # fr = RGB
        s48 = cv2.resize(fr, (48, 27), interpolation=cv2.INTER_AREA)
        small.append(s48)
        if fidx % stride == 0:
            cand_raw.append((fidx, _hist(fr) if need_hist else None,
                             cv2.cvtColor(s48, cv2.COLOR_RGB2GRAY)))
        if bar:
            bar.update(1)
    if bar:
        bar.close()

    if n_total and len(small) < 0.9 * n_total:      # GPU decode thiếu -> caller fallback tách rời
        return None

    shots = autoshot_infer(np.asarray(small, dtype=np.uint8),
                           sd_cfg.get("autoshot_prob_threshold", 0.5), fps, quiet)
    # GỘP shot liền kề gần-giống nhau (giảm AutoShot cắt vụn cảnh liên tục -> ít keyframe trùng)
    if kf_cfg.get("merge_similar_shots", False):
        n0 = len(shots)
        shots = _merge_adjacent_similar_shots(
            shots, small, float(kf_cfg.get("merge_shot_rms", 0.06)),
            float(kf_cfg.get("merge_shot_max_sec", 20.0)))
        if not quiet and len(shots) < n0:
            print(f"      [gộp shot] {n0} -> {len(shots)} shot (gộp cảnh liền kề gần-giống)", flush=True)
    # chọn keyframe + gán motion_hint + CẮT CHUỖI SỰ KIỆN cho shots
    idxs = _dispatch_action(cand_raw, shots, kf_cfg, video_id=video_id)
    keyframes = _dake_read_and_map(video_path, video_id, idxs, shots,
                                   kf_cfg.get("resize_long_side", 720))
    return shots, keyframes


def dedup_keyframes(keyframes: List[Keyframe], threshold: float,
                    max_gap_frames: int = 0, min_fill_diff: float = 0.0) -> List[Keyframe]:
    """Lọc để keyframe lưu lại THẬT SỰ KHÁC NHAU trong từng shot: chỉ giữ nếu khác MỌI
    keyframe đã giữ > `threshold` (histogram HSV, bất biến không gian -> camera lia không
    tính là 'khác'). Luôn giữ keyframe đầu mỗi shot.

    `max_gap_frames` (0 = tắt) — LỐI THOÁT theo THỜI GIAN: giữ keyframe kể cả khi giống
    frame trước, nếu bỏ sẽ để lại lỗ hổng dài hơn ngần này frame. CẦN vì hai luật chống nhau:
    trần lỗ hổng chèn frame vào giữa shot TĨNH, mà frame tĩnh giống nhau -> dedup xoá sạch."""
    if not keyframes or threshold <= 0:
        return keyframes
    from collections import defaultdict
    by_shot = defaultdict(list)
    for kf in keyframes:
        by_shot[kf.shot_index].append(kf)
    out: List[Keyframe] = []
    for si in sorted(by_shot):
        allkf = sorted(by_shot[si], key=lambda k: k.frame_index)
        kept, kept_h = [], []
        for kf in allkf:
            h = _hist(kf.image)                     # kf.image BGR -> _hist tự BGR2HSV
            if not kept or all(_hist_diff(h, kh) > threshold for kh in kept_h):
                kept.append(kf); kept_h.append(h)

        # LƯỢT 2 — BÙ cho đủ trần: đi qua keyframe SỐNG SÓT, chỗ nào hở quá trần thì lấy lại
        # keyframe đã loại ở lượt 1. PHẢI làm SAU dedup (làm trước thì dedup xoá đúng frame
        # vừa chèn vì cảnh tĩnh ảnh giống nhau).
        if max_gap_frames > 0 and len(kept) < len(allkf):
            # PHẢI phủ tới tận CUỐI shot: shot TĨNH dài bị dedup gộp về 1 keyframe -> không
            # còn "cặp" để chèn; chỉ lặp trên `kept` sẽ bỏ sót đúng shot cần bù nhất.
            anchors = list(kept)
            if allkf[-1].frame_index - anchors[-1].frame_index > max_gap_frames:
                anchors.append(allkf[-1])
            filled = [anchors[0]]
            for nxt in anchors[1:]:
                while nxt.frame_index - filled[-1].frame_index > max_gap_frames:
                    tgt = filled[-1].frame_index + max_gap_frames
                    mid = [k for k in allkf
                           if filled[-1].frame_index < k.frame_index < nxt.frame_index]
                    if not mid:
                        break
                    cand = min(mid, key=lambda k: abs(k.frame_index - tgt))
                    # CHỈ chèn frame phủ nếu nội dung KHÁC frame vừa giữ > min_fill_diff ->
                    # shot TĨNH không chèn -> hết keyframe trùng. min_fill_diff=0 = hành vi CŨ.
                    if min_fill_diff > 0 and _hist_diff(_hist(cand.image),
                                                        _hist(filled[-1].image)) <= min_fill_diff:
                        break
                    filled.append(cand)
                filled.append(nxt)
            kept = filled
        out.extend(kept)
    return out


def extract_keyframes(video_path: str, video_id: str, shots: List[Shot], cfg: dict) -> List[Keyframe]:
    """cfg = block `keyframe` trong config.yaml."""
    strategy = cfg.get("strategy", "adaptive").lower()
    long_side = cfg.get("resize_long_side", 720)

    # DAKE gốc: chọn keyframe TOÀN VIDEO rồi gán vào shot chứa nó (phần còn lại của
    # pipeline không đổi).
    if strategy == "dake":
        fps = shots[0].fps if shots else 25.0
        idxs = set(dake_global(video_path, fps, cfg))
        return _dake_read_and_map(video_path, video_id, idxs, shots, long_side)

    # "action": chọn keyframe THEO SHOT bằng 1 LƯỢT decode GPU, fallback cv2 per-shot
    if strategy == "action":
        idxs = action_global(video_path, shots, cfg, video_id=video_id)
        if idxs:
            return _dake_read_and_map(video_path, video_id, idxs, shots, long_side)
        # (stream lỗi) -> rơi xuống vòng cv2 per-shot bên dưới

    # Các chiến lược THEO SHOT (luồng cũ)
    cap = cv2.VideoCapture(video_path)
    keyframes: List[Keyframe] = []

    for shot in shots:
        if strategy == "middle":
            idxs = _pick_middle(cap, shot)
        elif strategy == "uniform":
            idxs = _pick_uniform(cap, shot, cfg.get("uniform_step", 24))
        elif strategy == "dake_per_shot":
            idxs = _pick_dake(cap, shot, cfg.get("jpeg_quality", 90),
                              cfg.get("adaptive_max_per_shot", 3))
        elif strategy == "clip_reldiff":
            idxs = _pick_clip_reldiff(cap, shot, cfg)
        elif strategy == "action":
            min_gap = int(cfg.get("action_min_gap_sec", 0.4) * shot.fps)
            idxs = _pick_action(cap, shot,
                                cfg.get("keyframes_per_shot", 10),
                                cfg.get("action_change_threshold", 0.30),
                                max(1, min_gap))
        else:  # adaptive
            min_gap = int(cfg.get("adaptive_min_gap_sec", 1.0) * shot.fps)
            idxs = _pick_adaptive(cap, shot,
                                  cfg.get("adaptive_max_per_shot", 3),
                                  cfg.get("adaptive_hist_threshold", 0.35),
                                  max(1, min_gap))

        # đọc ảnh thật cho từng index đã chọn
        for fidx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = _resize_long_side(frame, long_side)
            keyframes.append(Keyframe(
                video_id=video_id,
                shot_index=shot.index,
                frame_index=fidx,
                time_sec=fidx / shot.fps,
                image=frame,
            ))

    cap.release()
    return keyframes
