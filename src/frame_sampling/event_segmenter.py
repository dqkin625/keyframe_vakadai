"""
CHIA SHOT THÀNH CHUỖI SỰ KIỆN CÓ THỨ TỰ (sub-shot event segmentation).

Shot là đơn vị DỰNG PHIM, không phải đơn vị HÀNH ĐỘNG: một shot có thể chứa cả chuỗi
"chạy đà -> giậm nhảy -> bay qua xà -> tiếp đất". Module cắt MỖI SHOT thành các đoạn hành
động có thứ tự (start/end/anchor frame thật) để tầng truy xuất khớp chuỗi N sự kiện TRAKE.

HAI TÍN HIỆU cộng lại — score(t) = a*PA(t) + (1-a)*TURN(t):
  (1) PA — PredictAbility [NGUỒN: Shou et al., "Generic Event Boundary Detection", ICCV 2021,
      arXiv:2101.10511 — mục 5.2, baseline không giám sát]. Ranh giới sự kiện ở chỗ "khó đoán":
      đo L2² giữa đặc trưng TRUNG BÌNH w frame trước và w frame sau; ranh giới = cực đại địa phương.
  (2) TURN — điểm ngoặt động học [tự thiết kế, KHÔNG lấy từ paper]. PA nhìn ngoại hình; 4 khoảnh
      khắc TRAKE (chạm/rời đất, cực trị) là ĐIỂM TỚI HẠN CHUYỂN ĐỘNG với ngoại hình gần y hệt.
      Đo trực tiếp: đổi ĐỘ LỚN + đổi HƯỚNG chuyển động, dùng flow sẵn ở `_flow_vec` (KHÔNG tính lại).

GOM SHOT -> SCENE (`group_scenes`): chuỗi TRAKE có thể trải qua nhiều shot liền kề (cắt qua lại
góc máy); gom shot giống nhau thành SCENE để chuỗi không đứt ở ranh giới shot.

⚠️ ĐỘ PHÂN GIẢI: ứng viên theo `action_candidate_stride` (mặc định 5) -> sai số ±stride frame,
sát mép cửa sổ <10 frame của TRAKE. Muốn chính xác tới frame phải chạy `refine_boundaries()`
(stride 1) — "tầng DÀY", cố ý KHÔNG bật khi quét cả kho.

KHÔNG dùng scipy (máy này bị chặn DLL scipy — xem METHODS.md E5). Mọi thứ bằng numpy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
@dataclass
class Event:
    """Một ĐOẠN HÀNH ĐỘNG bên trong shot. Thứ tự thời gian = thứ tự `ord_in_shot`."""
    video_id: str
    shot_index: int
    ord_in_shot: int          # 0,1,2... theo thời gian TRONG shot
    start_frame: int          # frame THẬT trong video (không phải số thứ tự keyframe)
    end_frame: int            # inclusive
    anchor_frame: int         # frame ĐẠI DIỆN (điểm ngoặt) - ứng viên nộp cho TRAKE
    fps: float
    boundary_score: float = 0.0   # độ mạnh của ranh giới MỞ ĐẦU đoạn này (0 = đầu shot)
    motion: str = ""              # tóm tắt chuyển động của đoạn (dùng cho prompt/lọc)
    mag_mean: float = 0.0         # độ lớn chuyển động trung bình
    scene_id: int = -1            # điền sau bởi group_scenes()
    action: str = ""              # nhãn ngữ nghĩa (VLM điền sau; rỗng nếu chưa caption)

    @property
    def start_time(self) -> float:
        return self.start_frame / self.fps

    @property
    def end_time(self) -> float:
        return (self.end_frame + 1) / self.fps

    @property
    def anchor_time(self) -> float:
        return self.anchor_frame / self.fps


# --------------------------------------------------------------------------- #
# Bộ lọc 1D bằng numpy (thay scipy)
# --------------------------------------------------------------------------- #
def _gauss_kernel(sigma: float) -> np.ndarray:
    r = max(1, int(round(3.0 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / float(k.sum())


def _smooth1d(y: np.ndarray, sigma: float) -> np.ndarray:
    """Làm trơn Gauss, đệm biên bằng giá trị mép (không tạo ranh giới giả ở đầu/cuối)."""
    if sigma <= 0 or len(y) < 3:
        return y.astype(np.float32)
    k = _gauss_kernel(sigma)
    pad = len(k) // 2
    return np.convolve(np.pad(y.astype(np.float32), pad, mode="edge"), k, mode="valid")


def _norm01(y: np.ndarray) -> np.ndarray:
    """Chuẩn hoá về [0,1]. Mảng phẳng -> trả 0 hết (không có ranh giới nào)."""
    if len(y) == 0:
        return y
    lo, hi = float(y.min()), float(y.max())
    if hi - lo < 1e-9:
        return np.zeros_like(y, dtype=np.float32)
    return ((y - lo) / (hi - lo)).astype(np.float32)


def _norm_floor(y: np.ndarray, floor: float) -> np.ndarray:
    """Chuẩn hoá về [0,1] NHƯNG chia cho `max(biên_độ_thật, floor)` thay vì biên độ thật.

    ⚠️ ĐÃ TỪNG SAI NẶNG (METHODS I7-b): `_norm01` kéo giãn MỌI đường cong ra trọn [0,1], kể cả
    shot ĐỨNG YÊN biên độ chỉ là nhiễu -> bịa ra hành động. `_norm_floor` giữ shot dưới `floor`
    ở giá trị NHỎ (không tới ngưỡng) nên không bị cắt; liên tục, không phải cổng cứng bật/tắt."""
    if len(y) == 0:
        return y
    lo, hi = float(y.min()), float(y.max())
    denom = max(hi - lo, float(floor))
    if denom < 1e-12:
        return np.zeros_like(y, dtype=np.float32)
    return np.clip((y - lo) / denom, 0.0, 1.0).astype(np.float32)


def _local_maxima(y: np.ndarray, thr: float) -> List[int]:
    """Cực đại địa phương chặt (y[t] > y[t-1] và y[t] >= y[t+1]) và vượt ngưỡng.
    Bỏ hai đầu mảng: đầu/cuối shot đã LÀ ranh giới sẵn, không cần phát hiện lại."""
    out = []
    for t in range(1, len(y) - 1):
        if y[t] >= thr and y[t] > y[t - 1] and y[t] >= y[t + 1]:
            out.append(t)
    return out


# --------------------------------------------------------------------------- #
# Tín hiệu 1 — PA (GEBD, ICCV 2021 arXiv:2101.10511)
# --------------------------------------------------------------------------- #
def _pa_curve(feats: np.ndarray, w: int) -> np.ndarray:
    """feats: (m, d) đặc trưng theo thời gian. Trả về đường cong 'khó đoán' (m,).

    PA(t) = || mean(feats[t-w : t]) - mean(feats[t+1 : t+1+w]) ||² / d
    Dùng prefix sum nên chi phí O(m·d), không phải O(m·w·d).
    CHIA CHO d (số chiều): để giá trị không phụ thuộc kích thước ảnh -> ngưỡng `event_pa_floor`
    giữ nguyên ý nghĩa khi đổi độ phân giải. Chia hằng số KHÔNG đổi vị trí cực đại."""
    m = len(feats)
    out = np.zeros(m, dtype=np.float32)
    if m < 3:
        return out
    d = float(feats.shape[1]) or 1.0
    cs = np.cumsum(np.vstack([np.zeros((1, feats.shape[1]), np.float32), feats]), axis=0)

    def _mean(a: int, b: int):            # trung bình feats[a:b], giả định b > a
        return (cs[b] - cs[a]) / float(b - a)

    for t in range(m):
        a0, a1 = max(0, t - w), t
        b0, b1 = min(m, t + 1), min(m, t + 1 + w)
        if a1 <= a0 or b1 <= b0:
            continue
        v = _mean(a0, a1) - _mean(b0, b1)
        out[t] = float(np.dot(v, v)) / d
    return out


# --------------------------------------------------------------------------- #
# Tín hiệu 2 — TURN: điểm ngoặt động học (tự thiết kế)
# --------------------------------------------------------------------------- #
def _turn_curve(per: Sequence[Tuple[float, float, float, float]], w: int) -> np.ndarray:
    """per[i] = (dx, dy, mag, coh) cho BƯỚC i (chuyển động frame i-1 -> i); per[0] = 0.

    Trả về (m,) độ mạnh 'điểm ngoặt chuyển động', gồm 2 thành phần:
      - ĐỔI ĐỘ LỚN : |mean(mag sau) - mean(mag trước)| -> DỪNG LẠI / BẬT LÊN / VA CHẠM
      - ĐỔI HƯỚNG  : (1 - cos(vector trước, vector sau))/2 in [0,1] -> ĐẢO CHIỀU / QUAY ĐẦU / NẢY

    ⚠️ per[0] là PHẦN TỬ ĐỆM (không có frame trước để đo flow), KHÔNG phải phép đo thật. PHẢI loại
    khỏi cửa sổ "trước", nếu không mọi shot sinh ranh giới GIẢ ở đầu (đệm 0 vs chuyển động thật ->
    chênh lệch tối đa). Đã dính lỗi này, test_events.py [2]/[3] bắt được."""
    m = len(per)
    out = np.zeros(m, dtype=np.float32)
    if m < 3:
        return out
    dxs = np.array([p[0] for p in per], dtype=np.float32)
    dys = np.array([p[1] for p in per], dtype=np.float32)
    mags = np.array([p[2] for p in per], dtype=np.float32)

    for t in range(m):
        a0, a1 = max(1, t - w), t            # max(1, ...) -> BỎ phần tử đệm per[0]
        b0, b1 = min(m, t + 1), min(m, t + 1 + w)
        if a1 <= a0 or b1 <= b0:
            continue
        m_before, m_after = float(mags[a0:a1].mean()), float(mags[b0:b1].mean())
        d_mag = abs(m_after - m_before)

        vb = np.array([dxs[a0:a1].mean(), dys[a0:a1].mean()], dtype=np.float32)
        va = np.array([dxs[b0:b1].mean(), dys[b0:b1].mean()], dtype=np.float32)
        nb, na = float(np.linalg.norm(vb)), float(np.linalg.norm(va))
        if nb < 1e-6 or na < 1e-6:
            d_dir = 0.0        # một bên gần như đứng yên -> hướng vô nghĩa, chỉ tính độ lớn
        else:
            cos = float(np.dot(vb, va) / (nb * na))
            d_dir = (1.0 - max(-1.0, min(1.0, cos))) / 2.0

        # d_dir chỉ có ý nghĩa khi CÓ chuyển động: nhân cường độ để cảnh tĩnh không sinh
        # ranh giới giả (hướng của nhiễu là ngẫu nhiên -> cos dao động mạnh).
        out[t] = d_mag + d_dir * min(m_before, m_after)
    return out


# --------------------------------------------------------------------------- #
def _describe_motion(per_slice: Sequence[Tuple[float, float, float, float]],
                     still: float) -> Tuple[str, float]:
    """Tóm tắt chuyển động của MỘT đoạn -> (câu tả, độ lớn trung bình).
    Dùng lại quy ước của `_dir_phrase`: coherence cao = cả khung dịch đều = CAMERA lia."""
    if not per_slice:
        return "gần như tĩnh", 0.0
    dx = float(np.mean([p[0] for p in per_slice]))
    dy = float(np.mean([p[1] for p in per_slice]))
    mag = float(np.mean([p[2] for p in per_slice]))
    coh = float(np.mean([p[3] for p in per_slice]))
    from .keyframe_extractor import _dir_phrase      # import trễ: tránh vòng lặp import
    return _dir_phrase(dx, dy, mag, coh, still), mag


# --------------------------------------------------------------------------- #
# API chính — cắt 1 shot thành chuỗi sự kiện
# --------------------------------------------------------------------------- #
def segment_shot_events(video_id: str, shot, fidxs: Sequence[int],
                        grays: Sequence[np.ndarray],
                        per: Sequence[Tuple[float, float, float, float]],
                        cfg: dict) -> List[Event]:
    """Cắt MỘT shot thành các Event có thứ tự.

    fidxs[i] = frame THẬT của ứng viên i (đã theo stride)
    grays[i] = ảnh xám 48x27 của ứng viên i
    per[i]   = flow (dx,dy,mag,coh) của bước i (per[0] = 0) — TÍNH SẴN, không tính lại
    cfg      = block `keyframe` của config.yaml
    """
    fps = float(shot.fps) or 25.0
    m = len(fidxs)
    min_sec = float(cfg.get("event_min_sec", 0.4))
    max_ev = int(cfg.get("event_max_per_shot", 8))
    alpha = float(cfg.get("event_pa_weight", 0.5))
    thr = float(cfg.get("event_boundary_threshold", 0.25))
    sigma = float(cfg.get("event_smooth_sigma", 1.0))
    still = float(cfg.get("action_still_level", 0.5))

    def _one_event() -> List[Event]:
        """Shot quá ngắn / quá ít ứng viên -> coi cả shot là MỘT sự kiện."""
        desc, mg = _describe_motion(per[1:] if m > 1 else [], still)
        return [Event(video_id=video_id, shot_index=shot.index, ord_in_shot=0,
                      start_frame=shot.start_frame, end_frame=shot.end_frame,
                      anchor_frame=(shot.start_frame + shot.end_frame) // 2,
                      fps=fps, boundary_score=0.0, motion=desc, mag_mean=mg)]

    # cần tối thiểu 4 mẫu để "trước/sau" có nghĩa
    if m < 4:
        return _one_event()

    # cửa sổ trước/sau: đủ dài để ổn định nhưng không nuốt mất sự kiện ngắn
    stride = max(1, int(cfg.get("action_candidate_stride", 5)))
    w = max(1, int(round(float(cfg.get("event_window_sec", 0.5)) * fps / stride)))
    w = min(w, max(1, m // 3))

    # NGƯỠNG SÀN: shot có biên độ tín hiệu dưới sàn = KHÔNG có cấu trúc sự kiện -> không cắt.
    # Số mặc định đo thật trên L21_V001 (xem METHODS I7-b).
    pa_floor = float(cfg.get("event_pa_floor", 0.03))
    turn_floor = float(cfg.get("event_turn_floor", 0.8))

    feats = np.asarray([g.astype(np.float32).ravel() / 255.0 for g in grays], dtype=np.float32)
    pa = _norm_floor(_smooth1d(_pa_curve(feats, w), sigma), pa_floor)
    turn = _norm_floor(_smooth1d(_turn_curve(per, w), sigma), turn_floor)
    score = alpha * pa + (1.0 - alpha) * turn

    # KHOẢNG CÁCH TỐI THIỂU (đơn vị: số mẫu ứng viên).
    # Dùng CEIL chứ KHÔNG round: `event_min_sec` là mức TỐI THIỂU nên sai số phải lệch LÊN.
    min_gap = max(1, int(math.ceil(min_sec * fps / stride)))

    # Ranh giới PHẢI cách đầu/cuối shot ít nhất min_gap, nếu không đoạn đầu/cuối ngắn hơn
    # `event_min_sec` (min_gap cũ chỉ ép giữa các ranh giới, quên 2 mép shot).
    cands = [t for t in _local_maxima(score, thr) if min_gap <= t <= m - 1 - min_gap]
    if not cands:
        return _one_event()

    # sắp theo độ mạnh, giữ dần, ép khoảng cách tối thiểu giữa 2 ranh giới
    cands.sort(key=lambda t: -score[t])
    kept: List[int] = []
    for t in cands:
        if len(kept) >= max_ev - 1:
            break
        if all(abs(t - k) >= min_gap for k in kept):
            kept.append(t)
    if not kept:
        return _one_event()
    kept.sort()

    # ranh giới -> đoạn. Đoạn j chạy từ ranh giới j-1 tới ranh giới j.
    cut_frames = [shot.start_frame] + [int(fidxs[t]) for t in kept] + [shot.end_frame + 1]
    cut_samples = [0] + kept + [m]

    events: List[Event] = []
    for j in range(len(cut_frames) - 1):
        s_f = cut_frames[j]
        e_f = max(s_f, cut_frames[j + 1] - 1)
        a, b = cut_samples[j], cut_samples[j + 1]
        seg_per = per[max(a, 1):b] if b > max(a, 1) else []
        desc, mg = _describe_motion(seg_per, still)
        # ANCHOR: frame nộp cho TRAKE. Đoạn đầu chưa có điểm ngoặt -> lấy giữa; các đoạn sau
        # lấy ĐÚNG ranh giới (điểm ngoặt), vì khoảnh khắc TRAKE là thời điểm chuyển trạng thái.
        anchor = s_f if j > 0 else (s_f + e_f) // 2
        events.append(Event(
            video_id=video_id, shot_index=shot.index, ord_in_shot=j,
            start_frame=s_f, end_frame=e_f, anchor_frame=anchor, fps=fps,
            boundary_score=float(score[kept[j - 1]]) if j > 0 else 0.0,
            motion=desc, mag_mean=mg,
        ))
    return events


# --------------------------------------------------------------------------- #
def segment_events(video_id: str, cand_by_shot: Dict[int, list], shots,
                   per_by_shot: Dict[int, list], cfg: dict) -> Dict[int, List[Event]]:
    """Cắt sự kiện cho TOÀN BỘ shot của một video. Trả về {shot_index: [Event, ...]}."""
    out: Dict[int, List[Event]] = {}
    for shot in shots:
        cands = cand_by_shot.get(shot.index) or []
        per = per_by_shot.get(shot.index) or []
        fidxs = [c[0] for c in cands]
        grays = [c[1] for c in cands]
        if len(per) != len(fidxs):        # lệch (không nên xảy ra) -> hạ về 1 sự kiện/shot
            per = [(0.0, 0.0, 0.0, 0.0)] * len(fidxs)
        out[shot.index] = segment_shot_events(video_id, shot, fidxs, grays, per, cfg)
    return out


# --------------------------------------------------------------------------- #
# Gom shot liền kề -> SCENE (chuỗi sự kiện trải qua nhiều shot)
# --------------------------------------------------------------------------- #
def group_scenes(shots, cand_by_shot: Dict[int, list], cfg: dict) -> Dict[int, int]:
    """Gom các shot LIỀN KỀ có ngoại hình giống nhau thành SCENE. Trả về {shot_index: scene_id}.

    Vì sao cần: bản tin hay cắt qua lại 2-3 góc máy của CÙNG một sự kiện; khớp TRAKE trong phạm
    vi 1 shot sẽ không đủ N khoảnh khắc.
    ⚠️ HEURISTIC tự thiết kế (không từ paper): L2 giữa ảnh xám TRUNG BÌNH của 2 shot liền nhau —
    gom "cùng bối cảnh", không hiểu "cùng câu chuyện". Tắt bằng `event_scene_grouping: false`."""
    if not cfg.get("event_scene_grouping", True):
        return {s.index: s.index for s in shots}

    thr = float(cfg.get("event_scene_threshold", 0.12))       # L2 trung bình mỗi pixel (0-1)
    max_sec = float(cfg.get("event_scene_max_sec", 60.0))

    order = sorted(shots, key=lambda s: s.start_frame)
    means: Dict[int, Optional[np.ndarray]] = {}
    for s in order:
        cands = cand_by_shot.get(s.index) or []
        if cands:
            means[s.index] = np.mean([c[1].astype(np.float32) / 255.0 for c in cands], axis=0)
        else:
            means[s.index] = None

    scene_of: Dict[int, int] = {}
    sid = 0
    scene_start_t = order[0].start_time if order else 0.0
    for i, s in enumerate(order):
        if i == 0:
            scene_of[s.index] = sid
            continue
        prev = order[i - 1]
        a, b = means.get(prev.index), means.get(s.index)
        if a is None or b is None:
            same = False
        else:
            same = float(np.sqrt(np.mean((a - b) ** 2))) < thr
        if same and (s.end_time - scene_start_t) <= max_sec:
            scene_of[s.index] = sid
        else:
            sid += 1
            scene_start_t = s.start_time
            scene_of[s.index] = sid
    return scene_of


# --------------------------------------------------------------------------- #
# TẦNG DÀY — tinh chỉnh ranh giới tới TỪNG FRAME (chỉ chạy cho video đã khoanh)
# --------------------------------------------------------------------------- #
def refine_boundaries(video_path: str, frames: Sequence[int], fps: float,
                      radius_sec: float = 0.5) -> Dict[int, int]:
    """Giải mã lại vùng lân cận mỗi ranh giới ở STRIDE 1 rồi chốt điểm ngoặt CHÍNH XÁC.

    Quét cả kho dùng stride 5 -> sai số ±5 frame, ăn gần hết cửa sổ TRAKE <10 frame. Hàm này
    KHÔNG chạy khi index cả kho (quá đắt), chỉ chạy cho vài video retrieval đã khoanh.
    Trả về {frame_thô: frame_tinh}; frame không tinh chỉnh được thì giữ nguyên.
    """
    import cv2
    out = {f: f for f in frames}
    if not frames:
        return out
    from .keyframe_extractor import _flow_vec

    r = max(1, int(round(radius_sec * fps)))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return out
    try:
        for f0 in frames:
            lo = max(0, f0 - r)
            cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
            grays, idxs = [], []
            for k in range(lo, f0 + r + 1):
                ok, fr = cap.read()
                if not ok:
                    break
                g = cv2.cvtColor(cv2.resize(fr, (48, 27), interpolation=cv2.INTER_AREA),
                                 cv2.COLOR_BGR2GRAY)
                grays.append(g)
                idxs.append(k)
            if len(grays) < 5:
                continue
            per = [(0.0, 0.0, 0.0, 0.0)]
            for i in range(1, len(grays)):
                per.append(_flow_vec(grays[i - 1], grays[i]))
            turn = _smooth1d(_turn_curve(per, w=max(1, int(0.1 * fps))), 1.0)
            # bỏ 2 mép (cửa sổ trước/sau không đủ) rồi lấy cực đại
            lo_i, hi_i = 2, max(3, len(turn) - 2)
            j = int(np.argmax(turn[lo_i:hi_i])) + lo_i
            out[f0] = int(idxs[j])
    finally:
        cap.release()
    return out


# --------------------------------------------------------------------------- #
def events_to_records(events_by_shot: Dict[int, List[Event]],
                      scene_of: Dict[int, int]) -> List[dict]:
    """Làm phẳng thành list dict theo thứ tự thời gian, gán `scene_id` và `event_ord` (thứ tự
    TRONG SCENE — chuỗi TRAKE cần khớp, có thể trải nhiều shot). Giữ `ord_in_shot` để truy ngược."""
    flat: List[Event] = []
    for si in sorted(events_by_shot):
        for e in events_by_shot[si]:
            e.scene_id = scene_of.get(si, si)
            flat.append(e)
    flat.sort(key=lambda e: (e.start_frame, e.shot_index, e.ord_in_shot))

    ord_in_scene: Dict[int, int] = {}
    rows: List[dict] = []
    for e in flat:
        k = ord_in_scene.get(e.scene_id, 0)
        ord_in_scene[e.scene_id] = k + 1
        rows.append({
            "video_id": e.video_id,
            "scene_id": e.scene_id,
            "shot_index": e.shot_index,
            "event_ord": k,                   # thứ tự trong SCENE (chuỗi TRAKE)
            "ord_in_shot": e.ord_in_shot,     # thứ tự trong SHOT
            "start_frame": e.start_frame,
            "end_frame": e.end_frame,
            "anchor_frame": e.anchor_frame,   # frame ỨNG VIÊN nộp cho TRAKE
            "start_time": round(e.start_time, 3),
            "end_time": round(e.end_time, 3),
            "anchor_time": round(e.anchor_time, 3),
            "fps": round(e.fps, 4),           # fps THEO TỪNG VIDEO (kho có cả 25 và 30fps)
            "duration": round(e.end_time - e.start_time, 3),
            "boundary_score": round(e.boundary_score, 4),
            "motion": e.motion,
            "mag_mean": round(e.mag_mean, 4),
            "action": e.action,               # nhãn ngữ nghĩa (VLM điền; rỗng nếu chưa có)
        })
    return rows
