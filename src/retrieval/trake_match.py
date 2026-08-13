"""
KHỚP CHUỖI cho TRAKE — căn chỉnh N mô tả CÓ THỨ TỰ vào chuỗi ứng viên theo thời gian.

Truy vấn TRAKE = 1 hành động mô tả bằng N khoảnh khắc có thứ tự; mỗi khoảnh khắc nộp 1
frame trong cửa sổ đáp án (thường <10 frame). Lõi hậu xử lý trên ma trận "N mô tả × T
ứng viên", theo DANTE (arXiv:2512.13169): QHĐ có phạt thứ tự thời gian, O(N·T),
backtracking dựng lại chuỗi keyframe. (Tham khảo thêm U-CESE arXiv:2605.23274,
EEIoT arXiv:2512.06334.)

Lõi thuần numpy — KHÔNG cần torch/scipy (máy này chặn DLL scipy, xem METHODS E5).
Encoder (CLIP/e5) chỉ nạp khi gọi `localize_phases` với mô tả bằng chữ; `align_sequence`
nhận thẳng ma trận tương đồng nên test được không cần model.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_NEG = -1e18


def align_sequence(sim: np.ndarray,
                   pos: Optional[Sequence[int]] = None,
                   min_gap: int = 1,
                   lam: float = 0.0) -> Tuple[List[int], float]:
    """Căn chỉnh N mô tả (hàng) vào T ứng viên (cột) đã sắp theo thời gian, tối đa
    tổng tương đồng nhưng ép chỉ số cột KHÔNG GIẢM (thứ tự thời gian không đảo).

    sim      : (N, T) — sim[i, t] = độ hợp giữa khoảnh khắc i và ứng viên t.
    pos      : (T,) vị trí thời gian (frame thật) của ứng viên, TĂNG DẦN. None -> 0..T-1.
    min_gap  : ràng buộc CỨNG — 2 khoảnh khắc liên tiếp cách nhau >= min_gap (đơn vị pos).
               >=1 => strictly tăng => mỗi khoảnh khắc một frame khác nhau, đúng thứ tự.
    lam      : phạt MỀM độ trải (DANTE λ). Tổng phạt = lam·(độ trải chuỗi); lam>0 kéo
               chuỗi lại gần nhau, chống khớp rải khắp video. Mặc định 0 (chỉ ép thứ tự).

    Trả về (indices, score): indices[i] = cột chọn cho khoảnh khắc i; score = tổng đã trừ phạt.

    QHĐ chứ không tham lam: argmax từng hàng DỄ VI PHẠM THỨ TỰ; QHĐ tối ưu toàn cục dưới
    ràng buộc thứ tự. O(N·T): con trỏ hai đầu giữ tiền tố cực đại hàng trước; tách
    lam·pos[t] khỏi −lam·pos[t'] để giữ O(N·T) kể cả khi có phạt mềm.
    """
    sim = np.asarray(sim, dtype=np.float64)
    if sim.ndim != 2:
        raise ValueError("sim phải là ma trận 2 chiều (N, T)")
    n, t = sim.shape
    if pos is None:
        pos = np.arange(t, dtype=np.float64)
    else:
        pos = np.asarray(pos, dtype=np.float64)
        if len(pos) != t:
            raise ValueError("pos phải cùng số cột với sim")
        if np.any(np.diff(pos) < 0):
            raise ValueError("pos phải TĂNG DẦN (ứng viên đã sắp theo thời gian)")
    if n == 0:
        return [], 0.0
    if t < n:
        raise ValueError(f"cần >= {n} ứng viên cho {n} khoảnh khắc, chỉ có {t}")

    dp = np.full((n, t), _NEG, dtype=np.float64)
    back = np.full((n, t), -1, dtype=np.int64)
    dp[0] = sim[0]

    for i in range(1, n):
        # aug[j] = dp[i-1, j] + lam*pos[j]  -> lấy tiền tố cực đại của aug, rồi
        # dp[i, t] = sim[i, t] - lam*pos[t] + max_{j hợp lệ} aug[j]
        aug = dp[i - 1] + lam * pos
        best_val = _NEG
        best_idx = -1
        j = 0                                   # con trỏ: cột hợp lệ nhỏ nhất chưa nạp
        for c in range(t):
            cutoff = pos[c] - min_gap           # t' hợp lệ <=> pos[t'] <= cutoff
            while j < c and pos[j] <= cutoff:
                if aug[j] > best_val:
                    best_val = aug[j]
                    best_idx = j
                j += 1
            if best_idx >= 0:
                dp[i, c] = sim[i, c] - lam * pos[c] + best_val
                back[i, c] = best_idx

    end = int(np.argmax(dp[n - 1]))
    if dp[n - 1, end] <= _NEG / 2:
        raise ValueError("không có căn chỉnh hợp lệ (min_gap quá lớn so với số ứng viên)")
    idx = [0] * n
    idx[n - 1] = end
    for i in range(n - 1, 0, -1):
        idx[i - 1] = int(back[i, idx[i]])
    return idx, float(dp[n - 1, end])


def _l2norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-8, None)


def cosine_sim_matrix(query_feats: np.ndarray, cand_feats: np.ndarray) -> np.ndarray:
    """(N, D) × (T, D) -> (N, T) cosine. Tự chuẩn hoá L2 nên nhận vector thô cũng đúng."""
    q = _l2norm(query_feats)
    c = _l2norm(cand_feats)
    return q @ c.T


def _rank_normalize(sim: np.ndarray) -> np.ndarray:
    """Chuẩn hoá THEO HÀNG về [0,1] bằng thứ hạng (bền với thang đo khác nhau giữa
    mô thức — cosine CLIP và cosine e5 KHÔNG cùng dải). Dùng cho RRF/tổng có trọng số."""
    out = np.zeros_like(sim, dtype=np.float64)
    for i in range(sim.shape[0]):
        order = np.argsort(np.argsort(sim[i]))          # hạng 0..T-1
        denom = max(1, sim.shape[1] - 1)
        out[i] = order / denom
    return out


def fuse_scores(sims: Dict[str, np.ndarray],
                weights: Optional[Dict[str, float]] = None,
                method: str = "weighted") -> np.ndarray:
    """Trộn NHIỀU ma trận tương đồng (mỗi mô thức 1 cái, cùng shape (N, T)) thành 1.

    method="weighted": Σ w_m · rank_norm(sim_m) — chuẩn hoá hạng theo hàng rồi cộng trọng số.
    method="rrf"     : Reciprocal Rank Fusion (Cormack 2009): điểm = w/(k + rank), bền với
                       outlier, không cần cùng thang. weights None -> đều nhau; None thì bỏ qua.
    """
    keys = [k for k, v in sims.items() if v is not None]
    if not keys:
        raise ValueError("không có mô thức nào để trộn")
    if weights is None:
        weights = {k: 1.0 for k in keys}
    shape = sims[keys[0]].shape

    if method == "rrf":
        k0 = 60.0                                        # hằng RRF chuẩn (Cormack 2009)
        out = np.zeros(shape, dtype=np.float64)
        for k in keys:
            s = np.asarray(sims[k], dtype=np.float64)
            for i in range(shape[0]):
                rank = np.argsort(np.argsort(-s[i]))     # hạng 0 = điểm cao nhất
                out[i] += weights.get(k, 1.0) / (k0 + rank)
        return out

    out = np.zeros(shape, dtype=np.float64)
    wsum = 0.0
    for k in keys:
        w = weights.get(k, 1.0)
        out += w * _rank_normalize(np.asarray(sims[k]))
        wsum += w
    return out / max(wsum, 1e-9)


def snap_to_apex(center_frame: int, flow_frames: Sequence[int],
                 flow_mag: Sequence[float], radius: int = 5) -> int:
    """Trong bán kính `radius` quanh `center_frame`, chọn frame có |optical-flow| LỚN NHẤT.

    Khoảnh khắc TRAKE (chạm/rời đất, đảo chiều, va chạm) là ĐỈNH độ lớn chuyển động
    (apex spotting, arXiv:2012.11307); CLIP thường phẳng qua vài frame nên dùng đỉnh flow
    phá thế hoà. flow_frames/flow_mag tăng dần theo frame; không có mẫu -> trả center_frame.
    """
    flow_frames = np.asarray(flow_frames)
    flow_mag = np.asarray(flow_mag, dtype=np.float64)
    if len(flow_frames) == 0:
        return int(center_frame)
    m = np.abs(flow_frames - center_frame) <= radius
    if not m.any():
        return int(center_frame)
    sub_f = flow_frames[m]
    sub_v = flow_mag[m]
    return int(sub_f[int(np.argmax(sub_v))])


def build_submission(phase_frames: Sequence[int],
                     budget: int = 100,
                     stride: int = 8,
                     half_span: int = 12,
                     bounds: Optional[Sequence[Tuple[int, int]]] = None) -> List[List[int]]:
    """Với mỗi khoảnh khắc, rải nhiều frame_id quanh frame đã định vị để PHỦ sai số, sao
    cho TÍCH số phương án <= `budget` (mặc định 100 — trần đáp án/truy vấn TRAKE, PDF mục 2).
    Trả về list[list[int]] — frame ứng viên cho từng khoảnh khắc.

    Định vị sai số ~±10-15 frame > cửa sổ chấm <10 frame => nộp 1 frame dễ trượt. Mỗi
    khoảnh khắc rải k = floor(budget^(1/N)) phương án, bước `stride` trong ±`half_span`.
    bounds[i] = (lo, hi) giới hạn frame hợp lệ; rải KHÔNG vượt ra ngoài. None -> không giới hạn.
    """
    n = len(phase_frames)
    if n == 0:
        return []
    k = max(1, int(budget ** (1.0 / n)))                 # số phương án MỖI khoảnh khắc
    while k ** n > budget and k > 1:                      # chặn cho chắc tích <= budget
        k -= 1

    out: List[List[int]] = []
    for i, f in enumerate(phase_frames):
        f = int(f)
        # các offset đối xứng quanh f, bước `stride`, tối đa k phương án, trong ±half_span
        offs = [0]
        d = stride
        while len(offs) < k and d <= half_span:
            offs.append(+d)
            if len(offs) < k:
                offs.append(-d)
            d += stride
        cands = sorted({f + o for o in offs[:k]})
        if bounds is not None and i < len(bounds) and bounds[i] is not None:
            lo, hi = bounds[i]
            cands = [c for c in cands if lo <= c <= hi] or [int(min(max(f, lo), hi))]
        out.append(cands)
    return out


def localize_phases(phase_texts: Sequence[str],
                    candidates: List[dict],
                    clip_text_fn=None,
                    e5_query_fn=None,
                    weights: Optional[Dict[str, float]] = None,
                    min_gap_frames: int = 8,
                    lam: float = 0.0,
                    fuse: str = "weighted") -> Dict:
    """Định vị N khoảnh khắc TRAKE trong MỘT scene ứng viên.

    phase_texts : N mô tả khoảnh khắc, đúng thứ tự thời gian.
    candidates  : ứng viên đã sắp theo frame tăng dần, mỗi phần tử:
                    { "frame": int, "clip": np.ndarray|None, "siglip": np.ndarray|None,
                      "text": str|None }  # nhãn action/motion/caption
    clip_text_fn: text->(M,D) sang không gian CLIP (so với ảnh CLIP). None -> bỏ kênh CLIP.
    e5_query_fn : text->(D,) sang e5 (so chữ-chữ với candidate["text"]). None -> bỏ kênh chữ.
    min_gap_frames: khoảng cách frame tối thiểu giữa 2 khoảnh khắc (mặc định 8 ~ cỡ cửa sổ chấm).
    lam         : phạt mềm độ trải (DANTE), 0 = tắt.

    Trả về { "frames": [...], "cand_index": [...], "score": float, "sim": (N,T) }.
    """
    if not candidates:
        raise ValueError("không có ứng viên nào trong scene")
    frames = np.asarray([c["frame"] for c in candidates])
    order = np.argsort(frames)
    cands = [candidates[k] for k in order]
    frames = frames[order]

    sims: Dict[str, Optional[np.ndarray]] = {}

    # kênh 1 — CLIP ảnh-chữ
    if clip_text_fn is not None and all(c.get("clip") is not None for c in cands):
        qt = np.asarray(clip_text_fn(list(phase_texts)), dtype=np.float32)
        ci = np.stack([np.asarray(c["clip"], dtype=np.float32) for c in cands])
        sims["clip"] = cosine_sim_matrix(qt, ci)

    # kênh 2 — e5 chữ-chữ (mô tả khoảnh khắc vs nhãn action/motion của ứng viên)
    if e5_query_fn is not None and any((c.get("text") or "").strip() for c in cands):
        from ..embedding.embed import CaptionEmbedder  # noqa: F401  (chỉ để rõ nguồn)
        qv = np.stack([np.asarray(e5_query_fn(t), dtype=np.float32) for t in phase_texts])
        # candidate["text"] cần mã hoá sẵn ('passage:') — caller truyền qua "text_emb"
        if all(c.get("text_emb") is not None for c in cands):
            cv = np.stack([np.asarray(c["text_emb"], dtype=np.float32) for c in cands])
            sims["e5"] = cosine_sim_matrix(qv, cv)

    if not sims:
        raise ValueError("không mô thức nào khả dụng (thiếu encoder hoặc thiếu embedding ứng viên)")

    sim = fuse_scores(sims, weights=weights, method=fuse)
    idx, score = align_sequence(sim, pos=frames, min_gap=min_gap_frames, lam=lam)
    return {
        "frames": [int(frames[i]) for i in idx],
        "cand_index": [int(i) for i in idx],
        "score": score,
        "sim": sim,
        "modalities": list(sims.keys()),
    }
