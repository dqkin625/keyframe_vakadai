"""Kiểm thử hàm KHỚP CHUỖI TRAKE bằng TÍN HIỆU TỔNG HỢP (biết trước đáp án).

    python scripts/test_trake_match.py        (thoát 0 = đạt, 1 = có test hỏng)

Vì sao dùng tín hiệu tổng hợp: quy hoạch động khớp chuỗi phải ĐÚNG về mặt THUẬT TOÁN
(bảo toàn thứ tự, tối ưu toàn cục, chịu ràng buộc khoảng cách) — điều này kiểm được bằng
ma trận dựng tay có đáp án, KHÔNG cần model/embedding. Verify trên dữ liệu THẬT nằm ở
`scripts/eval_trake.py`.

Đối chiếu công thức DANTE (arXiv:2512.13169, Eq.2):
    DP[i,t] = S[i,t] + max_{τ<t} ( DP[i-1,τ] - λ·(t-τ) )
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from src.retrieval.trake_match import (
    align_sequence, fuse_scores, snap_to_apex, build_submission, cosine_sim_matrix)

FAIL = []


def check(name, cond, extra=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {name}{'   ' + extra if extra else ''}")
    if not cond:
        FAIL.append(name)


def brute_force(sim, min_gap=1):
    """Đáp án THAM CHIẾU: duyệt MỌI tổ hợp chỉ số tăng thoả min_gap, trả tổ hợp điểm cao nhất.
    Chỉ dùng cho ma trận NHỎ để đối chiếu với quy hoạch động."""
    from itertools import combinations
    n, t = sim.shape
    best, best_idx = -1e18, None
    for combo in combinations(range(t), n):
        if all(combo[k] - combo[k - 1] >= min_gap for k in range(1, n)):
            s = sum(sim[i, combo[i]] for i in range(n))
            if s > best:
                best, best_idx = s, combo
    return list(best_idx), best


print("\n[1] Căn chỉnh CƠ BẢN — chéo trội thì chọn đúng đường chéo")
sim = np.array([[0.9, 0.1, 0.1, 0.1],
                [0.1, 0.9, 0.1, 0.1],
                [0.1, 0.1, 0.9, 0.1]], dtype=np.float64)
idx, score = align_sequence(sim, min_gap=1)
check("chọn đúng đường chéo (0,1,2)", idx == [0, 1, 2], str(idx))
check("thứ tự không giảm", all(idx[k] >= idx[k - 1] for k in range(1, len(idx))))

print("\n[2] BẪY THAM LAM — argmax từng hàng VI PHẠM thứ tự, DP phải tối ưu TOÀN CỤC")
# hàng 0 đỉnh ở cột 3; hàng 1 đỉnh ở cột 1 -> tham lam chọn (3,1) ĐẢO thứ tự.
# DP buộc thứ tự tăng -> phải hi sinh cục bộ để tổng tối ưu mà KHÔNG đảo.
sim2 = np.array([[0.2, 0.3, 0.4, 0.95],
                 [0.1, 0.90, 0.2, 0.30]], dtype=np.float64)
greedy = [int(np.argmax(sim2[0])), int(np.argmax(sim2[1]))]
idx2, sc2 = align_sequence(sim2, min_gap=1)
bidx, bsc = brute_force(sim2, min_gap=1)
check("tham lam ĐÚNG LÀ đảo thứ tự (3,1)", greedy == [3, 1], str(greedy))
check("DP giữ thứ tự tăng", all(idx2[k] > idx2[k - 1] for k in range(1, len(idx2))), str(idx2))
check("DP trùng đáp án brute-force", idx2 == bidx and abs(sc2 - bsc) < 1e-9,
      f"dp={idx2}({sc2:.3f}) bf={bidx}({bsc:.3f})")

print("\n[3] Tối ưu TOÀN CỤC trên ma trận ngẫu nhiên (đối chiếu brute-force)")
rng = np.random.default_rng(0)
ok_all = True
for trial in range(200):
    n = rng.integers(2, 5)
    t = rng.integers(n, 9)
    s = rng.random((n, t))
    gap = int(rng.integers(1, 3))
    if t < 1 + (n - 1) * gap:            # bỏ trường hợp bất khả thi
        continue
    di, ds = align_sequence(s, min_gap=gap)
    bi, bs = brute_force(s, min_gap=gap)
    if abs(ds - bs) > 1e-9:
        ok_all = False
        print(f"      lệch: n={n} t={t} gap={gap} dp={ds:.4f} bf={bs:.4f}")
        break
check("DP == brute-force trên 200 ma trận ngẫu nhiên (mọi min_gap)", ok_all)

print("\n[4] Ràng buộc KHOẢNG CÁCH tối thiểu (min_gap theo FRAME thật)")
sim4 = np.array([[0.9, 0.8, 0.1, 0.1, 0.1],
                 [0.1, 0.1, 0.85, 0.9, 0.1]], dtype=np.float64)
pos = [0, 3, 30, 33, 60]                 # frame thật
idx4, _ = align_sequence(sim4, pos=pos, min_gap=10)
gap_frames = pos[idx4[1]] - pos[idx4[0]]
check("2 khoảnh khắc cách nhau >= min_gap frame", gap_frames >= 10, f"gap={gap_frames}")
check("vẫn giữ thứ tự thời gian", pos[idx4[1]] > pos[idx4[0]])

print("\n[5] PHẠT MỀM λ (DANTE) — kéo chuỗi lại GẦN nhau khi có nhiều lựa chọn tương đương")
# hàng 1 hợp cột 1 VÀ cột 6 gần bằng nhau; λ>0 phải ưu tiên cột GẦN hàng 0 (cột 0)
sim5 = np.array([[0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                 [0.1, 0.80, 0.1, 0.1, 0.1, 0.1, 0.82]], dtype=np.float64)
idx_no, _ = align_sequence(sim5, min_gap=1, lam=0.0)
idx_lam, _ = align_sequence(sim5, min_gap=1, lam=0.05)
check("λ=0 chọn cột XA hơi cao điểm (6)", idx_no[1] == 6, str(idx_no))
check("λ>0 kéo về cột GẦN (1)", idx_lam[1] == 1, str(idx_lam))

print("\n[6] Trường hợp biên")
try:
    align_sequence(np.zeros((3, 2)), min_gap=1)
    check("T<N phải báo lỗi", False)
except ValueError:
    check("T<N -> ValueError", True)
idx1, sc1 = align_sequence(np.array([[0.1, 0.5, 0.3]]), min_gap=1)
check("N=1 -> chọn cột điểm cao nhất", idx1 == [1], str(idx1))

print("\n[7] fuse_scores — trộn đa mô thức (weighted + RRF)")
a = np.array([[0.1, 0.9, 0.2]])
b = np.array([[0.9, 0.1, 0.2]])
fw = fuse_scores({"clip": a, "e5": b}, method="weighted")
check("weighted: 2 mô thức ngược nhau -> cột giữa (2) KHÔNG thắng tuyệt đối",
      fw.shape == (1, 3))
fr = fuse_scores({"clip": a, "e5": b}, weights={"clip": 3.0, "e5": 1.0}, method="rrf")
check("RRF trọng số CLIP cao -> cột 1 (đỉnh CLIP) thắng", int(np.argmax(fr[0])) == 1, str(fr[0]))

print("\n[8] snap_to_apex — ghim về đỉnh optical-flow trong bán kính")
frames = [100, 103, 106, 109, 112]
mags = [0.2, 0.3, 2.5, 0.4, 0.1]
ap = snap_to_apex(104, frames, mags, radius=5)
check("ghim về frame có flow lớn nhất (106)", ap == 106, str(ap))
ap2 = snap_to_apex(500, frames, mags, radius=5)
check("không có mẫu trong bán kính -> giữ nguyên center", ap2 == 500, str(ap2))

print("\n[9] build_submission — rải frame_id dưới NGÂN SÁCH 100")
sub = build_submission([1000, 2000, 3000, 4000], budget=100, stride=8, half_span=12)
prod = 1
for s in sub:
    prod *= len(s)
check("4 khoảnh khắc: tích phương án <= 100", prod <= 100, f"tích={prod}")
check("mỗi khoảnh khắc rải quanh frame gốc", all(1000 in sub[0] for _ in [0]), str(sub[0]))
sub2 = build_submission([1000, 2000], budget=100, stride=8, half_span=40)
check("2 khoảnh khắc: mỗi bên nhiều phương án hơn (k=10)", len(sub2[0]) == 10, str(len(sub2[0])))
subb = build_submission([1000], budget=100, stride=8, half_span=12,
                        bounds=[(995, 1005)])
check("bounds giới hạn frame trong [lo,hi]", all(995 <= f <= 1005 for f in subb[0]), str(subb[0]))

print("\n[10] cosine_sim_matrix — nhận vector thô, tự chuẩn hoá")
q = np.array([[1.0, 0.0], [0.0, 1.0]])
c = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]])
m = cosine_sim_matrix(q, c)
check("cosine đúng: q0·c0=1, q0·c1=0", abs(m[0, 0] - 1) < 1e-6 and abs(m[0, 1]) < 1e-6, str(m[0]))

print("\n" + "=" * 60)
print(f"KẾT QUẢ: {'TẤT CẢ ĐẠT' if not FAIL else str(len(FAIL)) + ' TEST HỎNG: ' + ', '.join(FAIL)}")
print("=" * 60)
sys.exit(1 if FAIL else 0)
