"""
Bộ chấm KIS — hiện thực ĐÚNG công thức trong thể lệ vòng sơ tuyển AIC 2026 (trang 3 & 4-5).

    python -u scripts/eval_kis.py --sub data/eval/submission.json > logs/eval_kis.log 2>&1
    python -u scripts/eval_kis.py --sub run.json --per-query      # in điểm từng truy vấn

CÔNG THỨC (thể lệ, mục 2.1.1 và 2.2):
    R-Score(r_i) = 1 nếu (video đúng VÀ frame_id ∈ [s,e]), ngược lại 0     ← NHỊ PHÂN
    R@k          = max{ R-Score(r_1) ... R-Score(r_k) },  k ∈ {1,5,20,50,100}
    Final Score  = trung bình 5 giá trị R@k

Vì R@k lấy MAX chứ không cộng dồn, điểm chỉ phụ thuộc HẠNG của câu đúng ĐẦU TIÊN:
    hạng 1 -> 1.00 | 2-5 -> 0.80 | 6-20 -> 0.60 | 21-50 -> 0.40 | 51-100 -> 0.20 | trượt -> 0
Hệ quả: nộp trùng nhiều frame trong CÙNG một shot là vứt đi. Đa dạng > lặp lại.

BỀ RỘNG [s,e]: BTC không công bố con số cố định (ví dụ KIS 11 frame, ví dụ Q&A 101 frame).
Nên chấm ở NHIỀU mức song song. Đọc cột nào cũng được, nhưng đừng chỉ đọc một cột:
    tol5    ~ ±5 frame   (sát ví dụ KIS trong thể lệ — khắt khe nhất)
    tol30   ~ ±1 giây
    tol90   ~ ±3 giây
    kf_span ~ đoạn "thuộc về" keyframe đó (nửa đường tới keyframe trước/sau)

Định dạng file nộp (JSON), khớp thứ tự xếp hạng của hệ:
    { "kis_001": [["L21_V001", 1500], ["L21_V003", 220], ...],   # tối đa 100, hạng 1 trước
      "kis_002": [...] }
"""
import argparse
import json
import os
import sys
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

KS = (1, 5, 20, 50, 100)
MAX_ANSWERS = 100


def r_score(ans, gt_video: str, rng) -> int:
    """R-Score một câu trả lời KIS: 1 nếu khớp video VÀ frame nằm trong [s,e]."""
    try:
        vid, frame = ans[0], int(ans[1])
    except (TypeError, ValueError, IndexError):
        return 0
    return int(vid == gt_video and rng[0] <= frame <= rng[1])


def final_score(answers, gt_video: str, rng):
    """Trả về (Final Score, hạng câu đúng đầu tiên hoặc None)."""
    scores = [r_score(a, gt_video, rng) for a in answers[:MAX_ANSWERS]]
    rank = next((i + 1 for i, s in enumerate(scores) if s == 1), None)
    r_at_k = [max(scores[:k], default=0) for k in KS]
    return sum(r_at_k) / len(KS), rank


def rank_bucket(rank):
    if rank is None:
        return "trượt"
    for lo, hi, name in ((1, 1, "hạng 1"), (2, 5, "hạng 2-5"), (6, 20, "hạng 6-20"),
                         (21, 50, "hạng 21-50"), (51, 100, "hạng 51-100")):
        if lo <= rank <= hi:
            return name
    return "ngoài 100"


def check_submission(answers, qid: str, warns: list):
    """Cảnh báo các lỗi làm mất điểm oan mà bộ chấm thật sẽ không tha."""
    if len(answers) > MAX_ANSWERS:
        warns.append(f"{qid}: nộp {len(answers)} câu, thể lệ cho tối đa {MAX_ANSWERS} — phần dư bị bỏ")
    if len(answers) < MAX_ANSWERS:
        warns.append(f"{qid}: chỉ nộp {len(answers)}/{MAX_ANSWERS} câu — bậc 51-100 vẫn được 0.2, "
                     f"bỏ trống là tự mất điểm")
    seen, dup = set(), 0
    for a in answers[:MAX_ANSWERS]:
        try:
            key = (a[0], int(a[1]))
        except (TypeError, ValueError, IndexError):
            continue
        if key in seen:
            dup += 1
        seen.add(key)
    if dup:
        warns.append(f"{qid}: {dup} câu trùng hệt (video,frame) — vô giá trị vì R@k lấy MAX")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/eval/kis_gt.json")
    ap.add_argument("--sub", required=True, help="file JSON kết quả của hệ thống")
    ap.add_argument("--per-query", action="store_true")
    ap.add_argument("--include-review", action="store_true",
                    help="tính cả mẫu needs_review (mặc định loại, vì truy vấn chung chung)")
    args = ap.parse_args()

    gt = json.load(open(args.gt, encoding="utf-8"))
    sub = json.load(open(args.sub, encoding="utf-8"))

    samples = [s for s in gt["samples"] if s.get("query")]
    dropped = 0
    if not args.include_review:
        before = len(samples)
        samples = [s for s in samples if not s.get("needs_review")]
        dropped = before - len(samples)

    tol_names = list(samples[0]["gt_ranges"].keys()) if samples else []
    totals = {t: [] for t in tol_names}
    buckets = {t: defaultdict(int) for t in tol_names}
    by_source = defaultdict(lambda: defaultdict(list))
    warns, missing = [], 0

    for s in samples:
        answers = sub.get(s["query_id"])
        if answers is None:
            missing += 1
            answers = []
        else:
            check_submission(answers, s["query_id"], warns)
        for t in tol_names:
            fs, rank = final_score(answers, s["gt_video"], s["gt_ranges"][t])
            totals[t].append(fs)
            buckets[t][rank_bucket(rank)] += 1
            by_source[s.get("source", "?")][t].append(fs)
        if args.per_query:
            fs30, rk30 = final_score(answers, s["gt_video"], s["gt_ranges"]["tol30"])
            print(f"  {s['query_id']}  tol30={fs30:.2f} ({rank_bucket(rk30)})  "
                  f"{s['gt_video']}@{s['gt_frame']}  {s['query'][:60]}", flush=True)

    n = len(samples)
    print(f"\n{'='*72}", flush=True)
    print(f"CHẤM KIS — {n} truy vấn"
          + (f" (loại {dropped} mẫu needs_review)" if dropped else ""), flush=True)
    if missing:
        print(f"⚠ {missing} truy vấn KHÔNG có trong file nộp -> tính 0 điểm", flush=True)
    print(f"{'='*72}", flush=True)

    print(f"\n{'mức [s,e]':<10} {'Final Score':>12}   phân bố hạng câu đúng đầu tiên", flush=True)
    for t in tol_names:
        avg = sum(totals[t]) / n if n else 0.0
        b = buckets[t]
        dist = "  ".join(f"{k}:{b[k]}" for k in
                         ("hạng 1", "hạng 2-5", "hạng 6-20", "hạng 21-50", "hạng 51-100", "trượt")
                         if b[k])
        print(f"{t:<10} {avg:>12.4f}   {dist}", flush=True)

    if len(by_source) > 1:
        print(f"\nTÁCH THEO NGUỒN TRUY VẤN (llm sinh thường DỄ hơn người viết tay):", flush=True)
        for src, d in sorted(by_source.items()):
            row = "  ".join(f"{t}={sum(v)/len(v):.4f}" for t, v in d.items())
            print(f"  {src:<8} (n={len(d[tol_names[0]])})  {row}", flush=True)

    if warns:
        print(f"\n⚠ {len(warns)} CẢNH BÁO VỀ FILE NỘP:", flush=True)
        for w in warns[:15]:
            print(f"  - {w}", flush=True)
        if len(warns) > 15:
            print(f"  ... còn {len(warns)-15} cảnh báo nữa", flush=True)

    print("\nNhắc: Final Score chỉ phụ thuộc HẠNG câu đúng đầu tiên "
          "(1 -> 1.00 | 2-5 -> 0.80 | 6-20 -> 0.60 | 21-50 -> 0.40 | 51-100 -> 0.20).", flush=True)
    print("Đẩy câu đúng qua các mốc 1/5/20/50 mới được điểm; xê dịch trong cùng bậc = 0.\n", flush=True)


if __name__ == "__main__":
    main()
