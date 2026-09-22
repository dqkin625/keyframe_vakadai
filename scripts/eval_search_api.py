"""
Chấm hệ thống tìm kiếm bằng bộ ground truth KIS tự tạo.

    python -u scripts/eval_search_api.py --api https://xxx.ngrok-free.app/search
    python -u scripts/eval_search_api.py --api http://localhost:8600/search --per-query

Công thức KIS theo thể lệ vòng sơ tuyển AIC 2026 (mục 2.1.1 và 2.2):
    R-Score = 1 nếu video đúng VÀ frame_idx nằm trong range_frame_id, ngược lại 0
    R@k     = max{R-Score(r_1) ... R-Score(r_k)},  k in {1, 5, 20, 50, 100}
    Final   = trung bình 5 giá trị R@k

Vì R@k lấy MAX chứ không cộng dồn, điểm chỉ phụ thuộc HẠNG của đáp án đúng ĐẦU TIÊN:
    hạng 1 -> 1.00 | 2-5 -> 0.80 | 6-20 -> 0.60 | 21-50 -> 0.40 | 51-100 -> 0.20 | trượt -> 0

API cần trả JSON: {"results": [{"video_id": "L28_V001", "frame_idx": 9835}, ...]}
xếp theo thứ hạng giảm dần. Chấp nhận cả `frame_id`/`frame` thay cho `frame_idx`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

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


def rank_bucket(rank):
    if rank is None:
        return "trượt"
    for lo, hi, name in ((1, 1, "hạng 1"), (2, 5, "hạng 2-5"), (6, 20, "hạng 6-20"),
                         (21, 50, "hạng 21-50"), (51, 100, "hạng 51-100")):
        if lo <= rank <= hi:
            return name
    return "ngoài 100"


def score(hits, gt):
    """hits: list dict theo thứ hạng. gt: 1 record trong L28_GT_update.json."""
    lo, hi = gt["range_frame_id"]
    ok = []
    for h in hits[:MAX_ANSWERS]:
        v = h.get("video_id")
        f = h.get("frame_idx", h.get("frame_id", h.get("frame")))
        try:
            f = int(f)
        except (TypeError, ValueError):
            ok.append(0); continue
        ok.append(int(v == gt["video_id"] and lo <= f <= hi))
    rank = next((i + 1 for i, s in enumerate(ok) if s), None)
    final = sum(max(ok[:k], default=0) for k in KS) / len(KS)
    return final, rank, len(hits)


def call_api(url, query, top_k, timeout):
    import urllib.request
    body = json.dumps({"query": query, "top_k": top_k}).encode()
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true",
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    if isinstance(d, list):
        return d
    for k in ("results", "hits", "data", "items"):
        if isinstance(d.get(k), list):
            return d[k]
    raise ValueError(f"Không tìm thấy danh sách kết quả trong phản hồi: {list(d)[:5]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True, help="URL endpoint POST /search")
    ap.add_argument("--gt", default="data/eval/L28_GT_update.json")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--per-query", action="store_true")
    ap.add_argument("--out", default="data/eval/search_result.json")
    args = ap.parse_args()

    gt = json.load(open(args.gt, encoding="utf-8"))
    print(f"{len(gt)} query | API: {args.api}", flush=True)

    rows, fails = [], []
    t0 = time.time()
    for i, q in enumerate(gt, 1):
        try:
            hits = call_api(args.api, q["Queries"], args.top_k, args.timeout)
        except Exception as e:
            fails.append((q["query_id"], str(e)[:90]))
            hits = []
        f, rank, n = score(hits, q)
        rows.append({"query_id": q["query_id"], "final": f, "rank": rank,
                     "n_hits": n, "video_id": q["video_id"]})
        if args.per_query:
            print(f"  {q['query_id']}  {f:.2f}  {rank_bucket(rank):<12} "
                  f"({n} kq)  {q['Queries'][:55]}", flush=True)
        elif i % 10 == 0:
            print(f"  {i}/{len(gt)} ...", flush=True)

    n = len(rows)
    avg = sum(r["final"] for r in rows) / n if n else 0.0
    import collections
    b = collections.Counter(rank_bucket(r["rank"]) for r in rows)

    print("\n" + "=" * 62, flush=True)
    print(f"KẾT QUẢ — {n} query, chạy {time.time()-t0:.0f}s", flush=True)
    print("=" * 62, flush=True)
    print(f"  FINAL SCORE trung bình : {avg:.4f}", flush=True)
    print(f"  tìm thấy trong top-100 : {sum(1 for r in rows if r['rank']):>3}/{n}", flush=True)
    print(f"  đúng ngay hạng 1       : {b['hạng 1']:>3}/{n}", flush=True)
    print("\n  phân bố hạng đáp án đúng đầu tiên:", flush=True)
    for k in ("hạng 1", "hạng 2-5", "hạng 6-20", "hạng 21-50", "hạng 51-100", "trượt"):
        if b[k]:
            print(f"     {k:<14} {b[k]:>3}  ({100*b[k]/n:.0f}%)", flush=True)
    if fails:
        print(f"\n  ⚠ {len(fails)} query LỖI GỌI API:", flush=True)
        for qid, e in fails[:5]:
            print(f"     {qid}: {e}", flush=True)

    json.dump({"api": args.api, "n": n, "final_score": avg,
               "buckets": dict(b), "rows": rows, "fails": fails},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n  chi tiết -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
