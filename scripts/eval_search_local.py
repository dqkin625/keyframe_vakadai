"""Cham he thong search cua AIHCM bang bo ground truth KIS tu tao.

Goi THANG tiers.dense_search.search_dense() trong cung tien trinh - khong qua HTTP, nen
model + ma tran embedding chi nap 1 lan roi dung lai cho ca 100 query.

    python scripts/eval_search_local.py --mode pe_core
    python scripts/eval_search_local.py --mode beit3 --limit 10 --per-query

Cong thuc KIS theo the le vong so tuyen AIC 2026 (muc 2.1.1 va 2.2) - xem eval_search_api.py.
Diem moi query chi phu thuoc HANG cua dap an dung DAU TIEN vi R@k lay max chu khong cong don.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from eval_search_api import KS, rank_bucket, score  # noqa: E402

V3 = os.path.join(_ROOT, "AIHCM", "v3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pe_core", help="pe_core | beit3 (siglip/rrf thieu index)")
    ap.add_argument("--gt", default=os.path.join(_ROOT, "data", "eval", "L28_GT_update.json"))
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="chi chay N query dau (0 = tat ca)")
    ap.add_argument("--per-query", action="store_true")
    ap.add_argument("--out", default=os.path.join(_ROOT, "data", "eval", "search_result_local.json"))
    args = ap.parse_args()

    gt = json.load(open(args.gt, encoding="utf-8"))
    if args.limit:
        gt = gt[: args.limit]

    sys.path.insert(0, os.path.join(V3, "share"))
    sys.path.insert(0, os.path.join(V3, "online"))
    os.chdir(V3)
    from tiers.dense_search import search_dense

    print(f"{len(gt)} query | mode={args.mode} | top_k={args.top_k}", flush=True)

    rows, fails = [], []
    t0 = time.time()
    for i, q in enumerate(gt, 1):
        t1 = time.time()
        try:
            df = search_dense(q["Queries"], mode=args.mode, top_k=args.top_k)
            hits = [{"video_id": r.video_id, "frame_idx": int(r.frame_id)}
                    for r in df.itertuples()]
        except Exception as e:
            fails.append((q["query_id"], str(e)[:90]))
            hits = []
        f, rank, n = score(hits, q)
        dt = time.time() - t1
        rows.append({"query_id": q["query_id"], "final": f, "rank": rank,
                     "n_hits": n, "video_id": q["video_id"], "sec": round(dt, 1)})
        if args.per_query:
            print(f"  {q['query_id']}  {f:.2f}  {rank_bucket(rank):<12} {dt:5.1f}s  "
                  f"{q['Queries'][:52]}", flush=True)
        elif i % 5 == 0:
            print(f"  {i}/{len(gt)}  ({time.time()-t0:.0f}s troi qua)", flush=True)

    n = len(rows)
    avg = sum(r["final"] for r in rows) / n if n else 0.0
    b = collections.Counter(rank_bucket(r["rank"]) for r in rows)
    secs = [r["sec"] for r in rows]

    print("\n" + "=" * 62, flush=True)
    print(f"KET QUA - {n} query, mode={args.mode}, tong {time.time()-t0:.0f}s", flush=True)
    print("=" * 62, flush=True)
    print(f"  FINAL SCORE trung binh : {avg:.4f}", flush=True)
    print(f"  tim thay trong top-100 : {sum(1 for r in rows if r['rank']):>3}/{n}", flush=True)
    print(f"  dung ngay hang 1       : {b['hạng 1']:>3}/{n}", flush=True)
    if len(secs) > 1:
        print(f"  thoi gian/query        : dau {secs[0]:.0f}s, sau do trung binh "
              f"{sum(secs[1:])/len(secs[1:]):.1f}s", flush=True)
    print("\n  phan bo hang dap an dung dau tien:", flush=True)
    for k in ("hạng 1", "hạng 2-5", "hạng 6-20", "hạng 21-50", "hạng 51-100", "trượt"):
        if b[k]:
            print(f"     {k:<14} {b[k]:>3}  ({100*b[k]/n:.0f}%)", flush=True)
    if fails:
        print(f"\n  ! {len(fails)} query LOI:", flush=True)
        for qid, e in fails[:5]:
            print(f"     {qid}: {e}", flush=True)

    json.dump({"mode": args.mode, "n": n, "final_score": avg, "buckets": dict(b),
               "rows": rows, "fails": fails},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n  chi tiet -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
