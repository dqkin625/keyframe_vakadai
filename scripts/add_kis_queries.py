"""
Nạp truy vấn KIS viết sẵn vào data/eval/kis_queries.jsonl (ghi đè bản ghi trùng query_id).

    python -u scripts/add_kis_queries.py batch.json
    python -u scripts/add_kis_queries.py batch.json --merge     # gộp luôn vào kis_gt.json

`batch.json` là danh sách:
    [{"query_id":"kis_027","query":"Tìm ...","distinct":4,"note":"...","source":"vlm"}, ...]

Dùng cho:
  - truy vấn do VLM đọc ảnh trực tiếp viết ra (source="vlm");
  - truy vấn NGƯỜI viết tay (source="manual") — nhóm đối chứng bắt buộc, xem docs/METHODS.md §K5.
"""
import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

JSONL_PATH = "data/eval/kis_queries.jsonl"
GT_PATH = "data/eval/kis_gt.json"
MIN_DISTINCT = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("--jsonl", default=JSONL_PATH)
    ap.add_argument("--gt", default=GT_PATH)
    ap.add_argument("--merge", action="store_true", help="gộp luôn vào kis_gt.json sau khi nạp")
    args = ap.parse_args()

    with open(args.batch, encoding="utf-8") as f:
        batch = json.load(f)

    rows = {}
    if os.path.exists(args.jsonl):
        with open(args.jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    rows[d["query_id"]] = d

    gt_ids = {s["query_id"] for s in json.load(open(args.gt, encoding="utf-8"))["samples"]}

    n_new = n_upd = 0
    for r in batch:
        qid = r["query_id"]
        if qid not in gt_ids:
            print(f"  ⚠ bỏ qua {qid}: không có trong {args.gt}", flush=True)
            continue
        rec = {"query_id": qid, "query": r["query"].strip(),
               "distinct": int(r.get("distinct", 0)), "note": r.get("note", "").strip(),
               "source": r.get("source", "manual")}
        if qid in rows:
            n_upd += 1
        else:
            n_new += 1
        rows[qid] = rec

    with open(args.jsonl, "w", encoding="utf-8") as f:
        for qid in sorted(rows):
            f.write(json.dumps(rows[qid], ensure_ascii=False) + "\n")
    print(f"nạp {len(batch)} bản ghi: {n_new} mới, {n_upd} ghi đè -> {args.jsonl} "
          f"(tổng {len(rows)})", flush=True)

    if not args.merge:
        return

    gt = json.load(open(args.gt, encoding="utf-8"))
    hist = {i: 0 for i in range(6)}
    src = {}
    n = 0
    for s in gt["samples"]:
        d = rows.get(s["query_id"])
        if not d:
            continue
        s["query"], s["source"] = d["query"], d["source"]
        s["distinct"], s["distinct_note"] = d["distinct"], d["note"]
        s["needs_review"] = d["distinct"] < MIN_DISTINCT
        hist[d["distinct"]] += 1
        src[d["source"]] = src.get(d["source"], 0) + 1
        n += 1
    gt["meta"]["n_with_query"] = n
    gt["meta"]["n_needs_review"] = sum(1 for s in gt["samples"] if s.get("needs_review"))
    gt["meta"]["distinct_hist"] = hist
    gt["meta"]["by_source"] = src
    with open(args.gt, "w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False, indent=2)
    print(f"gộp vào {args.gt}: {n}/{len(gt['samples'])} mẫu có truy vấn", flush=True)
    print(f"  nguồn: {src}", flush=True)
    print(f"  độ phân biệt (1=chung chung..5=riêng): {hist}", flush=True)
    print(f"  cần rà tay (distinct<{MIN_DISTINCT}): {gt['meta']['n_needs_review']}", flush=True)


if __name__ == "__main__":
    main()
