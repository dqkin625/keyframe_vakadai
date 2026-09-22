"""
Dựng ground truth JSON để tự chấm hệ thống tìm kiếm keyframe.

    python -u scripts/build_search_gt.py add batch.json     # nạp thêm 1 lô query
    python -u scripts/build_search_gt.py build              # xuất file GT cuối
    python -u scripts/build_search_gt.py stats

`batch.json` là danh sách: [{"file": "L28_V001/shot0219_f0029755.jpg", "query": "..."}, ...]

Các trường còn lại SUY RA TỪ TÊN FILE, không nhập tay:
    video_id       lấy từ đường dẫn
    shot_id        "shot0219" — phần đầu tên file
    frame_idx      29755 — phần sau "_f"
    range_frame_id [frame_idx - 6, frame_idx + 6]

Biên ±6 frame do người dùng quy định: hệ trả về frame nào rơi trong khoảng này thì tính đúng.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

STORE = "data/eval/search_gt_raw.jsonl"
OUT = "data/eval/search_gt.json"
KF_ROOT = "data/output/keyframes/L28"
MARGIN = 6

_PAT = re.compile(r"(shot\d+)_f(\d+)\.jpg$")


def parse(rel: str) -> dict:
    rel = rel.replace("\\", "/")
    video_id = rel.split("/")[-2] if "/" in rel else ""
    m = _PAT.search(rel)
    if not m:
        raise ValueError(f"Tên file không đúng dạng shotXXXX_fXXXXXXX.jpg: {rel}")
    shot_id, frame_idx = m.group(1), int(m.group(2))
    return {"video_id": video_id, "shot_id": shot_id, "frame_idx": frame_idx}


def load_store() -> list:
    if not os.path.exists(STORE):
        return []
    return [json.loads(l) for l in open(STORE, encoding="utf-8") if l.strip()]


def cmd_add(path: str):
    batch = json.load(open(path, encoding="utf-8"))
    rows = {r["file"]: r for r in load_store()}
    n_new = n_upd = 0
    for b in batch:
        rel = b["file"].replace("\\", "/")
        info = parse(rel)
        full = os.path.join(KF_ROOT, rel) if not rel.startswith("data/") else rel
        if not os.path.exists(full):
            print(f"  ⚠ KHÔNG THẤY ẢNH, bỏ qua: {full}", flush=True)
            continue
        rec = {"file": rel, "query": b["query"].strip(), **info}
        if rel in rows:
            n_upd += 1
        else:
            n_new += 1
        rows[rel] = rec
    os.makedirs(os.path.dirname(STORE), exist_ok=True)
    with open(STORE, "w", encoding="utf-8") as f:
        for k in sorted(rows):
            f.write(json.dumps(rows[k], ensure_ascii=False) + "\n")
    print(f"nạp {len(batch)}: {n_new} mới, {n_upd} ghi đè -> tổng {len(rows)}", flush=True)


def cmd_build(limit: int):
    rows = load_store()
    rows.sort(key=lambda r: (r["video_id"], r["frame_idx"]))
    if limit:
        rows = rows[:limit]
    out = []
    for i, r in enumerate(rows, 1):
        out.append({
            "query_id": f"q{i:03d}",
            "Queries": r["query"],
            "video_id": r["video_id"],
            "range_frame_id": [max(0, r["frame_idx"] - MARGIN), r["frame_idx"] + MARGIN],
            "shot_id": [r["shot_id"]],
        })
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"ghi {OUT}: {len(out)} query", flush=True)
    if out:
        print(json.dumps(out[0], ensure_ascii=False, indent=2), flush=True)


def cmd_stats():
    import collections
    rows = load_store()
    c = collections.Counter(r["video_id"] for r in rows)
    ln = [len(r["query"].split()) for r in rows]
    print(f"{len(rows)} query | {len(c)} video", flush=True)
    if ln:
        print(f"độ dài: min={min(ln)} tb={sum(ln)/len(ln):.1f} max={max(ln)} từ", flush=True)
    print(dict(sorted(c.items())), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("add", "build", "stats"))
    ap.add_argument("path", nargs="?")
    ap.add_argument("--limit", type=int, default=100)
    a = ap.parse_args()
    if a.cmd == "add":
        cmd_add(a.path)
    elif a.cmd == "build":
        cmd_build(a.limit)
    else:
        cmd_stats()
