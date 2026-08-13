"""
Gom thư mục keyframe đang PHẲNG thành cây theo BỘ (L21, L22, ...).

    python -u scripts/group_keyframes_by_collection.py --dry-run    # xem trước, không đụng gì
    python -u scripts/group_keyframes_by_collection.py
    python -u scripts/group_keyframes_by_collection.py --exclude L29
    python -u scripts/group_keyframes_by_collection.py --also-meta  # gom cả data/output/<video_id>/

Trước:  data/output/keyframes/L21_V001/  L21_V002/ ... (344 thư mục cùng một cấp)
Sau  :  data/output/keyframes/L21/L21_V001/  L21/L21_V002/ ...

CHẠY LẠI ĐƯỢC NHIỀU LẦN: thư mục đã gom rồi thì bỏ qua. Pipeline vẫn ghi PHẲNG, nên sau
mỗi lần chạy bộ mới thì chạy lại script này để gom tiếp.

⚠️ HAI ĐIỀU QUAN TRỌNG:

1. `keyframe_path` nằm trong keyframes.jsonl và keyframe_meta.jsonl/.csv trỏ tới đường dẫn
   CŨ. Script tự cập nhật hết. Nếu không, mọi file metadata thành trỏ vào hư không.

2. KHÔNG gom bộ đang chạy dở — pipeline đang ghi vào đó, di chuyển giữa chừng sẽ hỏng.
   Script tự bỏ qua thư mục vừa được sửa dưới `--busy-sec` giây (mặc định 120), và có
   `--exclude` để loại trừ tay.

Hardlink trong data/btc_format KHÔNG bị ảnh hưởng: hardlink trỏ tới inode, di chuyển file
trong cùng ổ đĩa không làm đứt liên kết.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
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

KF_ROOT = "data/output/keyframes"
META_ROOT = "data/output"


def _coll(video_id: str) -> str:
    """L21_V001 -> L21"""
    return video_id.split("_")[0]


def _recent(path: str, secs: float) -> bool:
    """Thư mục vừa bị ghi trong `secs` giây gần đây -> có thể đang chạy dở."""
    try:
        newest = max((os.path.getmtime(os.path.join(path, f)) for f in os.listdir(path)),
                     default=os.path.getmtime(path))
        return (time.time() - newest) < secs
    except OSError:
        return True


def _fix_paths(video_id: str, old_prefix: str, new_prefix: str) -> int:
    """Sửa keyframe_path trong mọi file metadata của video này."""
    n = 0
    d = os.path.join(META_ROOT, video_id)
    for name in ("keyframes.jsonl", "keyframe_meta.jsonl"):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            continue
        rows = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                kp = r.get("keyframe_path", "")
                if kp:
                    norm = kp.replace("\\", "/")
                    if norm.startswith(old_prefix):
                        r["keyframe_path"] = new_prefix + norm[len(old_prefix):]
                        n += 1
                rows.append(r)
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    p = os.path.join(d, "keyframe_meta.csv")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            rd = list(csv.DictReader(f))
            fields = rd[0].keys() if rd else []
        if rd and "keyframe_path" in fields:
            for r in rd:
                norm = (r["keyframe_path"] or "").replace("\\", "/")
                if norm.startswith(old_prefix):
                    r["keyframe_path"] = new_prefix + norm[len(old_prefix):]
            with open(p, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(fields))
                w.writeheader()
                w.writerows(rd)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--exclude", nargs="*", default=[], help="bộ cần bỏ qua, vd L29")
    ap.add_argument("--busy-sec", type=float, default=120,
                    help="bỏ qua thư mục vừa ghi dưới ngần này giây (đang chạy dở)")
    ap.add_argument("--also-meta", action="store_true",
                    help="gom cả data/output/<video_id>/ (thư mục metadata)")
    args = ap.parse_args()

    if not os.path.isdir(KF_ROOT):
        print(f"Không có {KF_ROOT}", flush=True)
        return

    flat = sorted(x for x in os.listdir(KF_ROOT)
                  if os.path.isdir(os.path.join(KF_ROOT, x)) and "_" in x)
    if not flat:
        print("Không còn thư mục phẳng nào — đã gom xong từ trước.", flush=True)
        return

    by_coll = {}
    skip_busy, skip_excl = [], []
    for v in flat:
        c = _coll(v)
        if c in args.exclude:
            skip_excl.append(v); continue
        if _recent(os.path.join(KF_ROOT, v), args.busy_sec):
            skip_busy.append(v); continue
        by_coll.setdefault(c, []).append(v)

    print(f"{len(flat)} thư mục phẳng | gom {sum(len(v) for v in by_coll.values())} "
          f"| bỏ qua {len(skip_excl)} (--exclude) + {len(skip_busy)} (đang ghi)", flush=True)
    for c, vs in sorted(by_coll.items()):
        print(f"   {c}: {len(vs)} video -> {KF_ROOT}/{c}/", flush=True)
    if skip_busy:
        print(f"   ⚠ đang ghi, để lần sau: {skip_busy[:5]}"
              f"{' ...' if len(skip_busy) > 5 else ''}", flush=True)

    if args.dry_run:
        print("\n(--dry-run: chưa đụng gì)", flush=True)
        return

    moved = fixed = 0
    for c, vs in sorted(by_coll.items()):
        os.makedirs(os.path.join(KF_ROOT, c), exist_ok=True)
        for v in vs:
            src = os.path.join(KF_ROOT, v)
            dst = os.path.join(KF_ROOT, c, v)
            if os.path.exists(dst):
                print(f"   ⚠ {dst} đã tồn tại -> bỏ qua {v}", flush=True)
                continue
            shutil.move(src, dst)
            moved += 1
            fixed += _fix_paths(v, f"{KF_ROOT}/{v}/", f"{KF_ROOT}/{c}/{v}/")

    print(f"\nĐã chuyển {moved} thư mục, sửa {fixed} đường dẫn trong metadata.", flush=True)

    if args.also_meta:
        flat_m = sorted(x for x in os.listdir(META_ROOT)
                        if os.path.isdir(os.path.join(META_ROOT, x)) and "_" in x
                        and _coll(x) not in args.exclude)
        n = 0
        for v in flat_m:
            c = _coll(v)
            os.makedirs(os.path.join(META_ROOT, c), exist_ok=True)
            dst = os.path.join(META_ROOT, c, v)
            if not os.path.exists(dst):
                shutil.move(os.path.join(META_ROOT, v), dst); n += 1
        print(f"Gom thêm {n} thư mục metadata -> {META_ROOT}/<bộ>/<video_id>/", flush=True)
        print("⚠ Sau bước này, scripts/export_keyframe_meta.py và package_btc_format.py "
              "sẽ KHÔNG tìm thấy video (chúng quét phẳng data/output/).", flush=True)


if __name__ == "__main__":
    main()
