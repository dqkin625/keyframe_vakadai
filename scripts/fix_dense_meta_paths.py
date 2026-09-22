r"""Doi cot `path` trong dense_meta.parquet sang duong dan keyframe tren may nay.

File dense_meta.parquet do may khac build, cot `path` tro toi layout cua may do:

    D:\Programming\AIHCM\data\Our\L21_extracted\L21\L21_V001\shot0000_f0000000.jpg

Ba thanh phan cuoi (bo suu tap / video / ten file) la thu duy nhat con y nghia; ghep chung
vao thu muc keyframe cua may nay de ra duong dan that:

    E:\HCM_AI\data\output\keyframes\L21\L21_V001\shot0000_f0000000.jpg

Mac dinh chi CHAY THU va bao cao do phu. Them --apply de ghi de (co backup .bak).

    python scripts/fix_dense_meta_paths.py
    python scripts/fix_dense_meta_paths.py --apply
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import PureWindowsPath

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

META = r"E:\HCM_AI\AIHCM\v3\index\dense\dense_meta.parquet"
KEYFRAME_ROOT = r"E:\HCM_AI\data\output\keyframes"


def remap(p: str, video_id: str, root: str) -> str:
    """Bo suu tap suy ra TU video_id, khong lay tu path.

    Hai layout khac nhau cung ton tai trong dense_meta: mot loai co cap bo suu tap
    (L21_extracted/L21/L21_V001/x.jpg), mot loai phang khong co (keyframes/L26_V200/x.jpg).
    Lay 3 phan cuoi cua path se ra "keyframes" thay vi "L26" o layout thu hai, nen chi giu
    TEN FILE tu path, con bo suu tap va video lay tu video_id.
    """
    return os.path.join(root, video_id.split("_")[0], video_id, PureWindowsPath(p).name)


def scan(root: str) -> dict:
    """Quet 1 lan ra tap ten file moi thu muc video - nhanh hon goi exists() tung dong."""
    have = defaultdict(set)
    for coll in sorted(os.listdir(root)):
        cdir = os.path.join(root, coll)
        if not os.path.isdir(cdir):
            continue
        for vid in os.listdir(cdir):
            vdir = os.path.join(cdir, vid)
            if os.path.isdir(vdir):
                have[(coll, vid)] = set(os.listdir(vdir))
    return have


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="ghi de file that (mac dinh chi chay thu)")
    ap.add_argument("--meta", default=META)
    ap.add_argument("--root", default=KEYFRAME_ROOT)
    args = ap.parse_args()

    df = pd.read_parquet(args.meta)
    print(f"{len(df):,} dong | {df['video_id'].nunique()} video")
    print(f"path cu  : {df['path'].iloc[0]}")

    new = [remap(p, v, args.root) for p, v in zip(df["path"], df["video_id"])]
    new = pd.Series(new, index=df.index)
    print(f"path moi : {new.iloc[0]}\n")

    have = scan(args.root)
    ok, miss = Counter(), Counter()
    miss_videos = set()
    for p in new:
        coll, vid, fn = PureWindowsPath(p).parts[-3:]
        if fn in have.get((coll, vid), ()):
            ok[coll] += 1
        else:
            miss[coll] += 1
            miss_videos.add(f"{coll}/{vid}")

    total_ok, total_miss = sum(ok.values()), sum(miss.values())
    print(f"{'bo':<6} {'khop':>9} {'thieu':>9}")
    for coll in sorted(set(ok) | set(miss)):
        print(f"{coll:<6} {ok[coll]:>9,} {miss[coll]:>9,}")
    print(f"{'TONG':<6} {total_ok:>9,} {total_miss:>9,}   "
          f"({100 * total_ok / len(df):.1f}% anh tim thay tren dia)")

    if miss_videos:
        print(f"\n{len(miss_videos)} video thieu anh, vi du:")
        for v in sorted(miss_videos)[:12]:
            print(f"   {v}")

    if not args.apply:
        print("\n(chay thu - them --apply de ghi)")
        return

    bak = args.meta + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(args.meta, bak)
        print(f"\nbackup -> {bak}")
    df["path"] = new
    df.to_parquet(args.meta, index=False)
    print(f"da ghi -> {args.meta}")


if __name__ == "__main__":
    main()
