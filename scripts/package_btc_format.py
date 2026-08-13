"""
Đóng gói keyframe đã sinh ra ĐÚNG cấu trúc BTC phát, để gửi cho người khác dùng ngay.

    python -u scripts/package_btc_format.py                      # tất cả video đã có output
    python -u scripts/package_btc_format.py --prefix L21         # chỉ 1 bộ
    python -u scripts/package_btc_format.py --copy               # copy thật thay vì hardlink

Cấu trúc xuất ra (khớp gói BTC):

    <out>/Keyframes_L21/L21_V001/001.jpg, 002.jpg, ...
    <out>/map-keyframes/L21_V001.csv       cột: n, pts_time, fps, frame_idx

ĐIỂM MẤU CHỐT — TÊN FILE:
    BTC đặt tên ảnh TUẦN TỰ khớp thẳng với cột `n` (001.jpg = dòng n=1). Code phía người
    nhận thường ghép `f"{n:03d}.jpg"` để tra ảnh. Pipeline của mình đặt tên
    `shot0000_f0000010.jpg` (mang thông tin shot + frame, tiện debug nhưng KHÔNG tương
    thích). Script này đổi sang quy ước BTC, đồng thời ghi `name_map.csv` để không mất
    thông tin shot/frame gốc.

MẶC ĐỊNH DÙNG HARDLINK, không copy: 86.000 ảnh ~7 GB, hardlink tạo tức thì và KHÔNG tốn
thêm dung lượng (cùng ổ NTFS). Zip lại vẫn ra file bình thường. Dùng --copy nếu cần bản
độc lập hoặc xuất sang ổ khác.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

BTC_FIELDS = ["n", "pts_time", "fps", "frame_idx"]


def _link_or_copy(src: str, dst: str, mode: str) -> str:
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    try:
        if os.path.exists(dst):
            os.remove(dst)
        os.link(src, dst)              # hardlink: tức thì, 0 byte thêm
        return "link"
    except OSError:
        shutil.copy2(src, dst)         # khác ổ / không phải NTFS -> đành copy
        return "copy"


def pack_video(video_id: str, out_root: str, mode: str) -> dict:
    src_meta = f"data/output/{video_id}/keyframe_meta.jsonl"
    if not os.path.exists(src_meta):
        return {}
    rows = [json.loads(l) for l in open(src_meta, encoding="utf-8") if l.strip()]
    if not rows:
        return {}
    rows.sort(key=lambda r: r["frame_idx"])

    coll = video_id.split("_")[0]                       # L21_V001 -> L21
    kf_dir = os.path.join(out_root, f"Keyframes_{coll}", video_id)
    map_dir = os.path.join(out_root, "map-keyframes")
    os.makedirs(kf_dir, exist_ok=True)
    os.makedirs(map_dir, exist_ok=True)

    btc_rows, name_map, n_link, n_copy = [], [], 0, 0
    for i, r in enumerate(rows, start=1):               # BTC đánh n từ 1
        src = r["keyframe_path"]
        if not os.path.exists(src):
            continue
        new = f"{i:03d}.jpg"
        how = _link_or_copy(src, os.path.join(kf_dir, new), mode)
        n_link += how == "link"
        n_copy += how == "copy"
        btc_rows.append({"n": i, "pts_time": r["pts_time"],
                         "fps": r["fps"], "frame_idx": r["frame_idx"]})
        name_map.append({"n": i, "btc_name": new,
                         "ten_goc": os.path.basename(src),
                         "shot_id": r["shot_id"], "keyframe_id": r["keyframe_id"],
                         "frame_idx": r["frame_idx"]})

    with open(os.path.join(map_dir, f"{video_id}.csv"), "w",
              encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BTC_FIELDS)
        w.writeheader()
        w.writerows(btc_rows)

    # giữ đường về tên gốc -> không mất thông tin shot/frame sau khi đổi tên
    with open(os.path.join(kf_dir, "name_map.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n", "btc_name", "ten_goc", "shot_id",
                                          "keyframe_id", "frame_idx"])
        w.writeheader()
        w.writerows(name_map)

    return {"n": len(btc_rows), "link": n_link, "copy": n_copy}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/btc_format")
    ap.add_argument("--prefix", default=None, help="chỉ đóng gói 1 bộ, vd L21")
    ap.add_argument("--only", default=None, help="chỉ 1 video")
    ap.add_argument("--copy", action="store_true", help="copy thật thay vì hardlink")
    args = ap.parse_args()

    vids = sorted(d for d in os.listdir("data/output")
                  if os.path.isdir(os.path.join("data/output", d))
                  and os.path.exists(os.path.join("data/output", d, "keyframe_meta.jsonl")))
    if args.prefix:
        vids = [v for v in vids if v.startswith(args.prefix)]
    if args.only:
        vids = [v for v in vids if v == args.only]
    if not vids:
        print("Không tìm thấy video nào có keyframe_meta.jsonl "
              "(chạy scripts/export_keyframe_meta.py trước)", flush=True)
        return

    mode = "copy" if args.copy else "link"
    print(f"{len(vids)} video | chế độ={mode} | ra: {args.out}", flush=True)

    tot = link = cp = 0
    per_coll = {}
    for i, v in enumerate(vids, 1):
        r = pack_video(v, args.out, mode)
        if not r:
            continue
        tot += r["n"]; link += r["link"]; cp += r["copy"]
        per_coll[v.split("_")[0]] = per_coll.get(v.split("_")[0], 0) + r["n"]
        if i % 20 == 0 or i == len(vids):
            print(f"   {i}/{len(vids)} video, {tot} ảnh", flush=True)

    print(f"\nXONG: {tot} ảnh ({link} hardlink, {cp} copy)")
    for c, n in sorted(per_coll.items()):
        print(f"   Keyframes_{c}: {n} ảnh")
    print(f"\nCấu trúc:")
    print(f"   {args.out}/Keyframes_<bo>/<video_id>/001.jpg ...")
    print(f"   {args.out}/map-keyframes/<video_id>.csv   (n, pts_time, fps, frame_idx)")
    print(f"   + name_map.csv trong mỗi thư mục video -> tra ngược ra shot_id/tên gốc")


if __name__ == "__main__":
    main()
