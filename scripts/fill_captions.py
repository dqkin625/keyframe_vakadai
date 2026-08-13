"""
Fill lại caption cho các keyframe bị LỖI ("[hết quota]"/"[lỗi API]") trong 1 output đã chạy.

- KHÔNG chạy lại cả video: chỉ đọc ảnh keyframe đã lưu trên đĩa (keyframe_path) và
  caption lại ĐÚNG những record hỏng.
- Lặp NHIỀU VÒNG, nghỉ giữa các vòng để nhịp RPM của key hồi lại, cho tới khi hết lỗi
  (hoặc hết số vòng / 3 vòng liên tiếp không tiến triển).
- Sau khi xong: ghi lại keyframes.jsonl và tính lại caption_emb.npy cho khớp thứ tự.

Dùng:
    python -m scripts.fill_captions                       # mặc định data/output/POV
    python -m scripts.fill_captions --dir data/output/POV --rounds 20 --wait 20
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import yaml  # noqa: E402

from src.captioning import build_captioner            # noqa: E402
from src.pipeline import split_caption_ocr            # noqa: E402


def _is_err(cap: str) -> bool:
    return (not cap) or str(cap).startswith(("[lỗi", "[hết quota", "["))


def _load_img(path: str):
    import cv2
    p = path.replace("\\", os.sep).replace("/", os.sep)
    img = cv2.imread(p)          # BGR (đúng định dạng captioner mong đợi)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dir", default="data/output/POV", help="thư mục output của 1 video")
    ap.add_argument("--rounds", type=int, default=20, help="số vòng thử tối đa")
    ap.add_argument("--wait", type=int, default=20, help="giây nghỉ giữa các vòng")
    args = ap.parse_args()

    kf_path = os.path.join(args.dir, "keyframes.jsonl")
    if not os.path.exists(kf_path):
        raise SystemExit(f"Không thấy {kf_path}")

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["caption"]["backend"] = "gemma"

    recs = [json.loads(l) for l in open(kf_path, encoding="utf-8")]
    n = len(recs)
    bad0 = [i for i, r in enumerate(recs) if _is_err(r.get("caption", ""))]
    print(f"[fill] {kf_path}: {n} keyframe, {len(bad0)} HỎNG cần fill", flush=True)
    if not bad0:
        print("[fill] Không có gì để fill. Xong.", flush=True)
        return

    captioner = build_captioner(cfg["caption"])

    no_progress = 0
    for rnd in range(1, args.rounds + 1):
        bad = [i for i, r in enumerate(recs) if _is_err(r.get("caption", ""))]
        if not bad:
            break
        # nạp ảnh cho các record hỏng (bỏ record mất ảnh)
        idxs, imgs = [], []
        for i in bad:
            im = _load_img(recs[i]["keyframe_path"])
            if im is not None:
                idxs.append(i); imgs.append(im)
        if not imgs:
            print("[fill] Không nạp được ảnh nào (thiếu file?). Dừng.", flush=True)
            break

        print(f"\n[fill] Vòng {rnd}/{args.rounds}: caption lại {len(imgs)} ảnh hỏng...", flush=True)
        caps = captioner.caption_batch(imgs)

        filled = 0
        for i, raw in zip(idxs, caps):
            if _is_err(raw):
                continue
            cap, ocr = split_caption_ocr(raw)
            recs[i]["caption"] = cap
            recs[i]["ocr"] = ocr
            filled += 1

        remain = sum(1 for r in recs if _is_err(r.get("caption", "")))
        print(f"[fill] Vòng {rnd}: +{filled} caption mới, còn {remain} hỏng", flush=True)

        # ghi tạm sau mỗi vòng (an toàn nếu bị ngắt giữa chừng)
        with open(kf_path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        if remain == 0:
            break
        no_progress = no_progress + 1 if filled == 0 else 0
        if no_progress >= 3:
            print("[fill] 3 vòng liên tiếp không tiến triển -> dừng (key có thể đang bị phạt nhịp).", flush=True)
            break
        if rnd < args.rounds:
            print(f"[fill] nghỉ {args.wait}s cho nhịp key hồi...", flush=True)
            time.sleep(args.wait)

    # ---- tính lại caption_emb.npy cho khớp ----
    ok = sum(1 for r in recs if not _is_err(r.get("caption", "")))
    print(f"\n[fill] KẾT QUẢ: OK={ok}/{n} ({100*ok//n}%), còn {n-ok} hỏng", flush=True)

    try:
        from src.embedding import CaptionEmbedder
        emb_cfg = dict(cfg["embedding"])
        emb_cfg["device"] = "cpu"
        embedder = CaptionEmbedder(emb_cfg)
        arr = embedder.encode_captions([r.get("caption", "") for r in recs])
        np.save(os.path.join(args.dir, "caption_emb.npy"), arr)
        print(f"[fill] Đã tính lại caption_emb.npy ({arr.shape})", flush=True)
    except Exception as e:
        print(f"[fill] (bỏ qua tính lại embedding: {e})", flush=True)

    print("[fill] Xong.", flush=True)


if __name__ == "__main__":
    main()
