"""KIỂM keyframe đã sinh có THẬT SỰ KHÁC NHAU không (chống trùng/gần-trùng).

    python scripts/check_keyframe_distinct.py [data/output/L21_V001] [--all]

Bước dedup lọc keyframe bằng HISTOGRAM (HSV). Script này kiểm bằng thước đo ĐỘC LẬP:
  - Nếu có clip_keyframe.npy: dùng COSINE CLIP (ngữ nghĩa) — độc lập với histogram -> mạnh nhất.
  - Nếu chỉ có ảnh: dùng histogram (cùng họ với dedup, yếu hơn nhưng vẫn bắt được trùng nặng).

Đo TRONG TỪNG SHOT (keyframe cùng shot mới có nguy cơ trùng; khác shot thì đương nhiên khác):
  - cosine giữa keyframe LIỀN NHAU + cặp GIỐNG NHẤT trong shot.
  - đếm % vượt các mốc gần-trùng: 0.95 / 0.98 / 0.995 (0.995 ~ gần như y hệt = phí).
Lưu ý ĐỌC ĐÚNG: shot phỏng vấn/tĩnh thì keyframe GIỐNG nhau là BÌNH THƯỜNG (cùng cảnh, đổi
chút tư thế) — chỉ coi là PHÍ khi cosine ~ 0.995+ (gần như trùng khít).
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def check_one(outdir):
    kfp = os.path.join(outdir, "keyframes.jsonl")
    if not os.path.exists(kfp):
        print(f"  THIẾU {kfp}"); return
    kfs = load_jsonl(kfp)
    clip_p = os.path.join(outdir, "clip_keyframe.npy")

    print(f"\n=== {outdir}  ({len(kfs)} keyframe) ===")

    # ma trận đặc trưng: CLIP nếu có, else histogram từ ảnh
    if os.path.exists(clip_p):
        E = np.load(clip_p).astype(np.float32)
        if len(E) != len(kfs):
            print(f"  ⚠️ clip.npy ({len(E)}) lệch keyframes.jsonl ({len(kfs)}) -> bỏ CLIP")
            E = None
        else:
            E = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-8, None)
            metric = "CLIP cosine (độc lập với dedup)"
    else:
        E = None
    if E is None:
        import cv2
        def hist(im):
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(h, h, 0, 1, cv2.NORM_MINMAX)
            return h.flatten()
        feats = []
        for k in kfs:
            im = cv2.imread(k["keyframe_path"])
            feats.append(hist(im) if im is not None else None)
        # cosine trên histogram
        def sim(a, b):
            if a is None or b is None:
                return 0.0
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        metric = "histogram cosine (cùng họ dedup)"
        E = ("hist", feats, sim)

    def cos(i, j):
        if isinstance(E, tuple):
            return E[2](E[1][i], E[1][j])
        return float(E[i] @ E[j])

    # gom theo shot (giữ chỉ số toàn cục)
    by_shot = defaultdict(list)
    for i, k in enumerate(kfs):
        by_shot[k["shot_index"]].append(i)

    consec, worst = [], []          # cosine cặp liền nhau; (cos, shot, frameA, frameB)
    multi = 0
    for si, idxs in by_shot.items():
        idxs.sort(key=lambda i: kfs[i]["frame_index"])
        if len(idxs) > 1:
            multi += 1
        for a, b in zip(idxs, idxs[1:]):
            c = cos(a, b)
            consec.append(c)
            worst.append((c, si, kfs[a]["frame_index"], kfs[b]["frame_index"]))

    if not consec:
        print("  (mỗi shot chỉ 1 keyframe -> không có cặp để so)"); return
    consec = np.array(consec)
    print(f"  thước đo: {metric}")
    print(f"  shot có >1 keyframe: {multi}/{len(by_shot)} | tổng cặp liền nhau: {len(consec)}")
    print(f"  cosine cặp LIỀN NHAU trong shot: trung vị={np.median(consec):.3f}  "
          f"p90={np.percentile(consec,90):.3f}  max={consec.max():.3f}")
    for thr, tag in [(0.95, "rất giống"), (0.98, "gần trùng"), (0.995, "≈ TRÙNG KHÍT (phí)")]:
        n = int((consec >= thr).sum())
        print(f"    ≥ {thr} ({tag}): {n}/{len(consec)} ({100*n/len(consec):.1f}%)")
    # top cặp giống nhất
    worst.sort(reverse=True)
    print("  5 cặp GIỐNG NHẤT (cosine | shot | frameA→frameB):")
    for c, si, fa, fb in worst[:5]:
        print(f"    {c:.4f} | shot {si} | {fa} → {fb} (cách {fb-fa} frame)")


def main():
    args = sys.argv[1:]
    all_mode = "--all" in args
    targets = [a for a in args if not a.startswith("--")]
    if all_mode:
        base = "data/output"
        targets = [os.path.join(base, d) for d in sorted(os.listdir(base))
                   if os.path.isfile(os.path.join(base, d, "keyframes.jsonl"))]
    if not targets:
        targets = ["data/output/L21_V001"]
    for t in targets:
        check_one(t)


if __name__ == "__main__":
    main()
