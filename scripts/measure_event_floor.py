"""Do BIEN DO THAT cua duong cong PA / TURN tren shot TINH vs shot DONG.
Muc dich: chon nguong san (floor) de _norm01 khong keo gian nhieu thanh ranh gioi gia.
Chi decode ~4 phut dau L21_V001 cho nhanh."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np
from collections import defaultdict
from src.frame_sampling.shot_detector import autoshot_infer
from src.frame_sampling.keyframe_extractor import _flow_vec
from src.frame_sampling.event_segmenter import _pa_curve, _turn_curve, _smooth1d

VID = (sys.argv[1] if len(sys.argv) > 1
       else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "video", "Videos_L21_a", "L21_V001.mp4"))
MAX_FRAMES = 30 * 60 * 4          # 4 phut
STRIDE = 5

cap = cv2.VideoCapture(VID)
fps = float(cap.get(cv2.CAP_PROP_FPS))
small, cand = [], []
i = 0
while i < MAX_FRAMES:
    ok, fr = cap.read()
    if not ok: break
    s48 = cv2.resize(fr, (48, 27), interpolation=cv2.INTER_AREA)
    small.append(cv2.cvtColor(s48, cv2.COLOR_BGR2RGB))
    if i % STRIDE == 0:
        cand.append((i, cv2.cvtColor(s48, cv2.COLOR_BGR2GRAY)))
    i += 1
cap.release()
print(f"decode {i} frame @ {fps}fps, {len(cand)} ung vien")

shots = autoshot_infer(np.asarray(small, dtype=np.uint8), 0.5, fps, True)
print(f"{len(shots)} shot")

by_shot = defaultdict(list)
ss = sorted(shots, key=lambda s: s.start_frame)
si = 0
for f, g in cand:
    while si < len(ss)-1 and f > ss[si].end_frame: si += 1
    if f >= ss[si].start_frame: by_shot[ss[si].index].append((f, g))

w = max(1, int(round(0.5 * fps / STRIDE)))
rows = []
for s in shots:
    c = by_shot.get(s.index) or []
    if len(c) < 4: continue
    grays = [x[1] for x in c]
    per = [(0.,0.,0.,0.)]
    for k in range(1, len(c)):
        per.append(_flow_vec(grays[k-1], grays[k]))
    ww = min(w, max(1, len(c)//3))
    feats = np.asarray([g.astype(np.float32).ravel()/255.0 for g in grays], np.float32)
    pa = _smooth1d(_pa_curve(feats, ww), 1.0)
    tu = _smooth1d(_turn_curve(per, ww), 1.0)
    mag = float(np.mean([p[2] for p in per[1:]])) if len(per) > 1 else 0.0
    rows.append((mag, float(pa.max()-pa.min()), float(tu.max()-tu.min()), len(c)))

rows.sort()
def band(name, sel):
    if not sel: return
    pa = np.array([r[1] for r in sel]); tu = np.array([r[2] for r in sel])
    print(f"{name:>26} | n={len(sel):>3} | PA range: p50={np.median(pa):8.4f} p90={np.percentile(pa,90):8.4f}"
          f" | TURN range: p50={np.median(tu):7.4f} p90={np.percentile(tu,90):7.4f}")

print()
print("BIEN DO (max-min) cua duong cong, theo muc chuyen dong cua shot:")
band("TINH   (mag<0.2)",  [r for r in rows if r[0] < 0.2])
band("RAT NHE(0.2-0.5)",  [r for r in rows if 0.2 <= r[0] < 0.5])
band("NHE    (0.5-1.0)",  [r for r in rows if 0.5 <= r[0] < 1.0])
band("VUA    (1.0-2.0)",  [r for r in rows if 1.0 <= r[0] < 2.0])
band("MANH   (>2.0)",     [r for r in rows if r[0] >= 2.0])
print()
print("-> floor nen dat KHOANG p90 cua nhom TINH (de nhom do khong vuot nguong 0.25),")
print("   va PHAI nho hon nhieu so voi p50 cua nhom VUA/MANH (de khong bop chet tin hieu that).")
