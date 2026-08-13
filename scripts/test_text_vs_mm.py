"""KIỂM CHỨNG: limit của Gemma là do TEXT/rate hay do MULTIMODAL (ảnh/video) ngầm?

    python scripts/test_text_vs_mm.py [N]

Bắn N request THUẦN TEXT (câu hỏi, không ảnh/video) qua ĐÚNG load-balancer + cơ chế pacing
mà luồng caption vừa rồi dùng (_call_parts: cùng 42 bucket, cùng per-account cap, cùng min-gap,
cùng cooldown/retry). N mặc định = 331 = đúng số shot của lần caption vừa chạy.

So sánh:
  - Caption (multimodal ảnh/video) lần vừa rồi: OK ~23%.
  - Text lần này: nếu OK ~100% với ÍT/0 lần 429  -> chốt limit là MULTIMODAL NGẦM, không phải
    limit text hay limit số-request/rate. Nếu text cũng hỏng nhiều -> limit ở tầng request/quota.

Prompt biến thiên theo chỉ số (tránh hiệu ứng cache prompt trùng) và yêu cầu output dài tương
đương caption (2-4 câu) để "độ dày" đầu ra không lệch — biến số DUY NHẤT đổi là: có ảnh/video hay không.
"""
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml

from src.captioning.captioner import GemmaAPICaptioner, metric_summary

N = int(sys.argv[1]) if len(sys.argv) > 1 else 331

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
cfg["caption"]["backend"] = "gemma"
cap = GemmaAPICaptioner(cfg["caption"])

prompts = [
    f"Bạn là hệ thống lập chỉ mục video tin tức. Hãy TƯỞNG TƯỢNG cảnh tin tức số {i} và viết "
    f"2-4 câu tiếng Việt mô tả: số người, trang phục, hành động và phương tiện trong cảnh. "
    f"Chỉ viết phần mô tả, không mở bài, không giải thích."
    for i in range(N)
]

res = {"OK": 0, "BAD": 0}
lock = threading.Lock()
lat = []


def one(i):
    t0 = time.time()
    txt = cap._call_parts([prompts[i]])
    dt = time.time() - t0
    ok = bool(txt and txt.strip() and not txt.strip().startswith("["))
    with lock:
        res["OK" if ok else "BAD"] += 1
        lat.append(dt)
    return ok


workers = cap._workers(N)
print(f"Bắn {N} request THUẦN TEXT qua load-balancer (workers={workers})...", flush=True)
t0 = time.time()
with ThreadPoolExecutor(max_workers=workers) as ex:
    futs = [ex.submit(one, i) for i in range(N)]
    done = 0
    for _ in as_completed(futs):
        done += 1
        if done % 50 == 0:
            print(f"  {done}/{N}  (OK={res['OK']} BAD={res['BAD']})", flush=True)
dt = time.time() - t0

import statistics as st
print("\n" + "=" * 56)
print(f"TEXT: {N} request trong {dt:.0f}s  ({N/max(dt,1)*60:.0f} req/phút)")
print(f"  OK  : {res['OK']}/{N} ({100*res['OK']/N:.0f}%)")
print(f"  LỖI : {res['BAD']}/{N} ({100*res['BAD']/N:.0f}%)")
print(f"  latency/request: trung vị {st.median(lat):.1f}s")
print(f"  429 theo metric (Google báo): {metric_summary()}")
print("-" * 56)
print(f"So sánh: caption MULTIMODAL (ảnh/video) lần vừa rồi = 23% OK.")
print(f"  -> text {100*res['OK']/N:.0f}% mà multimodal 23% ở CÙNG số request/pacing")
print(f"     => limit là MULTIMODAL NGẦM, không phải limit text/rate." if res['OK']/N > 0.8
      else "  -> text cũng thấp: nghi limit ở tầng quota/request, cần xét lại.")
print("=" * 56)
