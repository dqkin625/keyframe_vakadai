"""
Bước 2: sinh câu truy vấn KIS tiếng Việt cho từng mẫu trong data/eval/kis_gt.json.

    python -u scripts/gen_kis_queries.py > logs/gen_kis.log 2>&1
    python -u scripts/gen_kis_queries.py --only-missing      # chạy tiếp phần còn thiếu

NGUYÊN TẮC CHỐNG THỔI PHỒNG ĐIỂM (quan trọng nhất ở file này):
  Truy vấn sinh từ **ẢNH**, KHÔNG cho model nhìn caption/ASR/metadata. Nếu sinh từ caption,
  model dùng lại đúng từ ngữ trong caption -> kênh caption ăn điểm giả, đo xong tưởng hệ
  mạnh nhưng thi thật rớt. (docs/PIPELINE.md §6 đã cảnh báo.)

Model tự chấm luôn ĐỘ PHÂN BIỆT 1-5 của truy vấn nó vừa viết. Kho này là bản tin thời sự:
rất nhiều cảnh "người mặc áo trắng trả lời phỏng vấn" giống hệt nhau. Truy vấn chung chung
thì bộ chấm tự động vô nghĩa — hệ trả về một đoạn KHÁC cũng đúng nhưng bị tính là sai.
=> Mẫu có distinct <= 2 bị đánh dấu `needs_review`, không dùng để kết luận.

Ghi tăng dần ra JSONL -> tắt giữa chừng chạy lại không mất việc đã làm.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

GT_PATH = "data/eval/kis_gt.json"
JSONL_PATH = "data/eval/kis_queries.jsonl"

PROMPT = """Bạn đang soạn ĐỀ THI cho cuộc thi truy vấn video tiếng Việt (Textual KIS).
Nhìn tấm ảnh dưới đây và viết MỘT câu truy vấn mô tả cảnh trong ảnh, đúng văn phong ban giám khảo.

Văn phong mẫu của ban giám khảo:
"Tìm video về một diễn giả mặc áo đỏ phát biểu tại một cuộc họp báo ngoài trời, phía sau có nhiều cây xanh."

QUY TẮC BẮT BUỘC:
1. Đúng MỘT câu tiếng Việt, 15-40 từ, bắt đầu bằng "Tìm ".
2. Chỉ mô tả thứ NHÌN THẤY ĐƯỢC: chủ thể, hành động, màu sắc, trang phục, đồ vật, bối cảnh,
   vị trí trái/phải/trước/sau.
3. Ưu tiên CHI TIẾT PHÂN BIỆT. Kho video này là bản tin thời sự, có hàng nghìn cảnh
   "người mặc áo trắng đang nói". Phải nêu được cái riêng của cảnh này.
4. TUYỆT ĐỐI KHÔNG:
   - nhắc logo/watermark của đài (HTV, Tuổi Trẻ, VTV...) — cảnh nào cũng có, vô ích;
   - chép nguyên văn dòng chữ chạy trên màn hình;
   - dùng các từ "ảnh", "hình", "khung hình", "keyframe", "bức ảnh này", "video này";
   - bịa tên người, tên địa danh, ngày tháng, số liệu không nhìn thấy được trong ảnh.
5. Không suy diễn thứ không có trong ảnh.

Sau khi viết xong, TỰ CHẤM độ phân biệt của truy vấn theo thang:
  5 = gần như chắc chắn chỉ MỘT đoạn trong kho khớp mô tả này
  4 = ít đoạn khớp, có chi tiết riêng rõ
  3 = có thể vài chục đoạn khớp
  2 = cảnh khá chung chung
  1 = cảnh cực chung chung (chân dung nói chuyện, phòng họp, đường phố... không chi tiết riêng)

Trả về ĐÚNG một khối JSON, không thêm chữ nào khác:
{"query": "...", "distinct": 1-5, "note": "vì sao chấm mức đó, 1 câu ngắn"}"""

MIN_DISTINCT = 3          # dưới mức này -> needs_review


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def _parse(raw: str):
    """Bóc khối JSON khỏi câu trả lời (model hay bọc ```json ... ```)."""
    t = (raw or "").strip()
    if t.startswith("```"):
        t = t.split("```")[1] if "```" in t[3:] else t[3:]
        t = t[4:].strip() if t.lower().startswith("json") else t.strip()
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        d = json.loads(t[i:j + 1])
    except json.JSONDecodeError:
        return None
    q = str(d.get("query", "")).strip()
    if not q:
        return None
    try:
        dist = int(d.get("distinct", 0))
    except (TypeError, ValueError):
        dist = 0
    return {"query": q, "distinct": max(0, min(5, dist)), "note": str(d.get("note", "")).strip()}


def _load_done():
    """query_id đã sinh xong (đọc JSONL) -> chạy lại không làm lại."""
    done = {}
    if os.path.exists(JSONL_PATH):
        with open(JSONL_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    done[d["query_id"]] = d
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=GT_PATH)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--only-missing", action="store_true",
                    help="bỏ qua mẫu đã có trong kis_queries.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N mẫu đầu (để thử)")
    # config.yaml chỉnh cho caption theo shot (bắn dồn, chấp nhận rớt vài shot). Ở đây mỗi mẫu
    # là MỘT truy vấn của bộ test — rớt là thủng bộ đo. Nên mặc định kiên nhẫn hơn hẳn.
    ap.add_argument("--retries", type=int, default=10,
                    help="số lần gọi lại 1 ảnh khi 429 (config.yaml chỉ để 3 -> hay [hết quota])")
    ap.add_argument("--cooldown", type=float, default=25.0, help="giây nghỉ 1 bucket sau khi dính 429")
    ap.add_argument("--backend", choices=("api", "local"), default="api",
                    help="api = Gemma free tier (nhanh, dính quota); "
                         "local = Qwen3-VL-4B 4-bit trên GPU (chậm hơn, KHÔNG quota)")
    args = ap.parse_args()

    import cv2
    import yaml
    from src.captioning.captioner import GemmaAPICaptioner

    with open(args.gt, encoding="utf-8") as f:
        gt = json.load(f)
    samples = gt["samples"]
    if args.limit:
        samples = samples[:args.limit]

    done = _load_done() if args.only_missing else {}
    todo = [s for s in samples if s["query_id"] not in done]
    _log(f"tổng {len(samples)} mẫu, đã có {len(done)}, cần sinh {len(todo)}")
    if not todo:
        _log("không còn gì để sinh — nhảy sang bước gộp")
    else:
        cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))["caption"]
        cfg = dict(cfg, prompt=PROMPT,
                   api_max_retries=args.retries, api_cooldown_sec=args.cooldown)
        if args.backend == "local":
            # Free tier Gemma hay cạn quota giữa chừng -> thủng bộ đo. Qwen3-VL-4B nạp 4-bit
            # chạy vừa GPU 6GB, chậm hơn nhưng KHÔNG bao giờ 429. Một luồng (GPU nối tiếp).
            from src.captioning.captioner import Qwen3VLCaptioner
            cap = Qwen3VLCaptioner(dict(cfg, max_new_tokens=200))
            args.workers = 1
            _log("backend=local (Qwen3-VL-4B 4-bit) — 1 luồng, không giới hạn quota")
        else:
            cap = GemmaAPICaptioner(cfg)

        lock_f = open(JSONL_PATH, "a", encoding="utf-8")
        n_ok = n_fail = 0

        def _one(s):
            img = cv2.imread(s["keyframe_file"])
            if img is None:
                return s, None, "không đọc được ảnh"
            raw = cap._call([img], prompt=PROMPT)
            d = _parse(raw)
            return s, d, raw[:80] if d is None else ""

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_one, s) for s in todo]
            for k, fut in enumerate(as_completed(futs), 1):
                s, d, err = fut.result()
                if d is None:
                    n_fail += 1
                    _log(f"  {k}/{len(todo)} {s['query_id']} LỖI: {err}")
                    continue
                rec = {
                    "query_id": s["query_id"],
                    "query": d["query"],
                    "distinct": d["distinct"],
                    "note": d["note"],
                    "source": "llm",
                }
                lock_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                lock_f.flush()
                n_ok += 1
                if k % 10 == 0 or k == len(todo):
                    _log(f"  {k}/{len(todo)} xong (ok={n_ok}, lỗi={n_fail})")
        lock_f.close()
        _log(f"sinh xong: {n_ok} ok, {n_fail} lỗi")

    # ---- gộp JSONL vào kis_gt.json ----
    done = _load_done()
    n_filled = 0
    dist_hist = {i: 0 for i in range(6)}
    for s in gt["samples"]:
        d = done.get(s["query_id"])
        if not d:
            continue
        s["query"] = d["query"]
        s["source"] = d["source"]
        s["distinct"] = d["distinct"]
        s["distinct_note"] = d["note"]
        s["needs_review"] = d["distinct"] < MIN_DISTINCT
        dist_hist[d["distinct"]] += 1
        n_filled += 1

    gt["meta"]["n_with_query"] = n_filled
    gt["meta"]["n_needs_review"] = sum(1 for s in gt["samples"] if s.get("needs_review"))
    gt["meta"]["distinct_hist"] = dist_hist
    with open(args.gt, "w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False, indent=2)

    _log(f"gộp vào {args.gt}: {n_filled}/{len(gt['samples'])} mẫu có truy vấn")
    _log(f"  phân bố độ phân biệt (1=chung chung ... 5=riêng biệt): {dist_hist}")
    _log(f"  cần rà tay (distinct < {MIN_DISTINCT}): {gt['meta']['n_needs_review']} mẫu")


if __name__ == "__main__":
    main()
