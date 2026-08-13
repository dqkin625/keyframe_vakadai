"""
Sinh caption cho SHOT: đưa nhiều keyframe cùng shot vào VLM một lượt -> 1 caption mạch
lạc cho cả shot (temporal consistency, giống ReCap/U-CESE 2025).

Backend: qwen3-vl (mặc định, 4-bit vừa GPU 6GB), qwen2.5-vl, blip, blip2, mock.
Có thể thêm backend API bằng cách implement cùng interface ShotCaptioner.caption_shot().
"""
from __future__ import annotations

import re
from typing import List

import cv2
import numpy as np
from PIL import Image


def _bgr_to_pil(img: np.ndarray, max_side: int = 0) -> Image.Image:
    """BGR(numpy) -> PIL(RGB). max_side>0 thì thu nhỏ cạnh dài để hạn chế VRAM."""
    if max_side > 0:
        h, w = img.shape[:2]
        if max(h, w) > max_side:
            s = max_side / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def _select_frames(images: List[np.ndarray], k: int) -> List[np.ndarray]:
    """Chọn tối đa k ảnh trải đều trong shot."""
    if len(images) <= k:
        return images
    idx = np.linspace(0, len(images) - 1, k).round().astype(int)
    return [images[i] for i in idx]


_EV_LINE = re.compile(r"^\s*[\-\*•]?\s*(\d{1,2})\s*[\|\.\):\-]\s*(.+?)\s*$")


def _parse_event_lines(raw: str, n: int) -> List[str]:
    """Tách output 'gán nhãn sự kiện' thành ĐÚNG n nhãn, chịu được model lệch định dạng:
    dòng có số đầu -> gán đúng vị trí; không dòng nào có số -> gán theo thứ tự; lỗi API ->
    trả rỗng hết. Vị trí không có nhãn để chuỗi rỗng."""
    out = [""] * n
    if not raw or raw.lstrip().startswith(("[lỗi", "[hết quota")):
        return out
    numbered, plain = [], []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _EV_LINE.match(line)
        if m:
            numbered.append((int(m.group(1)), m.group(2).strip()))
        else:
            plain.append(line)
    if numbered:
        for k, text in numbered:
            if 1 <= k <= n and not out[k - 1]:
                out[k - 1] = text
        return out
    for i, text in enumerate(plain[:n]):        # model bỏ đánh số -> theo thứ tự
        out[i] = text
    return out


class ShotCaptioner:
    """Interface chung."""
    def caption_shot(self, frames: List[np.ndarray], memory: str = "", hint: str = "") -> str:
        """memory = tóm tắt các shot TRƯỚC (ReCap, U-CESE tr.7); mặc định bỏ qua."""
        raise NotImplementedError

    def caption_keyframe(self, target: np.ndarray,
                         context: List[np.ndarray] = None) -> str:
        """Caption cho 1 keyframe, có thể kèm cửa sổ ngữ cảnh lân cận (U-CESE tr.6).
        Mặc định (model 1 ảnh như BLIP): bỏ ngữ cảnh, chỉ mô tả ảnh đích."""
        return self.caption_shot([target])

    def caption_batch(self, images: List[np.ndarray]) -> List[str]:
        """Caption nhiều keyframe cùng lúc. Mặc định lặp tuần tự; backend hỗ trợ
        batch thật (Qwen) sẽ override."""
        return [self.caption_shot([img]) for img in images]


# --------------------------------------------------------------------------- #
class MockCaptioner(ShotCaptioner):
    def caption_shot(self, frames: List[np.ndarray], memory: str = "", hint: str = "") -> str:
        return f"[mock caption] shot gồm {len(frames)} keyframe."


# --------------------------------------------------------------------------- #
class Qwen25VLCaptioner(ShotCaptioner):
    def __init__(self, cfg: dict):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self.cfg = cfg
        self.frames_per_shot = cfg.get("frames_per_shot", 3)
        self.max_new_tokens = cfg.get("max_new_tokens", 128)
        self.prompt = cfg["prompt"]
        # prompt RIÊNG cho caption theo SHOT (mô tả diễn biến qua nhiều frame)
        self.shot_prompt = cfg.get("shot_prompt", self.prompt)

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
            cfg.get("dtype", "bfloat16"), torch.bfloat16
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg["model_id"], torch_dtype=dtype, device_map=cfg.get("device", "cuda")
        )
        self.processor = AutoProcessor.from_pretrained(cfg["model_id"])

    def caption_shot(self, frames: List[np.ndarray], memory: str = "", hint: str = "") -> str:
        frames = _select_frames(frames, self.frames_per_shot)
        content = [{"type": "image", "image": _bgr_to_pil(f)} for f in frames]
        content.append({"type": "text", "text": self.prompt})
        return self._generate([{"role": "user", "content": content}])

    def caption_keyframe(self, target: np.ndarray,
                         context: List[np.ndarray] = None) -> str:
        """U-CESE tr.6: đưa cửa sổ ngữ cảnh + ảnh đích, chỉ mô tả ảnh ĐÍCH."""
        content = []
        if context:
            ctx = _select_frames(context, self.cfg.get("context_window", 2) * 2)
            content.append({"type": "text",
                            "text": "Các ảnh sau là NGỮ CẢNH xung quanh (chỉ để hiểu bối cảnh, "
                                    "KHÔNG mô tả trực tiếp):"})
            content += [{"type": "image", "image": _bgr_to_pil(f)} for f in ctx]
        content.append({"type": "text", "text": "Ảnh CẦN MÔ TẢ:"})
        content.append({"type": "image", "image": _bgr_to_pil(target)})
        content.append({"type": "text", "text": self.prompt})
        return self._generate([{"role": "user", "content": content}])

    def _generate(self, messages) -> str:
        from qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)

        gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen)]
        out = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return out.strip()


# --------------------------------------------------------------------------- #
class BLIP2Captioner(ShotCaptioner):
    """Chỉ mô tả 1 keyframe đại diện (BLIP-2 không nhận nhiều ảnh)."""
    def __init__(self, cfg: dict):
        import torch
        from transformers import Blip2Processor, Blip2ForConditionalGeneration

        self.max_new_tokens = cfg.get("max_new_tokens", 64)
        model_id = cfg.get("model_id_blip2", "Salesforce/blip2-opt-2.7b")
        dtype = torch.float16 if cfg.get("device", "cuda") == "cuda" else torch.float32
        self.processor = Blip2Processor.from_pretrained(model_id)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map=cfg.get("device", "cuda")
        )

    def caption_shot(self, frames: List[np.ndarray], memory: str = "", hint: str = "") -> str:
        mid = _bgr_to_pil(frames[len(frames) // 2])
        inputs = self.processor(images=mid, return_tensors="pt").to(
            self.model.device, self.model.dtype
        )
        gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


# --------------------------------------------------------------------------- #
class Qwen3VLCaptioner(ShotCaptioner):
    """Qwen3-VL-4B-Instruct — caption tiếng Việt chi tiết, OCR mạnh, nhận nhiều ảnh/lượt.
    Nạp 4-bit (bitsandbytes) để vừa GPU 6GB."""
    def __init__(self, cfg: dict):
        import torch
        from transformers import (Qwen3VLForConditionalGeneration, AutoProcessor,
                                   BitsAndBytesConfig)

        self.cfg = cfg
        self.frames_per_shot = cfg.get("frames_per_shot", 3)
        self.max_new_tokens = cfg.get("max_new_tokens", 256)
        self.prompt = cfg["prompt"]
        # prompt RIÊNG cho caption theo SHOT (mô tả diễn biến qua nhiều frame)
        self.shot_prompt = cfg.get("shot_prompt", self.prompt)
        self.max_side = int(cfg.get("input_max_side", 512))   # giới hạn cạnh dài -> VRAM
        model_id = cfg.get("model_id", "Qwen/Qwen3-VL-4B-Instruct")

        kwargs = {"device_map": cfg.get("device", "cuda")}
        if cfg.get("load_4bit", True):
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        else:
            kwargs["torch_dtype"] = torch.float16

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def _generate(self, messages) -> str:
        import torch
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    def caption_shot(self, frames: List[np.ndarray], memory: str = "", hint: str = "") -> str:
        frames = _select_frames(frames, self.frames_per_shot)
        content = []
        if memory:   # ReCap: đưa bối cảnh các shot trước để caption mạch lạc theo thời gian
            content.append({"type": "text",
                            "text": f"Bối cảnh các cảnh TRƯỚC ĐÓ (chỉ để hiểu mạch truyện, "
                                    f"KHÔNG mô tả lại): {memory}"})
        content += [{"type": "image", "image": _bgr_to_pil(f, self.max_side)} for f in frames]
        prompt = self.shot_prompt
        if hint and self.cfg.get("inject_motion_hint", False):   # mặc định TẮT (flow đo camera, không phải người)
            prompt = (f"(Tham khảo — chuyển động nền/máy quay: {hint}. ĐỪNG tả máy quay; tập trung "
                      f"HÀNH ĐỘNG của NGƯỜI.)\n\n{prompt}")
        content.append({"type": "text", "text": prompt})
        return self._generate([{"role": "user", "content": content}])

    def caption_batch(self, images: List[np.ndarray]) -> List[str]:
        """Batch thật: gom nhiều keyframe vào một lượt generate. Batch size từ cfg.caption_batch_size."""
        import torch
        bs = int(self.cfg.get("caption_batch_size", 8))
        self.processor.tokenizer.padding_side = "left"
        out: List[str] = []
        for i in range(0, len(images), bs):
            chunk = images[i:i + bs]
            pil = [_bgr_to_pil(im, self.max_side) for im in chunk]
            msgs = [[{"role": "user", "content": [{"type": "image", "image": p},
                                                  {"type": "text", "text": self.prompt}]}]
                    for p in pil]
            texts = [self.processor.apply_chat_template(m, tokenize=False,
                                                        add_generation_prompt=True) for m in msgs]
            inp = self.processor(text=texts, images=pil, padding=True,
                                 return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                g = self.model.generate(**inp, max_new_tokens=self.max_new_tokens, do_sample=False)
            trimmed = [o[len(ii):] for ii, o in zip(inp.input_ids, g)]
            out += [s.strip() for s in self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)]
        return out

    def caption_keyframe(self, target: np.ndarray,
                         context: List[np.ndarray] = None) -> str:
        content = []
        if context:
            ctx = _select_frames(context, self.cfg.get("context_window", 2) * 2)
            content.append({"type": "text",
                            "text": "Các ảnh sau là NGỮ CẢNH xung quanh (chỉ để hiểu bối "
                                    "cảnh, KHÔNG mô tả trực tiếp):"})
            content += [{"type": "image", "image": _bgr_to_pil(f, self.max_side)} for f in ctx]
        content.append({"type": "text", "text": "Ảnh CẦN MÔ TẢ:"})
        content.append({"type": "image", "image": _bgr_to_pil(target, self.max_side)})
        content.append({"type": "text", "text": self.prompt})
        return self._generate([{"role": "user", "content": content}])


# --------------------------------------------------------------------------- #
class BLIPCaptioner(ShotCaptioner):
    """BLIP-base/large — model captioning nhẹ (~1-2GB), chỉ tiếng Anh; để kiểm chứng nhanh / máy yếu."""
    def __init__(self, cfg: dict):
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration

        self.max_new_tokens = cfg.get("max_new_tokens", 40)
        model_id = cfg.get("model_id_blip", "Salesforce/blip-image-captioning-large")
        self.device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=self.dtype
        ).to(self.device)

    def caption_shot(self, frames: List[np.ndarray], memory: str = "", hint: str = "") -> str:
        mid = _bgr_to_pil(frames[len(frames) // 2])
        inputs = self.processor(images=mid, return_tensors="pt").to(self.device, self.dtype)
        gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


# --------------------------------------------------------------------------- #
class _RateLimiter:
    """Rate limiter cho mỗi (key,model), 2 ràng buộc đồng thời: (1) tối đa `rpm` request/60s;
    (2) 2 request liên tiếp cách nhau >= 60/rpm giây. Ràng buộc (2) MẤU CHỐT: thiếu nó thì khi
    cửa sổ trống, `rpm` request nổ cùng lúc vào 1 key (burst) -> Google chặn -> 429. Thread-safe."""
    def __init__(self, rpm: int):
        import collections
        import threading
        self.rpm = max(1, rpm)
        self.min_gap = 60.0 / self.rpm      # khoảng cách tối thiểu giữa 2 request
        self.times = collections.deque()
        self.last = 0.0
        self.lock = threading.Lock()

    def acquire(self):
        import time
        while True:
            with self.lock:
                now = time.time()
                while self.times and now - self.times[0] >= 60.0:
                    self.times.popleft()
                gap_ok = (now - self.last) >= self.min_gap
                if len(self.times) < self.rpm and gap_ok:
                    self.times.append(now)
                    self.last = now
                    return
                wait = 0.0
                if len(self.times) >= self.rpm:                     # kẹt trần phút
                    wait = 60.0 - (now - self.times[0]) + 0.02
                if not gap_ok:                                      # kẹt giãn cách
                    wait = max(wait, self.min_gap - (now - self.last))
            time.sleep(min(max(wait, 0.0), 1.0))

    def load(self) -> float:
        """Tỉ lệ lấp đầy cửa sổ RPM (0=rỗng, 1=đầy) — không mutate, để load balancer so tải."""
        import time
        with self.lock:
            now = time.time()
            while self.times and now - self.times[0] >= 60.0:
                self.times.popleft()
            return len(self.times) / self.rpm


class _Bucket:
    """Một khe quota = (1 key, 1 model): rate-limiter RPM + bộ đếm RPD (request/ngày) riêng.
    Load balancer chọn bucket ít tải nhất để rải đều ra mọi key/model. Thread-safe."""
    def __init__(self, client, model, key_idx, rpm, rpd, account=""):
        import threading
        self.client = client
        self.model = model
        self.key_idx = key_idx
        self.account = account           # email account Google sở hữu key (theo dõi 429 theo account)
        self.rpm_lim = _RateLimiter(rpm)
        self.rpd = max(1, rpd)
        self.day_count = 0
        self.day_start = None            # mốc bắt đầu cửa sổ 24h (đặt khi request đầu tiên)
        self.dead = False                # key hỏng vĩnh viễn (403 / API_KEY_INVALID)
        self.cool_until = 0.0            # thời điểm (monotonic) hết nghỉ sau khi bị 429
        self.served = 0                  # tổng request đã phục vụ
        self.n429 = 0                    # số lần bucket này dính 429
        self.lock = threading.Lock()

    def hit_429(self) -> None:
        with self.lock:
            self.n429 += 1

    def day_ok(self) -> bool:
        """Còn quota NGÀY và key chưa chết? Cửa sổ 24h tự reset."""
        import time
        with self.lock:
            if self.dead:
                return False
            if self.day_start is not None and time.time() - self.day_start >= 86400:
                self.day_count = 0
                self.day_start = None
            return self.day_count < self.rpd

    def available(self) -> bool:
        """Bucket sẵn sàng nhận request? (chưa chết, còn quota ngày, KHÔNG đang cooldown do 429)."""
        import time
        return time.monotonic() >= self.cool_until and self.day_ok()

    def cool(self, secs: float) -> None:
        """Cho bucket nghỉ `secs` giây sau khi bị 429 -> _pick bỏ qua tới lúc đó (để cửa sổ
        quota Google trôi bớt trước khi thử lại)."""
        import time
        with self.lock:
            self.cool_until = max(self.cool_until, time.monotonic() + secs)

    def load(self) -> float:
        return self.rpm_lim.load()

    def acquire(self) -> None:
        """CHẶN tới khi request được phép (dưới trần RPM của bucket); rồi ghi nhận 1 request/ngày."""
        import time
        self.rpm_lim.acquire()
        with self.lock:
            if self.day_start is None:
                self.day_start = time.time()
            self.day_count += 1
            self.served += 1

    def kill(self) -> None:
        with self.lock:
            self.dead = True

    def exhaust_day(self) -> None:
        """Đánh dấu cạn RPD -> _pick bỏ qua tới hết cửa sổ 24h."""
        import time
        with self.lock:
            self.day_count = self.rpd
            if self.day_start is None:
                self.day_start = time.time()


# LOG LỖI API — in NGUYÊN VĂN, mỗi loại 1 lần + đếm tổng lúc kết thúc.
# ⚠️ Trước đây `_call_parts` nuốt sạch exception -> không biết "429" thực chất là gì.
import threading as _th

_ERR_SEEN = {}
_ERR_LOCK = _th.Lock()


def _err_kind(e) -> str:
    s = str(e)
    for code in ("429", "503", "500", "403", "401", "400"):
        if code in s[:200]:
            return f"{type(e).__name__}/{code}"
    return type(e).__name__


def _log_err(e, model: str = "") -> None:
    k = _err_kind(e)
    with _ERR_LOCK:
        n = _ERR_SEEN.get(k, 0) + 1
        _ERR_SEEN[k] = n
        first = n == 1
    if first:
        print(f"\n  ⚠️  LỖI API [{k}] (lần đầu, model={model}) — NGUYÊN VĂN:\n"
              f"      {str(e)[:600]}\n", flush=True)


def err_summary() -> dict:
    """Bảng đếm lỗi theo loại — gọi sau khi caption xong."""
    with _ERR_LOCK:
        return dict(_ERR_SEEN)


# Parse response 429 của Google. Body mẫu: "Quota exceeded for metric: .../requests,
# limit: 30, model: gemma-4-26b. Please retry in 39.8s." -> lấy retry-delay + metric + limit.
_METRIC_SEEN = {}
import re as _re


def _parse_429(msg: str):
    """Trả (retry_giây | None, tên_metric, limit | None, là_cạn_ngày) để cool bucket đúng thời
    gian Google báo, phân biệt RPM (chờ ~40s) vs RPD (chờ hàng giờ)."""
    rd = _re.search(r"retry in ([0-9.]+)s", msg)
    retry = float(rd.group(1)) if rd else None
    mt = _re.search(r"metric:\s*([^\s,]+)", msg)
    metric = mt.group(1).rsplit("/", 1)[-1] if mt else "unknown"
    lm = _re.search(r"limit:\s*([0-9]+)", msg)
    limit = int(lm.group(1)) if lm else None
    low = msg.lower()
    daily = ("perday" in low or "per day" in low or "per_day" in low
             or (limit is not None and limit >= 1000) or (retry is not None and retry > 300))
    with _ERR_LOCK:
        key = f"{metric}(limit={limit})"
        _METRIC_SEEN[key] = _METRIC_SEEN.get(key, 0) + 1
    return retry, metric, limit, daily


def metric_summary() -> dict:
    """Đếm 429 theo METRIC quota (RPM/RPD...) — cho biết trần nào bị chạm."""
    with _ERR_LOCK:
        return dict(_METRIC_SEEN)


# --------------------------------------------------------------------------- #
def _jpeg_bytes(img: np.ndarray, max_side: int = 768, quality: int = 90) -> bytes:
    """BGR numpy -> JPEG bytes (thu nhỏ để tiết kiệm token API)."""
    h, w = img.shape[:2]
    if max_side > 0 and max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


class GemmaAPICaptioner(ShotCaptioner):
    """Gọi Gemma/Gemini qua Google AI Studio API (free tier), song song nhiều luồng.
    Đọc key từ env GEMMA_API_KEY / GOOGLE_API_KEY hoặc .env. Có retry khi 429."""
    def __init__(self, cfg: dict):
        from google import genai

        self.cfg = cfg
        self.model = cfg.get("gemma_model", "gemma-4-26b-a4b-it")
        # Xen kẽ nhiều model: mỗi model có RPD riêng trên mỗi key -> gấp quota.
        self.models = cfg.get("gemma_models") or [self.model]
        self.prompt = cfg["prompt"]
        # prompt RIÊNG cho caption theo SHOT (mô tả diễn biến qua nhiều frame)
        self.shot_prompt = cfg.get("shot_prompt", self.prompt)
        self.frames_per_shot = cfg.get("frames_per_shot", 8)
        self.per_key_workers = int(cfg.get("api_max_workers", 8))
        self.max_side = int(cfg.get("input_max_side", 768))
        keys = self._load_keys()
        # 1 client / key (mỗi key quota riêng -> rải request ra nhiều key). Kiểm tra từng key,
        # chỉ giữ key hoạt động để không làm hỏng 1 phần caption.
        import time as _t
        self.clients = []
        self._client_keys = []          # key string tương ứng self.clients[i] (map -> account email)
        for i, k in enumerate(keys, 1):
            c = genai.Client(api_key=k)
            if cfg.get("api_validate_keys", True):
                ok, err = False, ""
                for _ in range(4):
                    try:
                        c.models.generate_content(model=self.model, contents=["ok"]); ok = True; break
                    except Exception as e:
                        err = str(e)
                        # 429/quota = TẠM THỜI (đang bị throttle) -> key vẫn tốt, chờ rồi thử lại
                        if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                            _t.sleep(3); continue
                        if "client has been closed" in err:
                            _t.sleep(1); continue
                        break                       # 403/lỗi khác = chết thật -> bỏ
                # 429 tới cùng vẫn coi là SỐNG (chỉ đang throttle), chỉ bỏ khi lỗi khác
                if not ok and not ("429" in err or "RESOURCE_EXHAUSTED" in err
                                   or "quota" in err.lower() or "client has been closed" in err):
                    print(f"[gemma] key {i} (...{k[-4:]}) BỎ QUA: {err[:50]}", flush=True)
                    continue
            self.clients.append(c); self._client_keys.append(k)
        if not self.clients:
            raise RuntimeError("Không có API key nào hoạt động.")
        self._emails = self._load_key_emails(self._client_keys)   # account email cho mỗi client
        # LOAD BALANCER: mỗi (key, model) là 1 bucket quota độc lập. CHỈ giới hạn RPM, không TPM.
        self.rpm = int(cfg.get("api_rpm_per_model", 30))     # trần RPM/bucket (Gemma = 30)
        self.rpd = int(cfg.get("api_rpd_per_model", 14400))  # trần request/ngày/bucket (Gemma = 14.400)
        self._cooldown_sec = float(cfg.get("api_cooldown_sec", 20))  # bucket nghỉ bao lâu sau 429
        self._max_retries = max(1, int(cfg.get("api_max_retries", 3)))  # số lần gọi API/request (chống storm)
        self._max_inflight = int(cfg.get("api_max_inflight", 0))     # cap TỔNG luồng (0 = theo bucket*workers)
        self._acct_cap = max(1, int(cfg.get("api_per_account_inflight", 1)))  # tối đa request đồng thời/account
        self._acct_gap = float(cfg.get("api_per_account_min_gap_sec", 0))     # giãn cách tối thiểu 2 request/account
        import collections as _col
        self._acct_inflight = _col.Counter()  # account -> số request đang bay (chống burst per-account)
        self._acct_last = {}                  # account -> mốc (monotonic) request gần nhất (để giãn cách)
        self._fallback_frames = int(cfg.get("fallback_frames", 3))   # số frame khi video lỗi -> fallback ảnh
        import threading as _thr
        # Bucket theo THỨ TỰ MODEL-MAJOR -> 2 bucket liền nhau luôn KHÁC KEY. Round-robin theo con
        # trỏ (bền qua các video) => mỗi request kế tiếp rơi vào 1 key khác nhau. Sửa gốc lỗi "dồn 1 key".
        self._buckets = [_Bucket(c, m, ki, self.rpm, self.rpd, self._emails[ki])
                         for m in self.models for ki, c in enumerate(self.clients)]
        self._cursor = 0
        self._cursor_lock = _thr.Lock()
        nb = len(self._buckets)
        print(f"[gemma] {len(self.clients)}/{len(keys)} key × {len(self.models)} model = {nb} bucket "
              f"-> trần ~{nb * self.rpm} req/phút, ~{nb * self.rpd} req/ngày "
              f"(load-balance round-robin, né bucket tải cao, tự loại key hỏng/cạn ngày)", flush=True)

    @staticmethod
    def _load_key_emails(client_keys) -> list:
        """Map mỗi key -> email account (đọc data/key_accounts.json). Key không có trong map -> '...last4'."""
        import os, json
        full = {}
        p = "data/key_accounts.json"
        if os.path.exists(p):
            try:
                full = json.load(open(p, encoding="utf-8")).get("full", {})
            except Exception:
                full = {}
        return [full.get(k, f"key(...{k[-4:]})") for k in client_keys]

    @staticmethod
    def _load_keys() -> list:
        """Gom tất cả key từ env + .env (GEMMA_API_KEY*, GOOGLE_API_KEY*), hỗ trợ nhiều key
        cách nhau dấu phẩy trong 1 biến."""
        import os
        raw = []

        def _match(name):
            return name.startswith(("GEMMA_API_KEY", "GOOGLE_API_KEY"))

        for name, val in os.environ.items():
            if _match(name) and val.strip():
                raw.append(val)
        if os.path.exists(".env"):
            with open(".env", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and _match(line.split("=", 1)[0].strip()):
                        raw.append(line.split("=", 1)[1])
        keys, seen = [], set()
        for v in raw:
            for k in v.replace('"', "").replace("'", "").split(","):   # tách theo dấu phẩy
                k = k.strip()
                if k and k not in seen:
                    seen.add(k); keys.append(k)
        if not keys:
            raise RuntimeError("Không tìm thấy GEMMA_API_KEY / GOOGLE_API_KEY (env hoặc .env)")
        return keys

    def _pick(self):
        """Chọn bucket (key,model) cho 1 request: round-robin từ con trỏ toàn cục, ưu tiên bucket
        RỖNG; nếu đều bận lấy bucket tải thấp nhất. Bỏ qua bucket chết/cạn-ngày. Trả None nếu hết."""
        import time
        now = time.monotonic()
        n = len(self._buckets)
        with self._cursor_lock:
            start = self._cursor
            self._cursor = (self._cursor + 1) % n
            best, best_load = None, 2.0
            for off in range(n):
                b = self._buckets[(start + off) % n]
                if not b.available():          # bỏ qua bucket chết / cạn ngày / đang cooldown do 429
                    continue
                if self._acct_inflight[b.account] >= self._acct_cap:   # account ĐANG bận tối đa -> bỏ qua
                    continue
                if self._acct_gap > 0 and now - self._acct_last.get(b.account, 0.0) < self._acct_gap:
                    continue                   # account CHƯA đủ giãn cách (min-gap) -> bỏ qua, pace theo account
                l = b.load()
                if l <= 0.0:
                    best = b; break            # bucket rỗng ĐẦU TIÊN từ con trỏ -> round-robin đều
                if l < best_load:
                    best, best_load = b, l
            if best is not None:
                self._acct_inflight[best.account] += 1   # RESERVE slot account (giải phóng sau khi gọi xong)
                self._acct_last[best.account] = now       # ghi mốc để giãn cách request kế tiếp cùng account
            return best

    def _release_account(self, b) -> None:
        with self._cursor_lock:
            if self._acct_inflight[b.account] > 0:
                self._acct_inflight[b.account] -= 1

    def _workers(self, b: int) -> int:
        """Số luồng cho batch b: theo bucket*per_key_workers, cap bởi api_max_inflight (nếu >0)
        vì quá nhiều luồng đồng thời gây 429."""
        w = min(max(1, b), len(self._buckets) * self.per_key_workers, 400)
        if self._max_inflight > 0:
            w = min(w, self._max_inflight)
        return w

    def _log_dist(self) -> None:
        """In thống kê phân phối tích lũy -> thấy tải có đều giữa key/model không."""
        if not self.cfg.get("api_log_dist", True):
            return
        n = len(self.clients)
        per_key = [sum(b.served for b in self._buckets if b.key_idx == k) for k in range(n)]
        per_model = {m: sum(b.served for b in self._buckets if b.model == m) for m in self.models}
        used = sum(1 for c in per_key if c > 0)
        dead = sum(1 for b in self._buckets if b.dead)
        near = sum(1 for b in self._buckets if not b.day_ok())
        print(f"      [gemma] phân phối tích lũy: {used}/{n} key có tải, "
              f"key min/max = {min(per_key)}/{max(per_key)} req, model = {per_model}"
              + (f", key CHẾT={dead}" if dead else "")
              + (f", bucket cạn-ngày={near}" if near else ""), flush=True)
        # Xếp hạng 429 theo account -> biết tài khoản nào hay dính nhất
        import collections
        acc429 = collections.Counter()
        accreq = collections.Counter()
        for b in self._buckets:
            acc429[b.account] += b.n429
            accreq[b.account] += b.served
        total429 = sum(acc429.values())
        if total429 > 0:
            ms = metric_summary()
            if ms:
                print(f"      [gemma] 429 THEO METRIC quota (Google báo): {ms}", flush=True)
            print(f"      [gemma] 429 THEO ACCOUNT (tổng {total429} lần 429), account hay dính NHẤT trước:", flush=True)
            for acc, c in sorted(acc429.items(), key=lambda x: -x[1]):
                if c == 0:
                    continue
                req = accreq[acc]
                rate = f"{100*c/req:.0f}%" if req else "-"
                print(f"          {c:>4} lần 429 / {req:>4} req ({rate})  {acc}", flush=True)

    def _call_parts(self, parts) -> str:
        """Gọi API cho `parts` (ảnh hoặc video). Khi 429 -> chuyển ngay sang key/model khác;
        403/key hỏng -> loại vĩnh viễn; cạn RPD -> loại tới hết ngày. Chỉ giới hạn RPM per bucket."""
        import time
        # ⚠️ KHÔNG set max_output_tokens: nó khiến Gemma trả finish=MAX_TOKENS -> `.text` RỖNG.
        max_tries = self._max_retries    # số lần gọi API/request. NHỎ (3) để 1 shot lỗi KHÔNG bắn 8
                                         # request -> chặn khuếch đại retry-storm (thủ phạm 429).
        api_tries = 0      # số lần gọi API thật (ngân sách lỗi) — không tính lần chờ bucket
        waits = 0          # số lần mọi bucket bận -> phải backoff
        while api_tries < max_tries and waits < 40:
            b = self._pick()
            if b is None:
                # MỌI bucket đang cooldown/cạn ngày -> BACKOFF (chờ cửa sổ quota trôi). Tăng dần, trần 20s.
                waits += 1
                time.sleep(min(20.0, 1.0 * 1.7 ** min(waits, 6)))
                continue
            try:
                b.acquire()                            # chờ dưới trần RPM (trong try -> finally chắc chắn release)
                r = b.client.models.generate_content(model=b.model, contents=parts)
                txt = (r.text or "").strip()
                if txt:
                    return txt
                # Response rỗng (không exception): finish=SAFETY/MAX_TOKENS/RECITATION -> lỗi mềm.
                api_tries += 1
                if api_tries >= max_tries:
                    return "[rỗng]"
                continue
            except Exception as e:
                api_tries += 1
                msg = str(e); low = msg.lower()
                _log_err(e, b.model)                   # LOG NGUYÊN VĂN 1 lần/loại lỗi
                if "429" in msg or "resource_exhausted" in low or "quota" in low:
                    b.hit_429()                        # đếm 429 theo bucket/account
                    retry, metric, limit, daily = _parse_429(msg)
                    if daily:
                        b.exhaust_day()                # cạn RPD -> né tới hết ngày
                    else:
                        # Google trả "Please retry in X.Xs" -> cool bucket đúng ngần đó (né đúng lúc)
                        b.cool((retry + 1.0) if retry else self._cooldown_sec)
                    # KHÔNG sleep ở request này -> _pick chuyển ngay sang bucket khác (còn dưới trần).
                    continue
                if any(c in msg for c in ("400", "401", "403")) and (
                        "api_key" in low or "api key" in low or "permission" in low
                        or "invalid" in low or "denied" in low):
                    b.kill()                           # key hỏng thật -> loại vĩnh viễn
                    continue
                b.cool(2.0)                            # lỗi lạ (500/503...) -> nghỉ ngắn
                if api_tries < max_tries:
                    time.sleep(1.5); continue
                return f"[lỗi API: {msg[:60]}]"
            finally:
                self._release_account(b)               # giải phóng slot account (mọi nhánh)
        return "[hết quota]"

    def _call(self, images: List[np.ndarray], memory: str = "", prompt=None) -> str:
        """Caption từ danh sách ảnh: dựng parts (memory + ảnh + prompt) rồi gọi _call_parts."""
        from google.genai import types
        parts = []
        if memory:
            parts.append(f"Bối cảnh các cảnh TRƯỚC (chỉ để hiểu mạch, KHÔNG mô tả lại): {memory}")
        parts += [types.Part.from_bytes(data=_jpeg_bytes(im, self.max_side),
                                        mime_type="image/jpeg") for im in images]
        parts.append(prompt or self.prompt)
        return self._call_parts(parts)

    def _shot_prompt_with_hint(self, hint: str = "") -> str:
        """Chèn motion hint vào đầu shot_prompt. MẶC ĐỊNH TẮT vì optical flow chủ yếu đo chuyển
        động MÁY QUAY -> dễ làm model tả camera. Bật bằng inject_motion_hint."""
        if not hint or not self.cfg.get("inject_motion_hint", False):
            return self.shot_prompt
        return (f"(Tham khảo — chuyển động NỀN/máy quay đo được: {hint}. ĐỪNG mô tả máy quay; chỉ dùng "
                f"để phân biệt đâu là do máy quay. HÃY TẬP TRUNG tả HÀNH ĐỘNG CỦA NGƯỜI.)\n\n{self.shot_prompt}")

    def caption_shot(self, frames: List[np.ndarray], memory: str = "", hint: str = "") -> str:
        # shot_prompt + tối đa frames_per_shot ảnh/request; _select_frames lấy frame rải đều.
        return self._call(_select_frames(frames, self.frames_per_shot), memory,
                          prompt=self._shot_prompt_with_hint(hint))

    def caption_shots(self, shots_frames: List[List[np.ndarray]], hints=None) -> List[str]:
        """Caption nhiều shot song song (khi KHÔNG dùng ReCap -> shot độc lập). Mỗi shot = 1
        request, load balancer rải round-robin qua (key, model) khác nhau.
        hints[i] = motion hint của shot i."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        b = len(shots_frames)
        workers = self._workers(b)
        results: List[str] = [None] * b

        def _one(i):
            return i, self._call(_select_frames(shots_frames[i], self.frames_per_shot), "",
                                 prompt=self._shot_prompt_with_hint(hints[i] if hints else ""))

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_one, i) for i in range(b)]
            for fu in tqdm(as_completed(futs), total=b, desc="      caption(shot)", unit="shot"):
                i, c = fu.result(); results[i] = c
        self._log_dist()
        return results

    # GÁN NHÃN CHO CHUỖI SỰ KIỆN (nguyên liệu TRAKE): ranh giới do tín hiệu cắt (event_segmenter),
    # VLM chỉ gán nhãn ngữ nghĩa. ⚠️ TUYỆT ĐỐI không để model tự khai mốc giây — nó chỉ thấy vài ảnh
    # rời rạc nên mọi con số giây là BỊA. Neo an toàn: gửi đúng 1 ảnh đại diện/sự kiện, theo thứ tự.
    _EVENT_PROMPT = (
        "Dưới đây là {n} ảnh trích theo THỨ TỰ THỜI GIAN từ CÙNG một cảnh quay. "
        "Mỗi ảnh đại diện cho MỘT đoạn hành động liên tiếp.\n"
        "Với MỖI ảnh, viết ĐÚNG MỘT DÒNG mô tả HÀNH ĐỘNG CỦA NGƯỜI (hoặc phương tiện) theo dạng:\n"
        "<số thứ tự ảnh> | <ai/cái gì> | <đang làm gì> | <với ai/cái gì hoặc bối cảnh ngắn>\n"
        "Quy tắc BẮT BUỘC:\n"
        "- Viết ĐÚNG {n} dòng, đánh số 1..{n}, KHÔNG thêm dòng nào khác, KHÔNG mở bài/kết luận.\n"
        "- TUYỆT ĐỐI KHÔNG ghi mốc thời gian/giây — bạn không biết chúng.\n"
        "- KHÔNG mô tả chuyển động của MÁY QUAY (không viết 'camera lia', 'góc máy đổi'...).\n"
        "- Chỉ tả điều NHÌN THẤY, không suy diễn. Nếu ảnh không có người thì tả vật/phương tiện.\n"
        "- Dùng TIẾNG VIỆT, mỗi dòng ngắn gọn (dưới 25 từ)."
    )

    def label_shot_events(self, images: List[np.ndarray]) -> List[str]:
        """Gán nhãn hành động cho 1 chuỗi sự kiện (1 ảnh/sự kiện, đúng thứ tự). Trả list dài
        BẰNG len(images); phần tử rỗng nếu model không trả đủ dòng."""
        n = len(images)
        if n == 0:
            return []
        prompt = self.cfg.get("event_prompt") or self._EVENT_PROMPT
        raw = self._call(images, "", prompt=prompt.format(n=n))
        return _parse_event_lines(raw, n)

    def label_events_batch(self, groups: List[List[np.ndarray]]) -> List[List[str]]:
        """Gán nhãn cho NHIỀU shot SONG SONG. groups[i] = ảnh đại diện các sự kiện của shot i."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        b = len(groups)
        out: List[List[str]] = [None] * b
        todo = [i for i in range(b) if groups[i]]
        for i in range(b):
            if not groups[i]:
                out[i] = []
        if not todo:
            return out

        def _one(i):
            return i, self.label_shot_events(groups[i])

        with ThreadPoolExecutor(max_workers=self._workers(len(todo))) as ex:
            futs = [ex.submit(_one, i) for i in todo]
            for fu in tqdm(as_completed(futs), total=len(todo),
                           desc="      nhãn sự kiện", unit="shot"):
                i, labs = fu.result(); out[i] = labs
        self._log_dist()
        return out

    # THỬ NGHIỆM: caption theo VIDEO CLIP của shot (hiểu chuyển động thật). Gemma nhận input
    # video; clip được cắt + giảm fps/res để token < trần TPM (16K/key).
    def _extract_clip(self, video_path: str, start_sec: float, end_sec: float):
        """Cắt clip = N frame rải đều bằng FAST-SEEK (-ss TRƯỚC -i, nhảy tới keyframe gần, không
        giải mã cả shot) rồi ghép thành mp4 1fps. Trả (bytes, n_frames, clip_fps=1.0)."""
        import os
        import subprocess
        import tempfile
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None, 0, 0.0
        dur = max(0.5, float(end_sec) - float(start_sec))
        # số frame: <= video_max_frames VÀ <= video_fps*dur (KHÔNG còn cap theo token budget)
        n = min(int(self.cfg.get("video_max_frames", 16)),
                max(1, int(round(float(self.cfg.get("video_fps", 1.5)) * dur))))
        n = max(1, n)
        side = int(self.cfg.get("video_max_side", 320))

        frames = []
        if dur <= float(self.cfg.get("video_fastseek_min_sec", 30)):
            # SHOT NGẮN: giải mã 1 lượt (nhanh cho đoạn ngắn, tránh N lần launch ffmpeg)
            fps = max(0.05, n / dur)
            tmp0 = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False); tmp0.close()
            cmd = [ff, "-y", "-ss", f"{start_sec:.3f}", "-to", f"{end_sec:.3f}", "-i", video_path,
                   "-vf", f"fps={fps:.4f},scale={side}:-2", "-an", "-loglevel", "error", tmp0.name]
            try:
                subprocess.run(cmd, capture_output=True, timeout=60)
                capv = cv2.VideoCapture(tmp0.name)
                while True:
                    ok, fr = capv.read()
                    if not ok:
                        break
                    frames.append(fr)
                capv.release()
            except Exception:
                pass
            finally:
                try:
                    os.unlink(tmp0.name)
                except OSError:
                    pass
        else:
            # SHOT DÀI: FAST-SEEK từng mốc (-ss trước -i -> nhảy tức thì, không giải mã cả shot).
            ts = [start_sec + dur * k / (n - 1) for k in range(n)] if n > 1 else [start_sec + dur / 2]
            for t in ts:
                cmd = [ff, "-ss", f"{t:.3f}", "-i", video_path, "-frames:v", "1",
                       "-vf", f"scale={side}:-2", "-f", "mjpeg", "-loglevel", "error", "pipe:"]
                try:
                    p = subprocess.run(cmd, capture_output=True, timeout=30)
                    if p.stdout:
                        img = cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR)
                        if img is not None:
                            frames.append(img)
                except Exception:
                    pass
        if not frames:
            return None, 0, 0.0

        h, w = frames[0].shape[:2]
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False); tmp.close()
        wr = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (w, h))  # 1fps
        for f in frames:
            wr.write(f)
        wr.release()
        try:
            data = open(tmp.name, "rb").read() if os.path.getsize(tmp.name) > 0 else None
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
        return data, len(frames), 1.0

    def caption_shot_video(self, video_path, start_sec, end_sec, hint=""):
        """Caption 1 shot bằng clip video (ghép từ N frame fast-seek). Trả caption, None nếu lỗi.
        Gắn video_metadata(fps=clip_fps) để Gemma sample đúng N frame -> token dưới trần TPM."""
        from google.genai import types
        data, n_frames, clip_fps = self._extract_clip(video_path, start_sec, end_sec)
        if not data:
            return None
        vpart = types.Part(inline_data=types.Blob(data=data, mime_type="video/mp4"),
                           video_metadata=types.VideoMetadata(fps=clip_fps))
        parts = [vpart, self._shot_prompt_with_hint(hint)]
        return self._call_parts(parts)

    def caption_shots_video(self, video_path, spans, frames_fallback, hints=None):
        """Caption nhiều shot bằng video, song song. Nếu video lỗi (clip rỗng hoặc trả '[lỗi]')
        -> FALLBACK sang caption ảnh (frames_fallback) cho shot đó.
        spans[i]=(start_sec,end_sec); frames_fallback[i]=list ảnh keyframe; hints[i]=motion hint.
        Trả (captions, n_video, n_fallback)."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        b = len(spans)
        workers = self._workers(b)
        results = [None] * b
        used = [None] * b   # "video" | "frame"

        def _one(i):
            hint = hints[i] if hints else ""
            cap = self.caption_shot_video(video_path, spans[i][0], spans[i][1], hint)
            if cap is None or cap.startswith("["):     # clip lỗi/API lỗi -> FALLBACK ảnh
                cap = self._call(_select_frames(frames_fallback[i], self._fallback_frames), "",
                                 prompt=self._shot_prompt_with_hint(hint))
                return i, cap, "frame"
            return i, cap, "video"

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_one, i) for i in range(b)]
            for fu in tqdm(as_completed(futs), total=b, desc="      caption(video)", unit="shot"):
                i, cap, how = fu.result()
                results[i] = cap; used[i] = how
        self._log_dist()
        return results, used.count("video"), used.count("frame")

    def caption_keyframe(self, target: np.ndarray, context: List[np.ndarray] = None) -> str:
        return self._call([target])

    def caption_batch(self, images: List[np.ndarray]) -> List[str]:
        """Bắn API song song cho nhiều ảnh. Load balancer tự rải mỗi request qua bucket ít tải
        nhất -> phủ đều mọi key/model, không dồn key."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        b = len(images)
        workers = self._workers(b)
        if b >= self.per_key_workers:
            print(f"      [gemma] batch {b} ảnh -> {len(self._buckets)} bucket, {workers} luồng "
                  f"(load-balance; trần {len(self._buckets) * self.rpm} req/phút)", flush=True)
        results = [None] * b

        def _one(i):
            return i, self._call([images[i]])

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_one, i) for i in range(b)]
            for f in tqdm(as_completed(futs), total=b, desc="      caption(API)", unit="ảnh"):
                i, c = f.result(); results[i] = c
        self._log_dist()
        return results


# --------------------------------------------------------------------------- #
def build_captioner(cfg: dict) -> ShotCaptioner:
    """cfg = block `caption` trong config.yaml."""
    backend = cfg.get("backend", "qwen3-vl").lower()
    if backend == "mock":
        return MockCaptioner()
    if backend == "blip":
        return BLIPCaptioner(cfg)
    if backend == "blip2":
        return BLIP2Captioner(cfg)
    if backend in ("qwen2.5-vl", "qwen2_5_vl"):
        return Qwen25VLCaptioner(cfg)
    if backend in ("gemma", "gemini", "api"):
        return GemmaAPICaptioner(cfg)
    return Qwen3VLCaptioner(cfg)   # "qwen3-vl" (mặc định)
