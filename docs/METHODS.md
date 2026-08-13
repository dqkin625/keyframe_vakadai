# Các phương pháp đã implement — giải thích & nguồn

Tài liệu này ghi lại mọi phương pháp/thuật toán dùng trong pipeline, để hiểu bản chất
và tra cứu về sau. Cập nhật mỗi khi thêm hoặc đổi phương pháp.

Nguồn chính (2 giải pháp HCMC AI Challenge 2025, loại "Excellent"):
- **U-CESE** (đội Nomial): https://arxiv.org/pdf/2605.23274
- **Vortex** (đội FocusOnFun): https://arxiv.org/pdf/2606.19682
- **AutoShot**: https://arxiv.org/pdf/2304.06116 · **TransNetV2**: https://arxiv.org/pdf/2008.04838

---

# A. FRAME SAMPLING (sample frame)

Mục tiêu: biến video (hàng vạn frame) thành ít keyframe đại diện, không mất nội dung.
Gồm 2 bước: **cắt shot** rồi **chọn keyframe trong/ngoài shot**.

## A1. Cắt shot (shot boundary detection)

Phát hiện điểm chuyển cảnh để chia video thành các đoạn quay liên tục (shot).
Code: `src/frame_sampling/shot_detector.py`. Chọn ở `config.yaml → shot_detection.detector`.

| Backend | Nguyên lý | Ghi chú |
|---|---|---|
| **autoshot** (mặc định) | mạng deep NAS (biến thể TransNetV2), xác suất chuyển cảnh > 0.5 | **cả 2 đội vô địch dùng** (U-CESE tr.7, Vortex tr.5); vượt TransNetV2 +4.2% F1 (AutoShot arxiv tr.1) |
| transnetv2 | mạng deep, dự đoán frame chuyển cảnh | phổ biến, nhưng đo thực tế ranh giới lệch |
| pyscenedetect | so histogram màu 2 frame kề > ngưỡng | nhẹ, không cần model; **bỏ sót chuyển cảnh mờ (dissolve)** |

- **Vì sao AutoShot**: kiểm chứng trên video VTV, AutoShot khớp 100% cú cắt cứng của
  PySceneDetect (thuật toán độc lập) VÀ bắt thêm được chuyển cảnh mờ — đáng tin nhất.
- **Cài đặt**: AutoShot không có package pip. Code kiến trúc lấy từ github.com/wentaozhu/AutoShot
  (`autoshot_arch.py`, `autoshot_linear.py`), weights tự tải từ HuggingFace `backseollgi/AutoShot`
  vào `models/` khi chạy lần đầu. Checkpoint bọc trong key `net`; input là `[B,C,T,H,W]` 48×27 RGB.

## A2. Chọn keyframe

Code: `src/frame_sampling/keyframe_extractor.py`. Chọn ở `config.yaml → keyframe.strategy`.

### Nhóm THEO SHOT (video đã dựng: tin tức, phóng sự)

| Chiến lược | Cách chọn | Nguồn |
|---|---|---|
| `middle` | 1 frame giữa mỗi shot | baseline VBS |
| `uniform` | lấy đều mỗi N frame trong shot | Vortex tr.5 ("every eighth frame") |
| `adaptive` | frame giữa + thêm frame khi nội dung đổi mạnh (histogram) VÀ cách ≥1s | **tự thiết kế** (chống trùng) |
| `clip_reldiff` | embedding CLIP mỗi 8 frame, giữ khi rel_diff>0.4 | Vortex tr.5 (Eq.1) |

**`adaptive` (tự thiết kế)** — greedy: hạt giống = frame giữa; thêm ứng viên chỉ khi
VỪA khác hình > ngưỡng histogram (0.35) VỪA cách mọi keyframe đã giữ ≥ 1 giây. Sửa lỗi
"3 ảnh trùng nhau trong 1 shot" do dissolve. Tối đa 3 keyframe/shot.

### Nhóm TOÀN VIDEO (video quay liên tục: CCTV, POV/kính)

| `dake` | DAKE gốc — phân tích đường cong kích thước JPEG cả video, KHÔNG cần shot detection | **U-CESE tr.5-6, Algorithm 1** |

**DAKE (Dynamic-Aware Keyframe Extraction)** — vì sao dùng cho CCTV/POV: video quay liên
tục gần như 1 shot → `adaptive` chỉ ra ~3 keyframe cho cả video. DAKE không phụ thuộc shot.

Nguyên lý (U-CESE tr.5): cảnh có chuyển động/thay đổi lớn → kích thước file JPEG biến thiên
mạnh; cảnh tĩnh → JPEG ổn định. Công thức "steepness" (độ dốc thay đổi kích thước):

```
S(i,j) = A / sqrt((j-i)² + A²),   với A = 100 × (s_j − s_i) / s_max
```
(s = kích thước JPEG, s_max = max toàn video). Thuật toán (Algorithm 1, tr.6): với mỗi frame i,
tính steepness trung bình trong cửa sổ j=i+1..i+3; sắp giảm dần; lấy top `ρ×n` (ρ=0.02, tr.10);
ép mỗi cửa sổ 2 giây có ≥1 keyframe. Code: `dake_global()`.

## A3. Caption theo SHOT hay theo KEYFRAME?

Code: `decide_caption_level()` trong `src/pipeline.py`. Config: `caption.level = shot|keyframe|auto`.

- **shot**: 1 caption/shot — hợp video đã dựng (1 shot = 1 cảnh thống nhất). Rẻ.
- **keyframe**: 1 caption/keyframe (U-CESE tr.6) — BẮT BUỘC cho CCTV/POV (cả video 1 shot →
  caption theo shot = 1 câu cho cả video, vô dụng).
- **auto**: đo **mật độ shot** = số shot / phút. `< 2 shot/phút` → keyframe (video liên tục);
  `≥ 2` → shot (video đã dựng). Ngưỡng 2 tách rõ 2 nhóm (tin tức ~10, CCTV/POV ~0.5-0.8).

---

# B. SINH CAPTION (Vision-Language Model)

Code: `src/captioning/captioner.py`. Config: `caption.backend`.

| Backend | Model | Đặc điểm |
|---|---|---|
| **qwen3-vl** (mặc định) | Qwen3-VL-4B-Instruct, nạp 4-bit | tiếng Việt CHI TIẾT, OCR mạnh, nhiều ảnh/lượt; VRAM ~3-4GB/6GB |
| qwen2.5-vl | Qwen2.5-VL-7B | đời trước, cần ~16GB |
| blip | BLIP-large | nhẹ, tiếng Anh, để test nhanh |

- **Vì sao Qwen3-VL-4B**: thế hệ mới (10/2025), bản 4B ≈ bản 7B đời trước (~92-95%) nhưng
  4-bit vừa GPU 6GB. Vortex dùng Qwen2.5-VL-3B (tr.6); ta dùng 3-VL-4B cho chi tiết hơn.
- **Nạp 4-bit** (bitsandbytes, nf4): giảm ~4B tham số từ ~8GB xuống ~2.5GB để vừa VRAM.
- **Caption theo shot** (`caption_shot`): đưa nhiều keyframe của shot vào 1 lượt → caption
  mạch lạc (ý tưởng ReCap, U-CESE tr.7).
- **Caption theo keyframe** (`caption_keyframe`): U-CESE tr.6 — đưa cửa sổ k keyframe trước/sau
  làm NGỮ CẢNH + keyframe đích, chỉ mô tả ảnh đích.
- **Prompt** (config): bắt mô tả chi tiết người (số lượng/tuổi/trang phục/hành động), bối cảnh,
  **OCR chữ trên hình**, tương tác — để sau tìm lại được bằng truy vấn.

---

# C. TỐI ƯU TỐC ĐỘ

## C1. GPU decode video (NVDEC)

Code: `src/frame_sampling/video_io.py`. **Nghẽn tìm ra khi profile**: giải mã H.264 trên CPU
chiếm 85-93% thời gian (đặc biệt shot detection đọc mọi frame).

**Cách sửa**: giải mã + thu nhỏ NGAY TRÊN GPU rồi chỉ tải ảnh nhỏ về:
```
ffmpeg -hwaccel cuda -hwaccel_output_format cuda -i video
       -vf scale_cuda=W:H,hwdownload,format=nv12 ...
```
Tự động fallback CPU nếu lỗi codec. **Kết quả**: POV 12ph 1080p shot-detect **236s → 38s (6×)**,
CCTV1 decode **124s → 21s**. Bật/tắt: `shot_detection.gpu_decode`.

## C2. Sửa BUG cv2 seek mất keyframe (nghiêm trọng)

**Phát hiện khi đo POV**: DAKE chỉ ra **27 keyframe** cho video 12 phút (đáng lẽ ~619) và mất
**604 giây**. Nguyên nhân: `cv2.CAP_PROP_POS_FRAMES` (seek lẻ tới từng frame) **đọc sai/trượt**
hầu hết frame trên video **long-GOP 1080p** → âm thầm bỏ sót 96% nội dung.

**Cách sửa**: `stream_frames()` — giải mã TUẦN TỰ qua ffmpeg pipe (không seek), yield từng frame.
Đọc đúng & đủ mọi frame. **Kết quả**: POV DAKE **604s→32.5s (18×), 27→619 keyframe (đúng)**;
testVTV DAKE 37s→7.8s. Áp cho cả khâu đo JPEG lẫn khâu đọc keyframe đã chọn.

> Bài học: `cv2` seek KHÔNG đáng tin trên video nén long-GOP. Luôn decode tuần tự khi cần độ chính xác.

## C3. DAKE đa nhân / streaming

Đo JPEG size mọi frame: giải mã streaming GPU + nén JPEG song song bằng thread (cv2.imencode
nhả GIL), giới hạn frame giữ trong RAM. Máy 20 luồng → dùng 18.

## C4. Giảm số caption: dedup nội dung + batch (đòn lớn nhất)

**Nghẽn sau khi tối ưu decode**: caption Qwen3-VL per-keyframe. POV3 có 143 keyframe ×
~9s = ~21 phút (99% tổng thời gian). Xử lý:

Code: `src/captioning/reduce.py` (`caption_keyframes_dedup`). Config: `caption.dedup`.

**(a) Dedup nội dung** (Vortex tr.5 "prioritize computational efficiency ... reduce redundancy"):
1. CLIP embed mọi keyframe (nhanh, batch GPU — embedding này **tái dùng cho Milvus**).
2. Gom cụm greedy: keyframe mở CỤM mới nếu `rel_diff = 1 − cosine > ngưỡng` so đại diện cụm
   hiện tại (Vortex Eq.1). Ngưỡng 0.10.
3. Chỉ caption ĐẠI DIỆN mỗi cụm, rồi GÁN caption đó cho mọi keyframe trong cụm.
→ Mọi keyframe vẫn có caption + vector (đủ retrieval), nhưng ít lần gọi model. POV3: 143 → ~61.

**(b) Batch inference**: gom nhiều keyframe (mỗi cái 1 ảnh + prompt) vào 1 lượt `generate`
(`caption_batch`, config `caption_batch_size=8`). GPU utilization 4%→95%. POV3: 7.5s→5.2s/ảnh.

**Vì sao KHÔNG mất chất lượng**: dedup chỉ bỏ keyframe TRÙNG NỘI DUNG (near-duplicate), keyframe
khác biệt vẫn được caption riêng chi tiết bằng Qwen. Đúng cách 2 paper làm (U-CESE dùng Gemini
API song song; Vortex dùng dedup rel_diff + model 3B).

**Kết quả POV3 (2.7 phút, 143 keyframe)**: tổng pipeline **~21 phút → 324.5s (~5.4 phút), nhanh ~4×**.
Chỉ 61 lần gọi model (thay vì 143) nhờ dedup; 143 keyframe vẫn đủ caption chi tiết tiếng Việt.
Breakdown: cắt shot 17.8s + DAKE 8.1s + (CLIP embed + dedup + batch caption 61 đại diện) ~298s.

---

# D. LƯU TRỮ & TRUY VẤN (thiết kế, tầng sau của team)

## D1. Ba kho dữ liệu (U-CESE tr.7)

| Kho | Nội dung | DB |
|---|---|---|
| VisualDB | vector ảnh keyframe (CLIP image) | Milvus |
| TextualDB (embedding) | vector caption (CLIP text) | Milvus |
| TextualDB (raw) | caption + OCR dạng chữ | Elasticsearch |

**Caption index 2 dạng** (raw + embedding): raw bắt từ khoá chính xác (tên riêng, số, OCR),
embedding hiểu nghĩa gần đúng. Nối 3 kho bằng `id` chung. Code skeleton: `src/db/indexer.py`.

## D2. Cách search (U-CESE Algorithm 2, tr.9 — "Unified Clipping")

1. Truy 3 kênh song song: vector ảnh + vector caption + text BM25.
2. Gộp, sắp theo (video, timestamp).
3. Quét 2 con trỏ gom keyframe gần nhau thành CLIP (đoạn ≤ T giây) — độ phức tạp tuyến tính.
4. Xếp hạng: clip nào phủ NHIỀU SUB-QUERY nhất thắng (hoà → similarity cao nhất).
→ Trả về **đoạn thời gian** (clip), không phải frame lẻ (vì giám khảo chấm theo timestamp).

## D3. Nâng cao (Vortex)

- **RRF fusion** (tr.7): trộn thứ hạng 2 model CLIP + SigLIP2, `score = Σ 1/(60+rank_i)`.
- **Rocchio feedback** (tr.7): user like/dislike → dịch vector truy vấn.
- **Temporal re-ranking** (tr.8): tách Q thành trước/hiện/sau, cộng điểm nếu cùng video có đủ chuỗi.

---

---

# E. HỌC HỎI TỪ CÁC HỆ THỐNG VBS (kỷ yếu MMM 2025/2026 trong `docs/`)

Đây là hệ thống Video Browser Showdown quốc tế (dataset V3C, tiếng Anh) — khác cuộc thi
HCM (U-CESE/Vortex) nhưng cùng bài toán. Chỉ tổng hợp phần liên quan captioning.

## E1. "Multi-modal LLM Video Captioning" (Cheng et al., MMM 2025 tr.301-307)
- **Caption phục vụ 2 việc, 2 model**: BLIP-2 (caption frame-level) sinh 7M caption cho 1.44M
  video WebVid để **pre-train** model search; **LLaVA-NeXT-Video-7B** (caption paragraph chi tiết:
  màu/thời gian/địa điểm) cho **text-to-caption retrieval**.
- **Kết luận (tr.318): caption paragraph chi tiết > caption frame thô** cho tìm kiếm tinh → validate
  hướng Qwen3-VL chi tiết.
- **Prompt (tr.318)**: "focusing on the main subjects, their actions, objects, the location, and the
  time of day", 8 frame — gần y hệt prompt ta dùng.
- **Text-to-caption retrieval (tr.318)**: encode caption + query bằng **Sentence Transformer**
  (all-mpnet-base-v2), cosine → KÊNH TÌM THỨ 3 (ngoài CLIP + BM25). Bản đa ngôn ngữ cho tiếng Việt.
- **Concept bank (tr.319)**: rút n-gram (1..4) từ caption, sắp theo tần suất → gợi ý user tinh chỉnh query.

## E2. VIREO (MMM 2026 tr.183-189, cùng nhóm)
- **Milvus làm CẢ dense + Sparse-BM25** trong 1 backend (tr.198) → có thể BỎ Elasticsearch riêng.
- **SSM shot segmentation** thích ứng video ngắn/dài. OCR **PaddleOCR v5**, ASR **Whisper-Turbo**.

## E3. U-CKER (MMM 2026 tr.176-182)
- Lưu 4.1M vector CLIP dạng ma trận dày `768×N` trong **GPU RAM**, inner-product **CHÍNH XÁC**
  (không ANN) để không mất độ chính xác (tr.191). Đánh đổi: cần GPU RAM lớn.

## E4. Việc nên thêm cho dự án (rút ra)
1. Kênh **text-to-caption bằng Sentence Transformer đa ngôn ngữ** (thứ đáng thêm nhất). → **ĐÃ LÀM (E5)**
2. Cân nhắc **Milvus Sparse-BM25** thay Elasticsearch (gọn còn 1 DB).
3. Tăng `frames_per_shot` (họ dùng 8) khi caption theo shot.
4. **Concept bank** n-gram từ caption cho UI gợi ý.

## E5. ĐÃ IMPLEMENT: output sẵn-sàng-index (2 loại embedding + id)

Code: `src/embedding/embed.py`, nối trong `src/pipeline.py`. Config: `embedding.enabled`.
Bám thiết kế 3 kho U-CESE (tr.7) + kênh text-to-caption MMM 2025 (Cheng tr.318).

Mỗi lần chạy, ngoài `keyframes.jsonl`/`shots.jsonl` còn xuất:
- **`clip_keyframe.npy`** [N, 512] — CLIP image embedding của keyframe → **VisualDB (Milvus)**.
- **`caption_emb.npy`** [N, 768] — caption embedding bằng **multilingual-e5-base** → **TextualDB**
  (kênh text-to-caption: mã hoá caption tiếng Việt, tìm theo NGHĨA). e5 dùng tiền tố
  `passage:`/`query:`. Đã test: query "người mặc áo đỏ" khớp đúng caption tiếng Việt.
- Thêm trường **`id`** (số dòng toàn cục) vào `keyframes.jsonl` = khoá nối Milvus↔Elasticsearch.
  Hàng thứ i của 2 file .npy khớp keyframe có `id=i`.

**CÓ CẢI THIỆN THỜI GIAN KHÔNG? → KHÔNG. Đây là cải thiện ĐỘ HOÀN THIỆN/độ chính xác, không phải tốc độ.**
Chi phí thêm (đo trên CCTV2, 67 keyframe):
- CLIP keyframe: **0s — MIỄN PHÍ** (tái dùng embedding đã tính khi dedup).
- Caption e5: **~0.025s/keyframe** (619 keyframe ≈ +15s) + load model e5 ~16s/lần chạy.
→ Tổng thêm chỉ ~15-30s/lần chạy, không đáng kể so với khâu caption (vài phút).

**Lưu ý cài đặt**: KHÔNG dùng `sentence-transformers` vì nó kéo `scikit-learn`+`scipy`, mà máy này
CHẶN DLL scipy (Application Control policy) → vỡ cả transformers. Ta gọi e5 qua `transformers`
trực tiếp (AutoModel + mean pooling), không cần scipy.

## E6. Bổ sung 4 kỹ thuật từ paper (đã implement)

1. **frames_per_shot = 8** (Cheng tr.318 dùng 8): đưa 8 keyframe/shot vào VLM để hiểu diễn biến.
   Config `caption.frames_per_shot`.

2. **OCR song ngữ Việt-Anh** — dùng CHÍNH Qwen3-VL làm OCR (không cài PaddleOCR/EasyOCR vì chúng
   kéo scipy bị chặn). Prompt yêu cầu model, sau phần mô tả, ghi `TEXT: <chữ nguyên văn trên hình>`.
   `split_caption_ocr()` tách thành 2 field `caption` + `ocr` trong keyframes.jsonl/shots.jsonl.
   Đã test đọc đúng tiếng Anh ("Queue Length Monitoring...", "www.marchnetworks.com"). Config `caption.ocr`.
   *Hạn chế*: OCR theo cùng độ chi tiết với caption (theo đại diện cụm khi dedup) → phụ đề đổi
   nhanh giữa keyframe gần trùng có thể bị bỏ. Muốn chính xác hơn: PaddleOCR v5 (VIREO tr.198)
   khi chạy trên máy không chặn scipy.

3. **ReCap — caption có bộ nhớ hồi quy** (U-CESE tr.7): caption theo shot duyệt shot theo THỨ TỰ
   thời gian, mỗi shot nhận `memory` = tóm tắt mô tả các shot TRƯỚC (cắt `recap_memory_chars`),
   giúp caption mạch lạc theo mạch truyện. Config `caption.recap_memory`. Code: vòng caption
   theo shot trong `pipeline.py` (tuần tự, cập nhật memory sau mỗi shot).

4. **SigLIP2 — embedding ảnh THỨ 2** (Vortex/VIREO): ngoài CLIP còn xuất `siglip2_keyframe.npy`
   (ViT-B-16-SigLIP2/webli, 768-dim) để tầng retrieval RRF-fuse với CLIP. Config `embedding.siglip`.

**VRAM**: embedding (CLIP/SigLIP/e5) — `embedding.device: auto`: GPU khi caption dùng API/mock
(GPU rảnh), CPU khi caption dùng Qwen local (tránh OOM 6GB).

## E7. Backend caption qua API free (Gemma/Gemini) — SONG SONG

Code: `GemmaAPICaptioner` trong `src/captioning/captioner.py`. Config `caption.backend: gemma`.
Giải quyết nghẽn GPU 6GB: gọi **Google AI Studio API** (free tier) thay chạy model local.

- Model: `gemma-4-26b-a4b-it` (đa phương thức, 140+ ngôn ngữ có tiếng Việt) hoặc `gemini-2.5-flash`.
- **Song song**: `caption_batch` bắn `api_max_workers=16` request đồng thời (đã test 16 đồng thời:
  0 rate-limit). Key đọc từ `.env` (`GEMMA_API_KEY`) — `.env` đã trong .gitignore.
- Retry backoff khi 429 (hết quota/phút).
- **Chất lượng: chi tiết HƠN Qwen local**, OCR song ngữ tốt (đọc "HAPPY HOUR", "EDEN HOTEL").
- Khi dùng API, GPU rảnh -> embedding tự chuyển sang cuda -> nhanh hơn.

**Quota free (đo/tra 2026)**: ~**1.500 request/ngày** (Gemini Flash; Gemma bị cắt 7/2026, xem
AI Studio để biết số live), RPM thoải mái (16 đồng thời không bị chặn), 1M token/phút.
→ **Đủ để TEST** (vài trăm/lần); **KHÔNG đủ cả dataset** (~250K caption = ~167 ngày) -> khi thi
thật cần trả phí Gemini Flash (~$25-90 cả bộ) hoặc chạy Qwen local/cloud.

**Kết quả POV3 (2.7 phút, 147 keyframe → 59 caption sau dedup)**:

| Cấu hình | Thời gian | Lỗi |
|---|---|---|
| gemma 10 luồng | 288.6s | 0 |
| gemma 16 luồng | 224.6s | 0 |
| gemma 20 luồng | 204.3s | 0 |
| gemma 25 luồng | 199.5s | 0 (bão hoà — hơn 20 lợi ích rất ít) |

Trần RPM 1 key ~20-25 luồng (bắn 60 → ~26 dính 429). Qwen local ~324s.

**Multi-key (nhiều tài khoản Google)**: `GemmaAPICaptioner._load_keys()` gom TẤT CẢ key từ env/.env
(biến `GEMMA_API_KEY*`, hoặc nhiều key cách nhau dấu phẩy) -> `caption_batch` phân phối request
**round-robin** ra từng key -> mỗi key chỉ gánh `api_max_workers` luồng -> tổng `16×N` luồng, vượt
trần RPM của 1 key. ⚠️ Né rate-limit bằng nhiều tài khoản là vùng xám TOS Google. Mẫu key: `.env.example`.

- **Tự loại key hỏng** (`api_validate_keys`): kiểm tra từng key lúc khởi tạo, chỉ giữ key hoạt động
  (vd key bị 403 "denied access" do tài khoản bị Google gắn cờ). In `N/M key hoạt động`.
- **Tự chọn số key theo batch** (`api_auto_keys`): dùng `min(N, ceil(batch/16))` key — batch nhỏ dùng
  ít key (tiết kiệm quota tài khoản), batch lớn dùng hết. Xoay key bắt đầu mỗi batch để trải đều tải.

**Đo tốc độ POV3 (59 caption, 16 luồng/key)**: 1 key 224s · 2 key 169s · 3 key 148s · 5 key ~150-185s
(nhiễu API; batch 59 chỉ cần ~4 key là "no"). Multi-key phát huy rõ nhất ở batch LỚN (cả dataset).

**Quota THẬT (từ AI Studio console của user, Free tier)**: rate limit tính THEO TỪNG MODEL/project.
- **Gemma 4 26B & 31B**: RPM 30, RPD **14.400/key** MỖI model — cao nhất, DUY NHẤT dùng được cho bulk.
- Gemini Flash/Flash-Lite: RPD chỉ **20-500** -> vô dụng cho caption hàng loạt.
- Live API (Native Audio Dialog, Flash Live, Live Translate): "Unlimited" nhưng là streaming real-time
  (thoại/video), KHÔNG hợp batch caption ảnh + TPM vẫn giới hạn.

**Xen kẽ 2 model (gấp đôi quota)** — `gemma_models: [26B, 31B]`. Mỗi model có RPD riêng -> 28.800 RPD/key.
Thiết kế **FAILOVER** (không phải xen kẽ mù): mỗi request dùng model ĐẦU (26B nhanh) làm chính; khi bị 429
thì nhảy sang model kế (31B - quota riêng) NGAY, chỉ sleep khi CẢ 2 model throttle. -> vừa nhanh (26B)
vừa gấp đôi quota, KHÔNG phạt tốc độ (POV3 vẫn ~181s). Với 14 key: 14×28.800 = **403.200 RPD** ->
caption CẢ dataset (~250K) trong **<1 ngày, FREE**.

*Lưu ý DAKE dùng CPU streaming decode (không GPU) cho chắc chắn đủ frame; GPU decode chỉ ở
shot detection. Gán keyframe→shot: clamp vào shot gần nhất, không drop (phòng shot phủ thiếu).*

---

*Cập nhật lần cuối: + học hỏi VBS MMM 2025/2026 (captioning). Trước đó: GPU decode, sửa bug cv2 seek,
dedup+batch caption, Qwen3-VL, AutoShot.*


---

## F. Caption theo SHOT mô tả HÀNH ĐỘNG (action-shot captioning)

### F1. Vấn đề giải quyết
Cách cũ (caption theo KEYFRAME) mô tả từng ẢNH TĨNH ("một người đứng trên vỉa hè...") →
(1) mất thông tin CHUYỂN ĐỘNG/diễn biến (người đi trái→phải, camera lia...), thứ rất
quan trọng để truy vấn theo hành động; (2) sinh RẤT NHIỀU request API (POV 12' = 311 request)
→ dính rate-limit của Gemma free (pool ~30 RPM dùng chung nhiều key) → hỏng 86%.

### F2. Cách hoạt động
**a) Trích keyframe theo shot (strategy "action")** — `keyframe_extractor.py`:
- Mỗi shot: quét ứng viên RẢI ĐỀU theo thời gian (phủ trọn diễn biến đầu→cuối cảnh).
- CHỈ GIỮ frame THỰC SỰ khác frame giữ trước đó > `action_change_threshold` (ngưỡng CAO 0.30,
  so histogram HSV). Đoạn tĩnh → 1 frame; đoạn nhiều chuyển động → nhiều frame (tới `keyframes_per_shot`=10).
- So với frame GIỮ TRƯỚC (không phải trung bình) → bám sát chuyển động tuần tự.
- Hàm: `_pick_action()` (bản cv2 per-shot, fallback) và `action_global()` (bản nhanh, xem F4).

**b) Caption theo shot, GỬI NHIỀU keyframe trong 1 request** — `captioner.py::caption_shot`:
- Gửi tối đa `frames_per_shot`=10 keyframe của shot (rải đều thời gian nếu dư) trong MỘT request.
- Dùng `shot_prompt` RIÊNG (khác prompt tả ảnh tĩnh): yêu cầu model coi các ảnh là KHUNG HÌNH
  LIÊN TIẾP THEO THỜI GIAN và mô tả DIỄN BIẾN + HƯỚNG chuyển động + trình tự hành động.
- ReCap memory (U-CESE tr.7): duyệt shot theo thứ tự thời gian, đưa mô tả shot trước làm bối cảnh.

### F3. Vì sao chọn cách này
- **Đúng bản chất bài toán**: truy vấn VBS thường theo hành động/sự kiện → cần mô tả chuyển động.
- **Thoát rate-limit**: POV 9 shot = 9 request (thay 311). Đo thực tế: 14 key Gemma free DÙNG CHUNG
  1 quota pool ~30 RPM/model (test 20+20 vào 2 key chỉ được 29/40) → càng ít request càng chắc.
- Đánh đổi: caption shot tuần tự (ReCap) ~45s/shot với 10 ảnh (nhiều token) → 9 shot ~7 phút.
  Chấp nhận được vì offline + đổi lại 100% thành công.

### F4. Tối ưu tốc độ trích keyframe (`action_global`)
Bản đầu (`_pick_action` + cv2) giải mã TOÀN BỘ video tuần tự → POV 12' mất >540s.
`action_global()` giải mã 1 LƯỢT bằng GPU streaming (NVDEC, `stream_frames`) ở res nhỏ (long-side 192),
tính histogram theo `stride`, gom theo shot rồi chọn → tái dùng `_dake_read_and_map` đọc frame đã chọn.
Fallback CPU nếu GPU thiếu frame (so `seen < 0.9*n_total`). **Đo: 540s → 30.3s (~18x).**

### F5. File/hàm
- `src/frame_sampling/keyframe_extractor.py`: `_pick_action()`, `action_global()`, nhánh "action" trong `extract_keyframes()`.
- `src/captioning/captioner.py`: `caption_shot()` (Qwen + Gemma) dùng `self.shot_prompt`; `_call(..., prompt=)`.
- `config.yaml`: `keyframe.strategy=action`, `keyframes_per_shot`, `action_change_threshold`,
  `action_min_gap_sec`; `caption.level=shot`, `caption.frames_per_shot`, `caption.shot_prompt`.

### F6. Kết quả đo được (POV.mp4, 12 phút, 9 shot)
| Chỉ số | Cách cũ (keyframe) | Cách mới (action-shot) |
|---|---|---|
| Số request API | 311 | **9** |
| Tỉ lệ caption thành công | 14% (86% dính 429) | **100% (9/9)** |
| Trích keyframe | ~65s (dake) / >540s (action cv2) | **30.3s** (action GPU) |
| Nội dung caption | tả ảnh tĩnh | **mô tả chuyển động/diễn biến** |


### F7. Tối ưu LƯU TRỮ output (chuẩn hóa, bỏ trùng lặp)
Ở chế độ caption theo shot, caption dùng CHUNG cho mọi keyframe của shot → nếu lưu theo keyframe
sẽ TRÙNG nhiều lần. Đã chuẩn hóa (`finalize_output`):
- **keyframes.jsonl** chỉ giữ metadata nhẹ `{id, video_id, shot_index, frame_index, time_sec, keyframe_path}`
  (bỏ caption/ocr/ranh-giới-shot vì đã có ở shots.jsonl, nối qua `shot_index`). Chế độ "keyframe" vẫn giữ caption riêng.
- **shots.jsonl** có `id` + caption/ocr DUY NHẤT ở cấp shot; `keyframes` chỉ là list metadata (không lặp caption).
- **caption_emb.npy** lưu 1 vector / SHOT (khớp shots.jsonl `id`) thay vì 1/keyframe (trùng) → giảm ~89%.
- **clip/siglip .npy** giữ theo keyframe (mỗi ảnh khác nhau, khớp keyframes.jsonl `id`).

Khóa liên kết: Milvus visual dùng keyframe `id`; text-to-caption dùng shot `id`; join `keyframe.shot_index → shot`.
**Đo (POV): tổng output 761 KB → 477 KB (−37%)**; keyframes.jsonl −82%, caption_emb −89%.

### F8. Gộp shot detection + trích keyframe "action" vào 1 LƯỢT decode
Profile POV.mp4 (12'): [1] AutoShot 65s + [2] action 47s = video bị GIẢI MÃ 2 LẦN. Đã thêm
`combined_autoshot_action()`: stream video 1 lần ở res ~192, mỗi frame VỪA resize 48x27 (cho AutoShot)
VỪA tính histogram theo stride (cho action), sau đó AutoShot cắt shot rồi gom histogram theo shot và
chọn keyframe (`_action_select_all`). Bật khi `detector=autoshot` + `strategy=action` + `fast_combined_decode`.
Fallback tách rời nếu decode thiếu frame. **Đo: tiết kiệm ~10s warm (nhiều hơn khi cold).**
Lưu ý: nút thắt CHÍNH của shot detection là CNN inference của AutoShot (~15-20s) + giải mã 21.500 frame
(~40s, chi phối bởi SỐ frame chứ không phải độ phân giải — thử hạ res histogram 192→96 KHÔNG nhanh hơn).
Tổng frame-processing ≈ 8s/phút-video. File: `combined_autoshot_action()`, `_action_select_all()`.

### F9. Chọn keyframe theo CHUYỂN ĐỘNG (optical flow) + Motion hint vào prompt (A+B)
**Vấn đề:** histogram màu BẤT BIẾN không gian → người đi trái→phải trên nền tĩnh không đổi histogram
→ bản cũ BỎ SÓT action. **A — chọn keyframe theo motion:** trong 1 lượt decode, tính optical flow
(Farneback) giữa các frame candidate; số keyframe theo NGÂN SÁCH motion (tĩnh→2, động→tới N), đặt
theo ARC-LENGTH của motion tích lũy → phủ cả shot, dồn vào đoạn nhiều action. **B — motion hint:**
chia shot thành vài đoạn đều thời gian, tổng hợp flow → câu tóm tắt hướng, PHÂN BIỆT camera-lia vs
chủ thể (coherence = |flow trung bình|/|độ lớn trung bình|; cao = cả khung dịch đều = camera lia,
đi NGƯỢC chiều nội dung). Chèn hint vào ĐẦU prompt caption → model bám tả đúng hành động/hướng.
- File: `_flow_vec`, `_dir_phrase`, `_action_select_motion`, `_dispatch_action` (keyframe_extractor.py);
  `Shot.motion_hint`; `caption_shots(hints=)`, `_shot_prompt_with_hint()` (captioner.py).
- Config: `keyframe_signal=motion|histogram`, `action_motion_step`, `action_still_level`, `action_hint_segments`.
- Kết quả (POV3): caption bắt đúng CHUỖI camera lia phải→trái→phải (trước chỉ 1 hướng); phân biệt
  "chủ thể đi trái" vs "camera lia phải". Tự thiết kế (dùng Farneback chuẩn của OpenCV).

### F10. [Ghi chú] Gemma API NHẬN input VIDEO trực tiếp (chưa dùng, để cân nhắc)
Đo thực tế: `gemma-4-26b-a4b-it` nhận `Part(mime_type="video/mp4")` và mô tả được hành động.
Token: video 4s = 1.209 token < 10 ảnh tĩnh = 2.609 token → RẺ HƠN + hiểu chuyển động THẬT (không đoán).
Hạn chế: shot dài gửi nguyên video → token bùng nổ (~1fps: 142s ≈ 37K > 16K TPM/key) → phải trim/giảm fps.
Hướng nếu dùng: gửi mỗi shot là clip ngắn giảm fps (cap ~15 frame), giữ keyframe cho embedding ảnh.

### F11. [ĐÃ IMPLEMENT - thử nghiệm] Caption theo VIDEO CLIP mỗi shot (fallback ảnh)
Bật bằng `caption.shot_video: true`. Mỗi shot: cắt clip [start,end] bằng ffmpeg, GIẢM fps + res và
CAP số frame (`video_max_frames`=16) -> token ≈ 16*~300 ≈ 5K < 16K TPM/key (an toàn). Gửi clip video
cho Gemma (nhận video trực tiếp). Nếu cắt clip lỗi HOẶC API trả '[lỗi]/[hết quota]' -> FALLBACK sang
caption ẢNH (frames + motion hint) cho ĐÚNG shot đó. Song song, mỗi shot 1 key (như bản ảnh).
- File: `_call_parts()` (tách từ `_call` để dùng chung ảnh/video), `_extract_clip()`,
  `caption_shot_video()`, `caption_shots_video()` (captioner.py); nhánh video trong pipeline.
- Config: `shot_video`, `video_fps`, `video_max_frames` (cap token), `video_max_side`.
- Đo (POV3, 3 shot): 3/3 dùng video, 0 fallback, caption stage 57s (vs 95s bản ảnh), token/req ~5K.
  Caption bắt đúng chuỗi camera lia phải→trái→phải + chuyển động tiến/zoom. RPM không đổi (1 req/shot),
  TPM THẤP HƠN bản ảnh (video 4s=1.2K token < 10 ảnh=2.6K).

### F12. Prompt tập trung HÀNH ĐỘNG NGƯỜI (không tả máy quay)
Yêu cầu: caption phải tả HÀNH ĐỘNG của NGƯỜI trong khung (đi/chạy/ngồi/mua bán/đẩy đồ/tương tác + hướng
di chuyển của họ), KHÔNG tả chuyển động máy quay ("camera lia trái/phải"). Đã:
- Viết lại `shot_prompt`: trọng tâm hành động người, CẤM tả camera ("dù cả khung dịch do máy quay -> BỎ QUA").
- TẮT chèn motion hint vào prompt (`inject_motion_hint: false`) vì optical flow đo chủ yếu là chuyển động
  MÁY QUAY -> trước đây làm model tả camera. Motion VẪN dùng để CHỌN keyframe (F9), chỉ không đưa vào lời tả.
- Đo (POV3, video mode): caption chuyển từ "camera lia phải/trái" sang "người đàn ông đi bộ hướng xa;
  bé gái tiến lại gần; người đạp xe về phía trước; thực khách ngồi ăn uống, trò chuyện". Đúng mục tiêu.

### F13. [SỬA LỖI] Video vượt TPM + chỉ gọi 26B
- **TPM vượt trần (bug):** clip shot dài encode fps thấp nhưng ĐỘ DÀI clip vẫn nguyên -> Gemma tự
  sample ~1fps theo độ dài -> shot 142s ≈ 43K token (VƯỢT 16K TPM). ĐO thực tế trên dashboard: spike 40K.
  **Sửa:** gắn `types.VideoMetadata(fps=sample_fps)` vào video Part (sample_fps = min(video_fps, max_frames/dur))
  -> Gemma sample đúng ≤ video_max_frames frame. Đo lại: shot 142s = 8.7K token, shot 5s = 2.3K token (đều <16K).
- **Chỉ 26B được gọi:** phân model shot dùng `(i//n)%nm` -> khi số shot ≤ số key thì i//n=0 -> luôn model 0 (26B),
  31B không bao giờ chạy. **Sửa:** đổi sang `i%nm` -> luân phiên 26B/31B (mỗi shot vẫn 1 key riêng) -> dùng CẢ 2 model.
- Kết quả (POV.mp4 9 shot): 9/9 OK, 0 fallback, tổng 269s (trước 423s), không còn spike TPM.

### F14. [TỐI ƯU] Cắt clip shot dài bằng FAST-SEEK + token/TPM guard
- **Cắt clip chậm (đo):** shot 142s cắt clip mất 50.5s vì ffmpeg `-to 142 -vf fps=...` giải mã CẢ 142s
  để lọc frame (pipeline đã decode video 1 lần rồi -> decode thừa). **Sửa:** shot dài (> video_fastseek_min_sec=30s)
  dùng FAST-SEEK: mỗi frame 1 lệnh `-ss <t> trước -i` (nhảy tức thì tới keyframe gần), ghép N frame thành mp4 1fps.
  Đo: **50.5s -> 10.8s (4.7x)**. Shot ngắn (<=30s) vẫn giải mã 1 lượt (nhanh ~2s, tránh N lần launch ffmpeg).
  Kết quả: caption stage POV 9 shot **165s -> 87s (~1.9x)**.
- **Guard token/TPM (bổ sung):** (1) chọn N frame theo `video_token_budget` (token ≈ 5000 + 300*N <= 12K);
  (2) gắn `VideoMetadata(fps)` để Gemma sample đúng N frame (nếu không -> Gemma sample 1fps theo độ dài
  clip -> shot 142s ~43K token vượt TPM); (3) `_TokenLimiter` giữ TỔNG token/phút mỗi (key,model) <= tpm_budget=15K
  -> KHÔNG bao giờ vượt TPM 16K dù nhiều shot dồn 1 key. Công thức token đo từ count_tokens API (7/16/36 frame
  = 6.3K/9.1K/14K token). Latency gen 1 shot (đo): API ~40s (ngắn/vừa) đến ~76s (dài); song song ~10s/shot khấu hao.

### F15. Lọc keyframe LƯU LẠI phải THẬT SỰ KHÁC NHAU (dedup ngưỡng lớn)
Caption lấy từ VIDEO clip, NHƯNG keyframe vẫn được LƯU (ảnh + keyframes.jsonl + embedding CLIP/SigLIP)
làm nguyên liệu cho bước sau (visual search Milvus). Thêm `dedup_keyframes()`: trong mỗi shot, 1 keyframe
chỉ giữ nếu KHÁC MỌI keyframe đã giữ > `keyframe_dedup_threshold` (histogram HSV, bất biến không gian nên
camera lia không tính là 'khác' — chỉ nội dung đổi mới tính). Chạy trong process_video SAU khi trích
keyframe, TRƯỚC caption/embed/ghi -> ảnh lưu + embedding đều là keyframe phân biệt.
- File: `dedup_keyframes()` (keyframe_extractor.py), gọi trong `process_video`.
- Config: `keyframe_dedup=true`, `keyframe_dedup_threshold=0.35` (lớn -> ít & khác rõ hơn).
- Đo (POV3): 24 -> 11 keyframe; khác biệt NHỎ NHẤT giữa 2 keyframe cùng shot = 0.36-0.38 (> 0.35) -> đảm bảo phân biệt.
  Ảnh vẫn lưu 11/11 trên đĩa; clip/siglip .npy khớp 11 keyframe, caption_emb .npy khớp 3 shot.

### F16. LOAD BALANCER key/model tập trung (sửa lỗi "dồn vào 1 key", 2026-08-03)

**Vấn đề:** cách gán key cũ tính `client = i % n`, `model = (i//n)%nm` với `i` **reset về 0 mỗi
video**. Video tin tức thường ~9 shot → luôn dùng key 0–8, **key 9–20 KHÔNG BAO GIỜ được gọi**;
key 0,1,2 nhận shot đầu của MỌI video → nóng nhất, dễ chạm 30 RPM và 429, trong khi 12/21 key ngồi
không. Chỉ `caption_batch` có xoay `_batch_i`; hai hàm caption theo shot thì không. Quota thật rất
lớn (21 key × 2 model × 30 RPM = **1.260 req/phút**, 14.400 RPD/bucket) nhưng bị lãng phí và lỗi hoài.

**Cách hoạt động (tự thiết kế):** gom mọi (key, model) thành **42 BUCKET** quota độc lập, mỗi bucket
có `_RateLimiter` (RPM) + `_TokenLimiter` (TPM) + bộ đếm **RPD** riêng. Một con trỏ **round-robin
toàn cục, BỀN qua mọi video** (`_pick`) rải từng request:
1. Bucket xếp **model-major** (key chạy nhanh nhất) → 2 bucket liền nhau luôn KHÁC KEY → mỗi request
   kế tiếp rơi vào 1 key khác (9 shot → 9 key khác nhau).
2. Ưu tiên bucket đang **RỖNG** (load=0) tính từ con trỏ; nếu mọi bucket bận thì lấy bucket **tải
   thấp nhất** (`_RateLimiter.load()` = tỉ lệ lấp đầy cửa sổ 60s).
3. Bỏ qua bucket **chết** (key 403/API_KEY_INVALID → `.kill()`) hoặc **cạn RPD** (`.exhaust_day()`,
   tự reset sau 24h).
4. Khi 429: `_call_parts` **đổi NGAY sang bucket khác (KHÁC KEY)**, không như bản cũ chỉ đổi model
   trên cùng key (key nóng vẫn nóng).

**Vì sao:** biến 21 key thành 1 pool điều phối tập trung thay vì gán cứng theo chỉ số cục bộ → tải
đều tuyệt đối, dùng hết quota, chịu lỗi (key hỏng/cạn ngày tự loại). Round-robin ưu tiên bucket rỗng
cho phân phối đều khi tải thấp (thực tế), tự chuyển least-loaded khi bão hoà.

**File/hàm:** `_Bucket`, `_RateLimiter.load()`, `GemmaAPICaptioner._pick()`, `_call_parts()`,
`_log_dist()` (captioner.py). `caption_shots/caption_shots_video/caption_batch` bỏ gán index thủ công.
**Config:** `api_rpm_per_model=30`, `api_rpd_per_model=14400`, `api_log_dist=true`; `api_auto_keys` bỏ dùng.

**Kết quả đo (offline, giả lập API, 180 request / 20 video × 9 shot):** cả **21/21 key có tải, mỗi
key 8–9 request (chênh max = 1)**, model 26B/31B = 96/84 (≈53/47). Trước đó: chỉ 9/21 key được dùng,
key 0–2 gánh nặng nhất, key 9–20 = 0. `KEY DÙNG ĐƯỢC: 21/21` (kiểm chứng gọi thật cả 2 model mỗi key).

**⚠️ SỰ CỐ RETRY STORM (đo thật L21_V001, 331 shot) & cách sửa — cooldown/backoff:**
Bản load balancer ĐẦU bỏ backoff khi 429 (chỉ `continue` bắn lại bucket khác NGAY) + đặt `rpm=30`
(không biên) + `workers=3`. Chạy thật L21_V001: **281/331 shot (85%) LỖI**, tổng **4741 request
thực cho chỉ 626 request logic (khuếch đại 7.6×)** — dù phân phối vẫn ĐỀU (21/21 key, 208–273 req/key).
- **Chẩn đoán:** request video token nặng chạm TPM/RPM ở mép cửa sổ 60s -> 429; KHÔNG backoff nên mỗi
  429 dội lại tức thì lên bucket khác -> mọi bucket cùng chạm trần -> sụp dây chuyền.
- **KIỂM CHỨNG key độc lập:** bắn 50 request đồng thời qua 21 key (mỗi req 1 key) = **0 lỗi/3.1s**
  -> key KHÔNG dùng chung pool; lỗi 100% do storm, không phải thiếu quota.
- **Sửa:** (1) mỗi bucket bị 429 -> `cool(api_cooldown_sec=20s)`, `_pick` bỏ qua bucket đang cooldown;
  (2) khi MỌI bucket nghỉ -> `_call_parts` backoff tăng dần (trần 20s) thay vì spin; (3) tách ngân sách
  "gọi API thật" (8) khỏi "chờ bucket" (30); (4) trả `rpm=25` lấy biên. File: `_Bucket.cool/available`,
  `_pick`, `_call_parts` (captioner.py). Config: `api_cooldown_sec`, `api_rpm_per_model=25`.

**⚠️ PHÁT HIỆN QUAN TRỌNG — trần MULTIMODAL free-tier THẤP (đo 2026-08-03):** sau khi sửa storm,
chạy lại L21_V001 vẫn hỏng nhiều. Đo cô lập tìm ra 429 KHÔNG do RPM/pool:
- **Key ĐỘC LẬP** (không chung pool): 50 request TEXT đồng thời qua 21 key = **0 lỗi/3.1s**.
- **429 do TOKEN multimodal**: cùng 40 request tới 1 key — **1 ảnh (269 tok) = 40/40 OK**, nhưng
  **10 ảnh (2591 tok) = 13 OK / 27 lỗi**. Token thật đo qua `usage_metadata`: 1 ảnh=269, 10 ảnh=2591,
  video 1 frame=793, video 4 frame=1678 (est code khớp: 720/2900/720/1680). Video RẺ token hơn ảnh.
- Video chạy TUẦN TỰ = 100% OK (~27s/call, độ trễ Google xử lý video), nhưng SONG SONG cao vẫn 429
  (42 luồng video-only = OK 41/60). **Trần multimodal free-tier thực sự thấp** — load balancer đã rải
  tối đa nhưng KHÔNG tạo thêm được quota. Khớp kết luận cũ (E7): free tier "KHÔNG đủ cả dataset".
- **Mitigation (đã làm):** (1) `api_max_output_tokens=300` chặn output -> TPM ổn định; (2) est cộng
  output; (3) `fallback_frames=3` (thay 10) -> fallback ảnh nhẹ token ~3.3x; (4) `api_max_inflight`
  cap luồng; (5) `tpm_budget 15K->13K` lấy biên. File: `_workers()`, `_call_parts` (config gcfg),
  `caption_shots_video` (fallback). Config: `api_max_output_tokens/api_max_inflight/fallback_frames`.
- **THỰC TẾ:** để caption ĐỦ & NHANH cả dataset vẫn phải Gemini trả phí hoặc Qwen local/cloud (E7).
  Free tier + load balancer 21 key: hợp TEST/batch nhỏ, hoặc chạy CHẬM (cap luồng) chấp nhận nhiều giờ.

**⚠️ BUG `max_output_tokens` gây caption RỖNG (đo A/B 2026-08-03) — ĐÃ SỬA:** để kiểm chứng nghi vấn
"429 pipeline không đến từ Google", chạy A/B CÙNG 30 clip: (A) gọi THÔ dồn 1 key, (B) qua `_call_parts`.
- Mức 429 A≈B (~17-20% ở 30 request đồng thời) -> 429 là THẬT từ Google (log nguyên văn: `code:429
  RESOURCE_EXHAUSTED`), KHÔNG phải pipeline bịa ra, KHÔNG phải bắt nhầm lỗi. Claim "40/40 dồn 1 key"
  KHÔNG tái hiện (dồn 1 key vẫn 429 ~20%).
- NHƯNG lộ bug MỚI: 25/30 request qua pipeline KHÔNG 429 nhưng trả `.text` RỖNG. Nguyên nhân: mình vừa
  thêm `config=GenerateContentConfig(max_output_tokens=300)` -> Gemma trả `finish=MAX_TOKENS` -> SDK cho
  `.text` rỗng. So sánh trực tiếp: `no-config` -> `finish=STOP, text=244 ký tự` (caption thật); output
  chuẩn của run TRƯỚC (không config) có 29 caption THẬT, 0 rỗng. **Sửa:** BỎ `max_output_tokens` khi gọi
  API (caption tự nhiên ~250-300 token, không cần chặn); thêm chốt: response rỗng -> coi LỖI MỀM, thử
  bucket khác, cùng lắm trả `[rỗng]` (trước đây trả `""` -> pipeline nhận nhầm là caption hợp lệ).
- Bài học: đừng set `max_output_tokens` sát mức output thật cho model không có "thinking" -> dễ MAX_TOKENS
  rỗng. `api_max_output_tokens` giờ CHỈ để cộng vào est_tokens (ước lượng TPM), không truyền cho API.

---

# G. ĐO ĐỐI CHIẾU VỚI KEYFRAME BAN TỔ CHỨC (benchmark, 2026-08-02)

## G1. Vấn đề giải quyết
BTC phát kèm bộ **keyframe mẫu** (`data/keyframes/<video_id>/001.jpg…`). Câu hỏi: pipeline
của mình sample frame **tốt hay tệ hơn** bộ đó? Trước đây không có cách trả lời bằng SỐ.

**Trở ngại:** BTC nói *"vị trí (frame index) tương ứng của mỗi keyframe được ghi trong file
metadata"* — nhưng phần **Metadata KHÔNG được tải về** (chỉ có Videos + Keyframes). Tên file
`001.jpg` chỉ là SỐ THỨ TỰ, không phải frame index → không đối chiếu trực tiếp được.

## G2. Cách hoạt động — khôi phục frame index bằng so khớp pixel
**Tự thiết kế** (không lấy từ paper nào):
1. Giải mã TUẦN TỰ toàn bộ video → mỗi frame rút thành **chữ ký 32×18 ảnh xám** (576 byte).
2. Mỗi keyframe BTC cũng rút chữ ký như vậy.
3. `argmin` khoảng cách **L1** trên 576 chiều → frame index của keyframe đó.
4. **Ràng buộc đơn điệu:** keyframe sắp theo thời gian nên index phải TĂNG DẦN. Tìm
   **dãy con tăng dài nhất (LIS)** = các khớp tin cậy; số còn lại khớp lại trong CỬA SỔ
   giữa 2 hàng xóm tin cậy → loại triệt để khớp nhầm ở cảnh lặp/gần giống.

Kết quả trên L21_V001: sai khác trung bình **1.19/255 mỗi điểm ảnh** (p95 = 1.36) → khớp
chắc chắn. 3/307 keyframe vi phạm thứ tự, sửa xong đạt **307/307 đơn điệu**.
- File: `scratchpad/locate_btc_keyframes.py`, `fix_monotonic.py` (script đo, ngoài repo).

## G3. Hai trục đo (bổ sung nhau)
- **Thời gian** — `Recall@τ`: % keyframe BTC có ≥1 keyframe của mình cách ≤ τ frame.
  τ chọn theo luật thi (TRAKE: đoạn đáp án *"thường dưới 10 frame"*).
- **Thị giác** — với mỗi ảnh keyframe BTC, lấy **cosine lớn nhất** với tập ảnh của mình
  (CLIP ViT-B-32 laion2b). Đây mới là thứ quyết định khả năng TRUY XUẤT: lệch vài frame
  nhưng nội dung đã lưu thì CLIP vẫn tìm ra.
- File: `scratchpad/eval_vs_btc.py`.

**Kiểm chứng ngược (quan trọng):** `frame_index` pipeline ghi ra có ĐÚNG không? Lấy 40
keyframe ngẫu nhiên, đọc lại đúng frame đó trong video, quét ±30 frame tìm chỗ khớp nhất
→ **80% khớp offset 0**, trung vị offset **0**, còn lại lệch ±1 (do sai số seek của cv2 trong
chính script kiểm chứng). Kết luận: frame_index đáng tin. File: `scratchpad/verify_my_index.py`.

## G4. Suy ngược QUY TẮC sinh keyframe của BTC
Đo trên L21_V001/V002/V003 (307/262/286 keyframe):

| | V001 | V002 | V003 |
|---|---|---|---|
| khoảng cách trung vị | 4.03s | 4.00s | 4.16s |
| **khoảng cách MAX** | **7.03s** | **7.00s** | **7.04s** |
| % khoảng ≤ 7s | 99.7% | 100% | 99.6% |
| % keyframe nằm trong 2 frame sau ĐẦU SHOT | 58.6% | 62.2% | 44.1% |

→ **Quy tắc BTC = "lấy frame ĐẦU mỗi shot" + "TRẦN lỗ hổng 7 giây"**. Trần 7s trùng khớp
ba lần và biểu đồ khoảng cách có đỉnh nhọn đúng tại **209–210 frame** (= 7.0s @30fps) →
đây là luật cứng, không phải ngẫu nhiên.

⚠️ **Keyframe BTC KHÔNG phải đáp án chấm điểm.** PDF vòng sơ tuyển ghi rõ dữ liệu thi chính
thức là Video, các phần khác *"chỉ nhằm hỗ trợ xây dựng giải pháp mẫu"*. Điểm chấm theo
khoảng `[s,e]` BGK gán cho từng truy vấn. Nên bám BTC là **mức sàn cho Textual KIS / Q&A**,
KHÔNG phải đích. Riêng **TRAKE thì bộ keyframe 4–7s của BTC hoàn toàn không đủ** (đoạn đáp
án < 10 frame = 0.33s) → bắt buộc phải có tầng lấy mẫu DÀY thứ hai sau khi khoanh được video.

## G5. Kết quả đo — pipeline hiện tại (trước khi sửa)
6 video L21 đầu tiên, `strategy=action`, `keyframe_dedup=0.5`:

| video | BTC kf | của mình | R@±10f | R@±30f | R@±90f | CLIP sim tb |
|---|---|---|---|---|---|---|
| L21_V001 | 307 | 379 | 67.4% | 76.9% | 87.6% | 0.899 |
| L21_V002 | 262 | 328 | 71.0% | 81.7% | 88.2% | 0.900 |
| L21_V003 | 286 | 318 | 57.7% | 74.1% | 87.8% | 0.886 |
| L21_V005 | 223 | 270 | 66.4% | 79.4% | 84.3% | 0.901 |
| L21_V006 | 257 | 315 | 66.5% | 79.0% | 87.9% | 0.901 |
| L21_V007 | 209 | 248 | 68.4% | 78.0% | 89.0% | 0.904 |

**Điểm mạnh:** chỗ nào có keyframe thì bám RẤT sát BTC — trung vị lệch **3 frame (0.10s)**.
So với uniform thuần mỗi 4s (chỉ 17.9% ở ±10 frame) thì sampler theo nội dung hơn **~4 lần**
→ phần khó (bám ranh giới shot) đang làm ĐÚNG.

## G6. LỖI TÌM RA — shot dài chỉ được 2 keyframe (đã sửa, 4 lần thử)
Phân bố khoảng cách giữa 2 keyframe liên tiếp (L21_V001): BTC max **7.03s**, của mình max
**32.0s**. Shot 11 (dài 32.2s) chỉ sinh **2** keyframe trong khi BTC đặt **6**.

**Nguyên nhân** (`_action_select_motion`): cả hai nhánh chọn số keyframe **không hề nhìn
THỜI LƯỢNG** — nhánh tĩnh luôn trả 2 keyframe (đầu+cuối) bất kể shot dài bao nhiêu, nhánh
động tính theo `total/step_thr` (lượng chuyển động). Video tin tức đầy cảnh phỏng vấn tĩnh
dài 20-30s -> mất trắng khúc giữa. Nhánh `dake` đã có `dake_min_window_sec` lo việc này.

**Cách sửa — phải qua 4 lần, ghi lại vì các lần hỏng đều có bài học:**

| lần | cách làm | kf | hở max | kết quả |
|---|---|---|---|---|
| 1 | chèn frame trong `_action_select_motion` | 1863 | 26.5s | ❌ dedup xoá sạch phần vừa chèn |
| 2 | + lối thoát trong vòng dedup (`gap > trần` thì giữ) | 2008 | 14.0s | ⚠️ chạy được nhưng hở ĐÚNG 2x trần |
| 3 | bù lại SAU dedup (2 lượt), guard `len(kept)>=2` | 1878 | 26.5s | ❌ bỏ sót shot bị gộp còn 1 keyframe |
| 4 | + lấy keyframe CUỐI SHOT làm mốc neo | **2196** | **7.0s** | ✅ đạt |

Ba bài học:
1. **Trần lỗ hổng và dedup CHỐNG NHAU.** Trần chèn frame vào shot TĨNH, mà frame cảnh tĩnh
   thì đương nhiên giống nhau -> dedup coi là trùng và xoá. Phải cho dedup một lối thoát
   theo THỜI GIAN, nếu không việc chèn là vô nghĩa (đo được: hở vẫn y nguyên 26.5s).
2. **Chặn ngay trong vòng dedup thì hở đúng 2x trần.** Frame nằm ĐÚNG mốc trần không thoả
   `gap > trần` nên bị xoá, phải tới frame kế mới thoát -> 14.0s với trần 7s. Phải bù ở
   LƯỢT RIÊNG sau dedup.
3. **Shot tĩnh dài bị dedup gộp về ĐÚNG 1 keyframe** -> không còn "cặp" nào để chèn vào giữa.
   Phải lấy keyframe cuối shot làm mốc neo thứ hai. Đây chính là loại shot cần bù nhất.

- File: `_action_select_motion()` + `dedup_keyframes()` (keyframe_extractor.py, lượt 2),
  gọi trong `process_video` (pipeline.py) có truyền `max_gap_frames`.
- Config: `keyframe.action_max_gap_sec = 7.0`.

⚠️ **Số mô phỏng ban đầu (dự đoán 514 kf / R@30f 91.2%) là SAI** — nó chèn frame vào output
đã dedup nên không thấy được là dedup sẽ xoá chúng. Bài học: mô phỏng bỏ qua một bước của
pipeline thì kết quả không dùng được, phải chạy thật.

## G6b. Thước đo đúng theo LUẬT CHẤM (PDF mục 2)
Các chỉ số `R@±30f / R@±90f` dùng lúc đầu **không phải luật chấm**. PDF quy định
`R-Score = I(v = GT_v ∧ id ∈ [s,e])` — NHỊ PHÂN, không có điểm cho "gần".

**HIT@W** = xác suất có >=1 keyframe rơi vào cửa sổ đáp án rộng W frame. Hai mô hình:
- **A** (không giả định): cửa sổ ở vị trí BẤT KỲ. Cửa sổ `[s, s+W-1]` chứa keyframe `k`
  <=> `s ∈ [k-W+1, k]` -> đúng W vị trí. HIT = |hợp các đoạn| / N.
- **B**: cửa sổ đặt giữa tại khoảnh khắc BTC đánh dấu -> trúng khi `|k-b| <= (W-1)//2`.

Mốc W lấy từ PDF: **W=10** (TRAKE: *"thường là dưới 10 frame"*), **W=11** (mọi VÍ DỤ trong
PDF đều dùng cửa sổ 11 frame: KIS `[500,510]`, TRAKE `[95,105]`...). Prose và ví dụ VÊNH nhau.

Bộ chấm đã kiểm chứng tái hiện ĐÚNG cả 3 ví dụ trong PDF (KIS 1/0, TRAKE 0.75, Final 0.74).
- File: `scratchpad/eval_pdf.py`.

**Kết quả (6 video L21, 1544 keyframe BTC):**

| phương án | n kf | W=10 (A/B) | W=11 (A/B) | W=30 (A/B) | W=150 (A/B) |
|---|---|---|---|---|---|
| BTC | 1544 | 8.4% / — | 9.2% / — | 24.9% / — | 91.8% / — |
| baseline | 1858 | 9.1% / 59.8% | 10.0% / 62.9% | 27.6% / 68.5% | 79.8% / 85.6% |
| **đã sửa** | 2196 | 10.5% / 64.6% | 11.6% / 67.9% | 31.2% / 74.0% | **92.1% / 96.2%** |

Ở cửa sổ 5s: **79.8% -> 92.1%**, BẮT KỊP BTC (91.8%). Nhưng dùng 2196 kf cho cùng độ phủ
mà BTC chỉ cần 1544 (+42%) -> phân bố còn LỆCH, dồn vào cảnh động. Đang dò tham số.

## G6c. ĐÍNH CHÍNH quan trọng — mật độ index KHÔNG phải trần điểm
Đã kết luận nhầm rằng "không keyframe nào trong `[s,e]` -> Final Score = 0". **SAI.**
PDF chỉ quy định định dạng `<video_id>, <frame_id>` và cho **tối đa 100 câu trả lời/truy vấn**;
KHÔNG bắt `frame_id` phải là keyframe đã index. Khoanh đúng shot rồi thì **rải frame_id dày
đặc lúc NỘP BÀI** là được — shot 5s = 150 frame, cửa sổ 11 frame -> chỉ cần 14 câu phủ kín.

Hệ quả (khác nhau theo dạng truy vấn):
- **KIS / Q&A**: chỉ cần khoanh đúng SHOT và xếp hạng cao. Mật độ index gần như không ràng buộc.
- **TRAKE**: đáp án là BỘ N frame cùng lúc -> phải rải TỔ HỢP. Định vị được trong ±15 frame
  -> ~3 phương án/khoảnh khắc -> `3^4 = 81 <= 100` VỪA ĐỦ. Sai số ±1.5s -> `9^4 = 6561` BẤT KHẢ THI.

=> Chỉ tiêu THỰC SỰ quan trọng cho sample frame là **P(sai số <= 15 frame)**, vì nó quyết định
TRAKE có rải tổ hợp nổi trong ngân sách 100 câu hay không. Hiện: **68.5% -> 74.0%**.

## G6d. Thời gian thực đo (RTX 4050)
Giai đoạn cắt shot + trích keyframe (1 lượt decode gộp):

| video | thời lượng | xử lý | nhanh hơn realtime |
|---|---|---|---|
| L21_V001 | 21.0 phút | 87.9s | 14.4x |
| L21_V002 | 17.6 phút | 74.0s | 14.3x |
| L21_V003 | 20.0 phút (25fps) | 70.7s | 16.9x |
| L21_V005 | 15.7 phút | 65.5s | 14.4x |

Toàn kho batch 1: **873 video = 130.7 giờ** (trung bình 9.0 phút/video, lấy từ `media-info`).
=> **~9.1 giờ** để index toàn bộ, chạy MỘT LẦN.

**Kiến trúc 2 tầng** (không cần decode dày cả kho):
- Tầng THÔ: index cả 873 video, 9.1 giờ, chạy trước.
- Tầng DÀY: chỉ decode ~5-10 video mà retrieval đã khoanh, LÚC trả lời từng truy vấn,
  mất vài chục giây. Khả thi vì PDF nói vòng sơ tuyển *"nội dung đoạn mô tả được cung cấp
  sẵn và trọn vẹn"* -> có thời gian xử lý từng truy vấn.

## G7. Còn tồn đọng
- Nhánh **histogram** `_action_select_all()` có ĐÚNG lỗi này, thậm chí nặng hơn (dòng
  `if len(kept) >= n_per: break` chặn cứng ở 10; shot tĩnh dài chỉ giữ frame đầu).
  Hiện KHÔNG hoạt động vì config để `keyframe_signal: "motion"` → chưa sửa.
- `paths.video_dir` từng trỏ `data/videos` (không tồn tại) → chạy hàng loạt chết ngay.
  Đã sửa thành `data/video` (quét đệ quy, tìm đủ 128 video).
- **`media-info` CHƯA được pipeline đọc.** Mỗi video có sẵn `title / description / keywords /
  publish_date / author` bằng TIẾNG VIỆT — text miễn phí, không cần model. Index vào
  Elasticsearch là lọc được theo chủ đề + ngày phát sóng, thu hẹp không gian tìm TRƯỚC khi
  CLIP phải làm việc. Rất đáng giá cho Textual KIS. (Kho là bản tin "60 Giây" HTV.)
- Chưa có **tầng lấy mẫu DÀY cho TRAKE** (xem G6c): cần decode cục bộ video đã khoanh, lúc
  trả lời truy vấn.

## G8. Tình trạng DATA (kiểm tra 2026-08-02, sau khi tải bổ sung)
Đã ĐỦ 5/5 thành phần cho **128 video L21–L24**, và **nhất quán 100%**: số ảnh keyframe ==
số dòng `map-keyframes` == số vector CLIP == số file object, đúng trên cả 128 video.

| nhóm | Videos | Keyframes | map/clip/media-info/objects |
|---|---|---|---|
| L21–L24 | **128** ✅ | 128 ✅ | 128 ✅ |
| L25–L30 | **0** ❌ | 446 (L26 chỉ 199/498 ⚠️) | 745 |
| tổng | 128 / 873 | 574 / 873 | 873 |

→ **Còn thiếu 745 video** (L25–L30). Kho thi thật của batch 1 là 873 video.

**Định dạng:**
- `map-keyframes/<vid>.csv`: `n, pts_time, fps, frame_idx`. `n` khớp tên ảnh (`001.jpg`),
  `frame_idx` = **vị trí frame THẬT**. ⚠️ **BẪY**: `001.jpg` chỉ là SỐ THỨ TỰ. Nộp
  `frame_id = 5` vì nó là ảnh thứ 5 (frame thật 411) → **R-Score = 0** dù tìm đúng cảnh.
- `clip-features-32/<vid>.npy`: float16, `(n, 512)`, đã chuẩn hoá L2.
- `objects/<vid>/<n>.json`: 100 box/ảnh (`detection_scores/class_entities/boxes/...`).
- ⚠️ **fps KHÔNG đồng nhất** giữa các video (L21_V003 = 25fps, còn lại 30fps) → mọi chỗ
  quy đổi giây↔frame phải đọc fps THEO TỪNG VIDEO.

**Lưu ý chiến lược:** PDF nói *"Dữ liệu thi chính thức là Video; các thành phần còn lại ...
chỉ nhằm ... hỗ trợ xây dựng giải pháp mẫu"*, và KHÔNG cam kết batch 2 sẽ kèm chúng.
→ Thiết kế phải chạy được **chỉ từ .mp4**; mọi thứ khác coi là quà. `map-keyframes` chỉ dùng
làm tập validation lúc phát triển, KHÔNG phải phụ thuộc lúc thi.


---

# H. TỐI ƯU CAPTION QUA GEMMA API (2026-08-02)

## H1. Vấn đề
Caption nửa video L21_V002 (142 shot) mất **13.4 phút** và **27.5% shot MẤT TRẮNG**
(trả về `[hết quota]`). Nhìn triệu chứng (thông lượng sụp, dao động răng cưa, ETA phình)
rất giống bị Google bóp quota.

## H2. Bốn chẩn đoán SAI trước khi tìm ra đúng (ghi lại để không lặp lại)
| # | giả thuyết | bác bỏ bằng |
|---|---|---|
| 1 | 14 key dùng chung pool TPM | key là 14 tài khoản riêng; dashboard từng key độc lập |
| 2 | Chế độ VIDEO nặng nên chậm | đo: video **62 req/phút** vs ảnh **29** (payload 44KB vs 249KB) |
| 3 | `_RateLimiter` là nút thắt chính | sửa `rpm` 3→25: py-spy cho thấy luồng chờ RPM 39%→0.7% nhưng **thông lượng KHÔNG tăng** |
| 4 | Mỗi request mất ~185s (suy từ định luật Little) | đo trực tiếp: **16 giây** |

Bài học: **suy luận từ triệu chứng luôn thua đo trực tiếp.** Công cụ quyết định là
(a) `py-spy dump` xem luồng kẹt ở dòng nào, (b) bọc `acquire()` để đếm giây chờ từng limiter,
(c) `usage_metadata.prompt_token_count` lấy token THẬT, (d) dashboard Google lúc ĐANG chạy.

## H3. Ba nguyên nhân THẬT (đều nằm trong code, không phải phía Google)

**(1) Ước lượng token sai 2–7 lần** — `_est_video_tokens()`
Đo bằng `usage_metadata.prompt_token_count`: n=3→859, n=6→1712, n=7→2452, n=15→4271, n=16→4726.
Khớp `token ≈ 300 × n_frame`, tức **hằng số nền ≈ 0**, trong khi code đặt `_VID_TOK_BASE = 5000`.
Sai nặng nhất ở shot NGẮN 2-4s (6.9×) — chiếm đa số trong video tin tức.
Hậu quả: `_TokenLimiter` tưởng mỗi request ăn 6800 token nên chỉ cho ~2 request/phút mỗi
(key,model), trong khi thực tế cho được ~8-9. Dashboard lúc đó chỉ hiện 3.95K/16K TPM.
→ Sửa: `_VID_TOK_BASE 5000→400`, `_VID_TOK_PER_FRAME 300→320` (vẫn để dư ~15%).

**(2) Backoff 8 giây < cửa sổ quota 60 giây** — `_call_parts()`
`time.sleep(min(8, 1.5 * 2**cycles))` + `range(5)` -> tổng kiên nhẫn chỉ **~14 giây**.
Gặp gai tải ngắn là mất trắng shot, dù quota KHÔNG cạn (RPD mới dùng 109/14.400).
→ Sửa: `range(8)` + `min(45, 2.0 * 2**cycles)` -> tổng kiên nhẫn ~2 phút.

**(3) Lỗi ghép key–model** — `caption_shots_video()` VÀ `caption_shots()`
`self.models[i % nm]` với `n = 14` (CHẴN): vì `i = c + 14k` mà `14k` luôn chẵn nên
`i % 2 == c % 2` → **mỗi key vĩnh viễn chỉ dùng 1 model**, chỉ 14/28 cặp được dùng.
Log vẫn in "trần ~84 request/phút" (tính 14×2×3) nên KHÔNG ai phát hiện.
→ Sửa: `self.models[(i // n) % nm]` ở CẢ HAI hàm.

## H4. Kết quả đo (nửa L21_V002, 142 shot, workers=2)
| | trước | sau |
|---|---|---|
| thời gian | 13.4 phút | **4.6 phút** |
| tốc độ | 10.6 shot/phút | **30.9 shot/phút** (2.9×) |
| mất trắng | 39/142 = **27.5%** | **2/142 = 1.4%** |
| dùng VIDEO | 83 | **139** |
| fallback ảnh | 59 | **3** |
| chờ TPM cộng dồn | 25.310s | **0s** |

## H5. `api_max_workers` — càng nhiều KHÔNG càng nhanh
| | thời gian | tốc độ | mất trắng |
|---|---|---|---|
| workers=2 (56 luồng) | **4.6 phút** | **30.9/phút** | **1.4%** |
| workers=5 (140 luồng) | 5.9 phút | 24.2/phút | **69.0%** ❌ |

workers=3 (84 luồng) cũng đã tệ: dừng ở 57% (81/142) khi đang chạy **16.2 shot/phút** (chỉ bằng
NỬA workers=2) và đã có ≥23 shot fallback, chờ RPM 1.645s (vs 217s).
→ **Vách đứng nằm giữa 2 và 3**, không phải giữa 3 và 5. Chỉ 3 request đồng thời mỗi cặp
(key,model) đã đủ vượt trần **30 RPM** của Google → 429 → cạn lần thử → `[hết quota]`.
CHỐT `api_max_workers: 2`. KHÔNG thử cao hơn.

⚠️ **BẪY ĐỌC SỐ LIỆU**: workers=5 chạy **37-42 shot/phút ở phút ĐẦU** (nhìn như đang thắng)
rồi mới sụp khi cửa sổ quota 60s lấp đầy. Mọi kết luận về thông lượng phải lấy từ
TOÀN BỘ lượt chạy, không bao giờ từ giai đoạn khởi động.

## H6. Giảm frame vì chỉ cần tả TĨNH
`video_fps: 1.5→0.5`, `video_max_frames: 16→4` (shot 4s: 6 frame → 2 frame).
Token/request giảm ~một nửa. Caption VẪN giữ được diễn biến trước-sau ("ban đầu cúi xuống…
sau đó ngẩng đầu…") và OCR vẫn hoạt động → đủ cho Textual KIS. Nếu sau này cần tả chuyển
động chi tiết (TRAKE) thì nâng lại `video_fps`.

**Vẫn nên gói VIDEO kể cả khi chỉ tả tĩnh**: 3 frame gói mp4 = 23 KB, còn 3 ảnh JPEG 512px
= ~125 KB → video rẻ hơn về đường truyền dù nội dung như nhau (đo: 62 vs 29 req/phút).

## H7. Còn tồn đọng
- Mỗi shot fallback gọi API **2 lần** (`caption_shot_video` rồi `_call`), và mỗi lần thất bại
  đã đốt tới 8 lượt xin limiter. Nên giảm fallback quan trọng hơn giảm thời gian mỗi request.
- `_call_parts` **nuốt thông báo lỗi** (cắt còn 60 ký tự, fallback im lặng) → không biết 429
  hay 503 hay timeout. Nên log nguyên văn ít nhất 1 lần mỗi loại lỗi.
- Chưa đo end-to-end CẢ video (sample frame + caption + embedding) sau khi sửa.

### F17. [SỬA] Retry-storm là thủ phạm 429 (không phải hết quota) — giảm retry + backoff
**Nghịch lý:** dashboard Google (account doquockien12345) cho thấy RPM 14/30, TPM 9.68K/16K, RPD
238/14.4K — ĐỀU DƯỚI TRẦN, mà pipeline vẫn 429 tràn (79-85% shot lỗi). Đo cô lập:
- **Tuần tự 8 video/1 key = 8/8 OK**; 8 đồng thời/1 key (lúc quota nghỉ) = 8/8 OK → quota KHÔNG cạn.
- 429 chỉ bùng ở RUN LỚN. Nguyên nhân: `_call_parts` cũ retry tới **8 lần**/request → 623 shot logic
  nở thành **4643 API call (×7.5)**. Nhiều shot cùng 429 → dội lại TỨC THÌ → nhân request/phút →
  chạm ngưỡng ẩn (burst per-account) → 429 dây chuyền → lại retry → xoáy ốc.
**Sửa:** (1) `api_max_retries=3` (8→3); (2) BACKOFF trước mỗi retry (`sleep(min(10, 2*lần))`) thay vì
bắn lại ngay; (3) `api_max_inflight=12` (ít burst). **Đo (60 shot): khuếch đại 7.5×→2.2×, thành công
24%→42%.** Còn lại là throttle cửa-sổ-ngắn per-account (tuần tự=100%, càng giãn nhịp càng OK) + quota
bị vắt trong buổi test. File: `_call_parts` (captioner.py). Config: `api_max_retries`, `api_max_inflight`.

### F18. Theo dõi 429 THEO ACCOUNT
Map 21 key ↔ 21 email (`data/key_accounts.json`, `_load_key_emails`). Mỗi bucket đếm `n429`; `_log_dist`
in bảng xếp hạng account hay dính 429 nhất. Dùng để phát hiện account bị Google gắn cờ (sẽ ~100% riêng lẻ).

### F19. Per-account in-flight + phát hiện QUOTA-FRESHNESS chi phối
Thêm `api_per_account_inflight` (mặc định 1): giới hạn số request ĐỒNG THỜI trên mỗi account (email)
qua `_acct_inflight` counter + reserve trong `_pick`, release trong `_call_parts` (finally). =1 ép giống
"tuần tự 100% OK" nhưng 21 account chạy song song → chống burst per-account.
**Sweep (30 shot, cap 3→2→1, nghỉ 45s) BỊ NHIỄU:** cap=3 (chạy đầu, quota tươi) = 87%/1.3x khuếch đại;
cap=2 (giữa)=23%; cap=1 (cuối, quota kiệt)=10%. → **thứ tự chi phối, KHÔNG so được cap.** Bài học:
**độ tươi quota là biến chi phối thật** (fresh 87% ↔ kiệt 10%), át mọi config. Phải test trên quota TƯƠI,
KHÔNG chạy nhiều lần back-to-back. Kết quả TỐT NHẤT cả buổi: **87% + khuếch đại 1.3×** (fix retry hoạt động).
Muốn chốt cap chính xác: chạy 1 lần/mức trên quota tươi (cách nhau nhiều giờ). File: `_pick`, `_release_account`,
`_call_parts` (captioner.py). Config: `api_per_account_inflight`.


---

# I. CẮT SHOT THÀNH CHUỖI SỰ KIỆN CÓ THỨ TỰ (nguyên liệu TRAKE) — 2026-08-05

## I1. Vấn đề giải quyết

**Shot là đơn vị DỰNG PHIM, không phải đơn vị HÀNH ĐỘNG.** Đọc kỹ ví dụ TRAKE của BTC
(PDF `Thong tin vong So tuyen AIC2026.pdf` tr.2) thấy rõ: một cú "nhảy cao" gồm 4 khoảnh khắc
(chạy đà → giậm nhảy → bay qua xà → tiếp đất) thường nằm **trong CÙNG một shot quay liên tục**.
Caption cấp shot viết "vận động viên thực hiện cú nhảy" là đúng nội dung nhưng **không định vị
nổi từng khoảnh khắc**, trong khi cửa sổ chấm của TRAKE *"thường là dưới 10 frame"* (tr.4) —
ở 30fps là **1/3 giây**.

Hai giai đoạn TRAKE (tr.1) cần hai thứ khác nhau:
- **Giai đoạn 1 – Truy xuất** (tìm đúng video): caption/CLIP lo được. Sai video = **0 điểm ngay**.
- **Giai đoạn 2 – Căn chỉnh** (chốt frame mỗi khoảnh khắc): caption **KHÔNG** lo được.

→ Module này sinh ra tầng dữ liệu còn thiếu: **`events.jsonl`** — mỗi shot được cắt thành các
ĐOẠN HÀNH ĐỘNG có thứ tự, mỗi đoạn kèm frame THẬT (`start/end/anchor_frame`).

## I2. Cách hoạt động — hai tín hiệu cộng lại

### Tín hiệu 1: PA (PredictAbility)
**NGUỒN:** Shou et al., *"Generic Event Boundary Detection: A Benchmark for Event Segmentation"*,
ICCV 2021, arXiv:2101.10511 — **mục 5.2**, baseline KHÔNG GIÁM SÁT tốt nhất trong bài.
Link: https://arxiv.org/pdf/2101.10511

Ý tưởng gốc từ tâm lý học tri giác: con người cảm nhận ranh giới sự kiện ở chỗ **khó đoán trước**
nhất. Công thức:

```
PA(t) = || mean(feat[t-w : t])  −  mean(feat[t+1 : t+1+w]) ||²
```

Khoảng cách LỚN ⇒ "trước" và "sau" khác nhau nhiều ⇒ t khó đoán ⇒ nhiều khả năng là ranh giới.
Bài báo chốt ranh giới ở **cực đại địa phương** sau khi lọc LoG. Bài báo cũng nêu rõ PA **vượt
hẳn phương pháp cắt shot (PySceneDetect)** ở bài toán này — đúng lý do ta cần nó: **ranh giới
SỰ KIỆN mịn hơn ranh giới SHOT**.

Cài đặt ở đây dùng tổng tích luỹ (prefix sum) → `O(m·d)` thay vì `O(m·w·d)`.

### Tín hiệu 2: TURN — điểm ngoặt động học (**TỰ THIẾT KẾ**, không lấy từ paper)

PA nhìn **NGOẠI HÌNH**. Nhưng đọc lại 4 khoảnh khắc BTC định nghĩa (tr.2):

> "khoảnh khắc **đầu tiên** bàn chân **chạm đất**" · "chân **rời hoàn toàn** khỏi mặt đất"
> · "hông ở vị trí **cao nhất**" · "lưng **bắt đầu chạm** đệm"

**Cả 4 đều là ĐIỂM TỚI HẠN CỦA CHUYỂN ĐỘNG** (tiếp xúc / rời tiếp xúc / cực trị), **không phải
thay đổi ngoại hình** — ngoại hình trước và sau gần như y hệt. PA sẽ mù ở đúng những chỗ này.
Nên phải có tín hiệu thứ hai đo trực tiếp động học:

```
TURN(t) = |mean(mag sau) − mean(mag trước)|                    ← dừng lại / bật lên / va chạm
        + (1 − cos(vector trước, vector sau))/2 · min(mag)      ← đảo chiều / quay đầu / nảy lên
```

Nhân với `min(mag_trước, mag_sau)` để **cảnh tĩnh không sinh ranh giới giả** — khi gần như đứng
yên thì hướng của flow là nhiễu ngẫu nhiên, cos dao động mạnh, nếu không nhân sẽ nổ khắp nơi.

### Trộn và chọn ranh giới
```
score(t) = a · PA_chuẩn_hoá(t)  +  (1−a) · TURN_chuẩn_hoá(t)      (a = event_pa_weight, mặc định 0.5)
```
Làm trơn Gauss (σ = `event_smooth_sigma`) → lấy cực đại địa phương vượt `event_boundary_threshold`
→ sắp theo độ mạnh, giữ dần, ép khoảng cách tối thiểu `event_min_sec`, trần `event_max_per_shot`.

**KHÔNG dùng scipy** (máy này bị chặn DLL scipy — xem E5): kernel Gauss + convolve bằng numpy.

### ANCHOR — frame ứng viên nộp cho TRAKE
- Đoạn **đầu tiên**: chưa có điểm ngoặt mở đầu → lấy **giữa đoạn**.
- Đoạn **thứ 2 trở đi**: lấy **ĐÚNG frame ranh giới**. Lý do: khoảnh khắc TRAKE là *thời điểm
  CHUYỂN TRẠNG THÁI*, không phải giữa pha.

### Gom shot → SCENE
Chuỗi TRAKE có thể trải qua nhiều shot liền nhau (bản tin cắt qua lại 2–3 góc máy của cùng sự
kiện). Nếu chỉ khớp chuỗi trong phạm vi 1 shot thì không bao giờ khớp đủ N khoảnh khắc.
`group_scenes` gom shot liền kề có ảnh xám trung bình giống nhau (RMS < `event_scene_threshold`),
trần `event_scene_max_sec`. `event_ord` trong `events.jsonl` đánh **theo SCENE** — đây mới là
chuỗi mà TRAKE cần khớp. **Đây là HEURISTIC tự thiết kế, KHÔNG phải từ paper**: nó gom được
"cùng bối cảnh", KHÔNG hiểu được "cùng câu chuyện". Muốn chuẩn hơn phải dùng ranh giới câu của
ASR — chưa có.

## I3. Vì sao chọn cách này

| Phương án | Ưu | Nhược | Kết luận |
|---|---|---|---|
| **PA + TURN (đã chọn)** | training-free; **tái dùng optical flow ĐÃ TÍNH** ở bước chọn keyframe → chi phí thêm ~0; chạy được trên mọi video chỉ từ `.mp4` | ranh giới ±`stride` frame; không hiểu ngữ nghĩa | dùng làm tầng THÔ toàn kho |
| Chỉ PA (đúng paper) | có nguồn, đơn giản | **mù ở đúng 4 loại khoảnh khắc TRAKE** (ngoại hình không đổi) | không đủ |
| Chỉ TURN | bắt đúng động học | mù khi chủ thể đổi mà không chuyển động (cắt cảnh mềm) | không đủ |
| Model GEBD có giám sát (PC, DDM-Net) | F1 cao hơn nhiều | cần train + weights + 1 lượt suy luận cả kho (130.7h) | quá đắt cho tầng thô |
| Pose keypoint (RTMPose/MediaPipe) | "hông cao nhất", "chân rời đất" thành **cực trị đo trực tiếp**, chính xác tới từng frame | chỉ đúng khi đề thiên về động tác cơ thể; đắt | **bước nâng cấp tự nhiên** khi biết đề TRAKE thật |

## I4. ⚠️ Giới hạn ĐỘ PHÂN GIẢI — điều quan trọng nhất phải nhớ

Ứng viên lấy theo `action_candidate_stride` (mặc định **5 frame**). Nên ranh giới thô có sai số
**±5 frame ở 30fps**, trong khi cửa sổ TRAKE **< 10 frame**. Sai số đó **ăn gần hết cửa sổ**.

→ Có `refine_boundaries()` (tầng DÀY): giải mã lại vùng lân cận mỗi ranh giới ở **stride 1**,
tính lại TURN, chốt cực đại chính xác. **CỐ Ý KHÔNG bật khi quét cả kho** (quá đắt) — chỉ chạy
cho vài video mà retrieval đã khoanh, lúc trả lời truy vấn. Đúng kiến trúc 2 tầng ở G6c.

## I5. File/hàm

| Việc | File/hàm |
|---|---|
| PA, TURN, cắt sự kiện | `src/frame_sampling/event_segmenter.py`: `_pa_curve`, `_turn_curve`, `segment_shot_events` |
| Gom scene | `event_segmenter.group_scenes` |
| Tinh chỉnh tới từng frame | `event_segmenter.refine_boundaries` |
| Xuất record | `event_segmenter.events_to_records` |
| Giữ lại flow (khỏi tính lại) | `keyframe_extractor._action_select_motion` → trả thêm `per_by_shot` |
| Nối vào luồng | `keyframe_extractor._dispatch_action` → gắn `shot.events`, `shot.scene_id` |
| Dựng `events.jsonl` | `pipeline.build_event_records`, `pipeline.finalize_output` |
| Gán nhãn hành động (VLM) | `pipeline.label_event_actions`, `captioner.label_events_batch` |
| Config | `config.yaml` → `keyframe.event_*`, `caption.event_labels` |

## I6. GÁN NHÃN bằng VLM — nguyên tắc "cắt bằng tín hiệu, gán nhãn bằng model"

**KHÔNG BAO GIỜ để VLM tự khai mốc giây.** Config gửi `video_fps: 0.5`, `video_max_frames: 4` →
model chỉ thấy 4 ảnh rời rạc; mọi con số giây nó viết ra là **BỊA**.

Cách neo an toàn nhất đã chọn: gửi **ĐÚNG 1 ẢNH ĐẠI DIỆN cho MỖI sự kiện, theo thứ tự thời gian**
→ ảnh thứ i ứng với sự kiện thứ i, không còn chỗ cho model nhầm chỉ số. Ảnh đại diện = keyframe
gần `anchor_frame` nhất. Gom theo SHOT nên **số request = số shot, KHÔNG nở theo số sự kiện**.

`_parse_event_lines` chịu được model lệch định dạng (bỏ số, thêm mở bài, dùng `.` thay `|`) và
**trả rỗng khi gặp lỗi API** (`[hết quota]`) để không nhiễm dữ liệu rác vào `events.jsonl`.

**MẶC ĐỊNH TẮT** (`caption.event_labels: false`): đây là request VLM PHỤ THÊM ngoài caption shot,
mà quota đang là nút cổ chai. `events.jsonl` **vẫn đầy đủ về cấu trúc** khi tắt — chỉ trống
trường `action`, nối `shot_index` sang `shots.jsonl` là có caption.

## I7. BUG tìm ra khi verify (ghi lại để không lặp lại)

Test tổng hợp (`scratchpad/test_events.py` mục [2]/[3]) bắt được: **`per[0]` là phần tử ĐỆM**
(không có frame trước nó để đo flow), **không phải phép đo thật**. Ban đầu nó được tính vào cửa
sổ "trước" → **mọi shot đều sinh một ranh giới GIẢ ngay đầu**: cửa sổ trước = [đệm 0] còn cửa sổ
sau = chuyển động thật → chênh lệch độ lớn tối đa.

Triệu chứng: đỉnh TURN rơi vào `t=1` thay vì đúng chỗ đảo chiều `t=11`; ranh giới sự kiện lệch về
frame **1005** thay vì **1100**. Sửa: `a0 = max(1, t-w)` trong `_turn_curve`.

**Bài học:** tín hiệu tổng hợp có đáp án biết trước phát hiện được lỗi này trong vài giây; chạy
trên video thật thì **không thể biết** vì không có ground truth ranh giới. Đây là lý do phải test
bằng tín hiệu dựng sẵn TRƯỚC khi chạy video thật.

## I7-b. BUG THỨ HAI — chuẩn hoá khuếch đại NHIỄU thành "hành động" (nặng hơn I7)

Test tổng hợp KHÔNG bắt được lỗi này (dữ liệu test tĩnh là ĐÚNG BẰNG 0, còn video thật thì tĩnh
= 0 + nhiễu cảm biến). Phải chạy video thật rồi **đối chiếu số cắt với mức chuyển động** mới lộ ra.

**Triệu chứng (L21_V001, 331 shot):**

| mức chuyển động của shot | số shot | TB sự kiện/shot |
|---|---|---|
| **0.0–0.2 (gần như TĨNH)** | 143 | **3.76** ⚠️ |
| 0.2–0.5 (rất nhẹ) | 84 | 3.15 |
| 0.5–1.0 (nhẹ) | 66 | 3.20 |
| 1.0–2.0 (vừa) | 31 | 3.68 |
| >2.0 (mạnh) | 7 | 2.86 |

**Shot ĐỨNG YÊN bị cắt NHIỀU HƠN shot chuyển động mạnh.** Vô lý — thuật toán đang *bịa ra hành
động ở đúng chỗ không có hành động nào*. 43% số shot (người dẫn chương trình trong trường quay)
bị băm thành ~4 "sự kiện" giả.

**Nguyên nhân:** `_norm01` kéo giãn **MỌI** đường cong ra trọn `[0,1]`, kể cả đường cong mà biên
độ thật chỉ là nhiễu. Sau chuẩn hoá thì nhiễu trông y hệt tín hiệu → vượt ngưỡng 0.25 như thường.

**Sửa:** thay bằng `_norm_floor(y, floor)` — chia cho `max(biên_độ_thật, sàn)` thay vì chia cho
biên độ thật. Shot dưới sàn ra giá trị nhỏ → không vượt ngưỡng; shot trên sàn hành xử y như cũ.
Liên tục, không phải cổng cứng bật/tắt.

**Sàn ĐO THẬT** (`scratchpad/measure_floor.py`, 4 phút đầu L21_V001, 54 shot):

| nhóm shot | biên độ TURN (p50 / p90) | biên độ PA/d (p50 / p90) |
|---|---|---|
| TĨNH (mag<0.2) | 0.038 / **0.152** | 0.0010 / **0.0073** |
| rất nhẹ (0.2–0.5) | 0.277 / 0.519 | 0.0163 / 0.0262 |
| vừa (1.0–2.0) | **3.120** / 3.392 | **0.0872** / 0.1490 |
| mạnh (>2.0) | 2.002 / 3.191 | 0.0954 / 0.1545 |

Tách bạch ~20 lần giữa TĨNH và VỪA/MẠNH → chọn `event_turn_floor = 0.8`, `event_pa_floor = 0.03`
(≈ 4–5× p90 của nhóm tĩnh, vẫn nhỏ hơn nhiều so với p50 nhóm động).

Đồng thời đổi `_pa_curve` **chia cho d** (số chiều = 1296) để giá trị không phụ thuộc kích thước
ảnh → ngưỡng sàn giữ nguyên ý nghĩa nếu sau này đổi độ phân giải. Chia cho hằng số không đổi vị
trí cực đại nên thuật toán không đổi.

**Bài học:** *chuẩn hoá min-max luôn tạo ra tín hiệu, kể cả khi không có tín hiệu nào.* Bất cứ chỗ
nào dùng `(y-min)/(max-min)` rồi so ngưỡng đều phải có **sàn tuyệt đối**, và sàn đó phải **ĐO**
trên dữ liệu thật chứ không đoán.

## I7-c. BUG THỨ BA — `event_min_sec` bị quên ở 2 mép shot
`min_gap` chỉ ép khoảng cách **GIỮA các ranh giới với nhau**, không ép khoảng cách tới **đầu/cuối
shot** → đoạn đầu và đoạn cuối có thể ngắn hơn `event_min_sec`. Đo: 69/1147 sự kiện vi phạm, trong
đó **64 nằm đúng ở 2 mép** (28 đầu + 36 cuối), chỉ 5 ở giữa. Sửa: lọc ứng viên về
`min_gap <= t <= m-1-min_gap`. Test hồi quy: `scripts/test_event_segmenter.py` mục [4c].

## I7-d. BUG THỨ TƯ — làm tròn XUỐNG ở `min_gap`
Sau khi sửa I7-c vẫn còn 12 đoạn dài 0.333–0.367s < `event_min_sec = 0.4`. Nguyên nhân:
`round(0.4 × 30 / 5) = round(2.4) = 2` mẫu = 10 frame = **0.333s**. `event_min_sec` là mức **TỐI
THIỂU** nên sai số phải lệch LÊN. Sửa: `math.ceil` thay `round` → 3 mẫu = 0.5s.

Còn lại 4/588 đoạn ngắn hơn 0.4s là **shot vốn đã ngắn** (shot 0.17s) — cả shot = 1 sự kiện,
không thể dài hơn chính nó. Đây là hợp lệ, `check_events.py` đã phân biệt 2 trường hợp này.

## I7-e. KẾT QUẢ ĐO SAU KHI SỬA (L21_V001, 21 phút, 331 shot)

| | trước (bug) | sau khi sửa |
|---|---|---|
| số sự kiện | 1147 | **588** |
| shot bị cắt >1 sự kiện | 323/331 (97.6%) | **133/331 (40.2%)** |
| trung vị thời lượng sự kiện | 0.83s | **1.33s** |
| đoạn ngắn hơn `event_min_sec` | 69 (6.0%) | **4 (0.7%, đều là shot ngắn hợp lệ)** |

**Kiểm tra quyết định** — số sự kiện/shot phải TĂNG theo mức chuyển động:

| mức chuyển động | n shot | TB sự kiện/shot (trước) | TB sự kiện/shot (sau) |
|---|---|---|---|
| 0.0–0.2 (gần như TĨNH) | 145 | 3.76 ⚠️ | **1.04** ✅ |
| 0.2–0.5 (rất nhẹ) | 85 | 3.15 | **1.72** |
| 0.5–1.0 (nhẹ) | 65 | 3.20 | **2.69** |
| 1.0–2.0 (vừa) | 29 | 3.68 | **3.31** |
| >2.0 (mạnh) | 7 | 2.86 | 2.86 |

Đã **đơn điệu đúng chiều**. (Nhóm >2.0 chỉ có 7 shot — mẫu quá nhỏ để kết luận, và chuyển động
rất mạnh thường là camera lia nhanh = MỘT hành động liên tục nên ít điểm ngoặt là hợp lý.)

**Chi phí thêm: ~0** — tái dùng optical flow đã tính ở bước chọn keyframe, không decode thêm lượt.

## I7-f. Công cụ kiểm tra (chạy được mọi lúc)
- `python scripts/test_event_segmenter.py` — 26 test trên **tín hiệu tổng hợp có đáp án biết
  trước** (PA/TURN đúng chỗ, thứ tự, phủ kín, frame thật, fps 25 vs 30, parser nhãn, 2 test hồi quy
  cho I7-b và I7-c/d).
- `python scripts/check_events.py data/output` — 9 kiểm tra tính đúng đắn trên `events.jsonl`
  THẬT (frame thật không phải số thứ tự, anchor trong đoạn, thứ tự, không hở/chồng, event_ord liên
  tục, fps hợp lệ & khớp time, keyframe trong đoạn, nối được sang `shots.jsonl`, đoạn ngắn bất
  thường) + thống kê phân bố.
- `python scripts/measure_event_floor.py` — đo lại biên độ PA/TURN để hiệu chỉnh 2 ngưỡng sàn
  nếu đổi kho dữ liệu.

## I8. Định dạng `events.jsonl`

```json
{"id": 0, "video_id": "L21_V001", "scene_id": 3, "shot_index": 12,
 "event_ord": 1,          // thứ tự trong SCENE  <- chuỗi TRAKE khớp theo trường này
 "ord_in_shot": 1,        // thứ tự trong SHOT
 "start_frame": 1420, "end_frame": 1494, "anchor_frame": 1420,   // frame THẬT
 "start_time": 47.333, "end_time": 49.833, "anchor_time": 47.333,
 "fps": 30.0,             // fps THEO TỪNG VIDEO (kho có cả 25 và 30fps)
 "duration": 2.5, "boundary_score": 0.61,
 "motion": "chủ thể/vật di chuyển sang phải (vừa)", "mag_mean": 1.12,
 "keyframe_frames": [1420, 1455, 1490],   // keyframe THẬT rơi vào đoạn (sau dedup)
 "action": ""}            // nhãn ngữ nghĩa; rỗng khi caption.event_labels=false
```

**Cố ý KHÔNG chép `caption` của shot vào từng sự kiện** — một shot 4 sự kiện sẽ nhân 4 lần
caption, đúng cái lỗi phình file đã sửa ở F7. Nối bằng `shot_index` khi cần.

`events.jsonl` **luôn phủ 100% thời lượng video, không có lỗ**: shot nào không cắt được (nhánh
histogram, `event_segmentation: false`, hoặc shot quá ngắn) thì hạ về **1 sự kiện phủ trọn shot**.

## I9. Tham khảo các hệ thống ĐÃ ĐẠT GIẢI (đọc để định hướng, chưa implement)

- **DANTE** — *"Integrated Semantic and Temporal Alignment for Interactive Video Retrieval"*,
  arXiv:2512.13169. Hệ AIC HCMC 2025, **Outstanding ở TRAKE** (Excellent Textual KIS).
  Điểm mấu chốt: **"reconstructs the full chronology of events"** và **enforcing coherent
  ordering** — cải thiện cả KIS chứ không riêng TRAKE. Xác nhận hướng "chuỗi sự kiện CÓ THỨ TỰ"
  là đúng, và ràng buộc THỨ TỰ là thứ sinh điểm (chính là `event_ord` + khớp đơn điệu).
- **MERVIN** — *"A Unified Framework for Multimodal Event Retrieval in Vietnamese News Videos"*,
  arXiv:2605.16120. **Đúng loại kho của ta** (tin tức tiếng Việt), 79/88 điểm vòng loại AIC 2025.
- **MADTempo** — arXiv:2512.12929, multi-event temporal retrieval + query augmentation.
- **U-CESE** — arXiv:2605.23274 (đã dùng: DAKE + ReCap).
- **UBoCo** — arXiv:2111.14799, GEBD bằng ma trận tự tương đồng thời gian (TSM) + kernel ranh
  giới, chạy được **không giám sát**. Phương án thay/bổ sung cho PA nếu cần mạnh hơn.

## I10. Còn tồn đọng

- **Chưa đo được độ chính xác ranh giới trên dữ liệu THẬT** — không có ground truth khoảnh khắc.
  Chỉ mới xác nhận đúng trên tín hiệu tổng hợp. Muốn đo thật phải tự gán nhãn tay vài chục
  chuỗi, hoặc chờ bộ truy vấn TRAKE mẫu của BTC.
- ~~`refine_boundaries()` đã viết nhưng chưa đo~~ → **ĐÃ ĐO** (mục J4).
- ~~Chưa có hàm khớp chuỗi~~ → **ĐÃ LÀM** (`src/retrieval/trake_match.py`, mục J).
- `event_boundary_threshold`/`event_pa_weight` **đang để mặc định theo trực giác, CHƯA dò tham số**.


# ============================================================================
# J. KHỚP CHUỖI cho TRAKE — tầng truy xuất (2026-08-05)
# ============================================================================

Đây là mảnh CUỐI của bài toán TRAKE và là "điểm ăn tiền" (nhiều đội không xử lý được). Bổ
sung cho tầng dữ liệu `events.jsonl` (mục I) một **hàm khớp N khoảnh khắc CÓ THỨ TỰ của truy
vấn vào chuỗi sự kiện của video** — chính là mảnh I10 ghi là còn thiếu.

## J1. Vấn đề & khảo sát 3 hệ ĐÃ ĐẠT GIẢI (đọc paper thật)

TRAKE (PDF sơ tuyển tr.1-2, tr.4): truy vấn = 1 hành động gồm **N khoảnh khắc CÓ THỨ TỰ**;
với mỗi khoảnh khắc j nộp ĐÚNG 1 frame ∈ cửa sổ `[s_j, e_j]` ("thường < 10 frame"); điểm =
(1/N)·Σ I(frame_j ∈ [s_j,e_j]). Khảo sát cho thấy **cả 3 đội mạnh AIC-HCMC 2025 đều giải bằng
một BỘ CĂN CHỈNH CHUỖI hậu xử lý trên ma trận tương đồng, KHÔNG dùng model mới**:

| Hệ | Nguồn | Cách khớp chuỗi TRAKE |
|---|---|---|
| **DANTE** | arXiv:2512.13169 (HCMUS/BK-VNU, *Outstanding on TRAKE*) | QHĐ trái→phải có **phạt thứ tự λ** + backtracking, O(N·T) |
| **U-CESE** | arXiv:2605.23274 | two-pointer sweep dưới trần độ dài clip, xếp theo ĐỘ PHỦ pha |
| **EEIoT** | arXiv:2512.06334 | pivot + tối đa điểm dưới ràng buộc **cửa sổ 10 frame** |
| MADTempo | arXiv:2512.12929 | neo 2 đầu (E₁, Eₙ) + beam-search ép `k₁<…<kₙ` |

Chọn **DANTE** làm lõi vì nó đúng dạng TRAKE nhất (mỗi pha đúng 1 frame, KHÔNG bỏ pha) và
đơn giản nhất (~30 dòng numpy). Drop-DTW (arXiv:2108.11996) mạnh hơn khi cần BỎ pha/nhiễu
mạnh — ghi làm phương án dự phòng, chưa cần.

## J2. Cách hoạt động — QHĐ bảo toàn thứ tự (DANTE)

Ma trận `S[i,t] = sim(mô_tả_khoảnh_khắc_i, ứng_viên_t)`, ứng viên đã SẮP theo thời gian.
Công thức DANTE (arXiv:2512.13169, Eq.2):

    DP[i,t] = S[i,t] + max_{τ : pos[t]−pos[τ] ≥ min_gap} ( DP[i−1,τ] − λ·(pos[t]−pos[τ]) )

- **Thứ tự** ép bằng chính miền `τ` đứng TRƯỚC t (chỉ số cột không giảm) → không đảo pha.
- **λ** (phạt mềm) trừ theo độ dời thời gian; tổng phạt telescope = λ·(nhịp cuối − nhịp đầu)
  = λ·ĐỘ TRẢI → kéo chuỗi lại gần nhau (một hành động liền mạch). λ=0 = chỉ ép thứ tự.
- **O(N·T)** nhờ giữ TIỀN TỐ CỰC ĐẠI của hàng trước bằng con trỏ 2 đầu; mẹo tách `λ·pos[t]`
  (hằng theo t) khỏi `−λ·pos[τ]` (gộp vào hàng trước) giữ nguyên O(N·T) kể cả khi có λ.
  (Đối chiếu: DANTE Algorithm 1 dùng đúng "running_max" này.)
- Backtracking dựng lại frame cho từng khoảnh khắc.

Quanh lõi có: **fusion đa mô thức** (CLIP ảnh-chữ + SigLIP + e5 chữ-chữ) bằng tổng-trọng-số
chuẩn-hoá-hạng hoặc **RRF** (Vortex fuse CLIP+SigLIP bằng RRF, arXiv:2606.19682); **snap_to_apex**
ghim frame về đỉnh optical-flow (apex spotting, arXiv:2012.11307); **build_submission** rải
frame_id dưới **ngân sách 100 đáp án** (PDF mục 2) — với N pha, mỗi pha rải `k=⌊100^(1/N)⌋`
frame bước ≤ W (điều kiện phủ: bước ≤ W thì cửa sổ rộng W chắc chắn chứa ≥1 đáp án).

## J3. Vì sao QHĐ chứ không tham lam (đo trên L21_V001)

Verify tự-giám-sát (`scripts/eval_trake.py`): coi rep-keyframe mỗi sự kiện là "mô tả lý
tưởng", query lại vào scene, đo khớp chuỗi có trả đúng frame + đúng thứ tự không.

| chỉ số (CLIP / SigLIP / RRF — GẦN NHƯ Y HỆT) | kết quả |
|---|---|
| DP giữ ĐÚNG THỨ TỰ | **27/27 chuỗi (100%)** |
| Tham lam (argmax từng hàng) ĐẢO thứ tự | **9/27 chuỗi (33%)** → nộp sai |
| DP khôi phục đúng frame (pha có rep riêng) | **84.6%** (55/65) |
| trong cửa sổ W=10 / W=30 | 84.6% / 84.6% |
| bền khi query bị nhiễu 40% (trộn hàng xóm) | 77% trong W=10 |

**Kết luận & phát hiện (trung thực):**
- Ràng buộc thứ tự là chỗ ĂN ĐIỂM: tham lam đảo thứ tự ở **1/3 số chuỗi** (frame pha-sau đứng
  TRƯỚC pha-trước → chắc chắn ngoài cửa sổ). QHĐ loại 100% lỗi này. Thuật toán đã kiểm
  `DP == brute-force` trên 200 ma trận ngẫu nhiên (`scripts/test_trake_match.py`).
- 15% trượt là do **khung THẬT SỰ giống nhau** (bản tin lặp bàn dẫn/MC): CLIP=SigLIP=RRF y hệt
  → đổi model thị giác KHÔNG cứu. Đòn bẩy: (1) BẬT `caption.event_labels` → thêm kênh e5 chữ-chữ
  phân biệt bằng NGỮ NGHĨA; (2) embed CHÍNH `anchor_frame` thay vì mượn keyframe hàng xóm.
- **Phát hiện dữ liệu:** 588 sự kiện > 555 keyframe → một số sự kiện KHÔNG có keyframe riêng
  (event mịn hơn keyframe), phải mượn rep hàng xóm → giới hạn phân biệt. Đây là lý do cần
  con trỏ `rep_keyframe_id` + tầng dày.

## J4. Tầng DÀY — `refine_boundaries()` (ĐÃ ĐO, `scripts/measure_dense_layer.py`)

Kiến trúc 2 tầng (G6c): tầng thô index cả kho ở stride 5; tầng dày giải mã lại vùng quanh
anchor ở stride 1 rồi ghim về đỉnh optical-flow, CHỈ chạy cho ~5-10 video đã khoanh lúc trả lời.

Đo trên L21_V001 (60 anchor ranh giới, ±0.3s):
- **Thời gian:** 115 ms/anchor → ~54s cho 10 video/truy vấn (chấp nhận cho online).
- **Độ dời:** trung vị **6 frame**, cân bằng 2 hướng (30 âm/28 dương → không lệch hệ thống),
  sát-mép giảm khi tăng bán kính (9→15: 27%→13% → apex là THẬT, không phải artifact mép).
- Trung thực: cơ chế `_turn_curve` đã kiểm trên tín hiệu tổng hợp CÓ đáp án; CHƯA có GT thật
  nên chỉ kết luận "tầng thô lệch ~6f so với đỉnh flow — đáng kể với cửa sổ <10f — nên bật
  tầng dày cho video đã khoanh", KHÔNG khẳng định đó là apex TRAKE 'đúng'.

## J5. File / hàm

| việc | chỗ |
|---|---|
| Lõi QHĐ khớp chuỗi | `src/retrieval/trake_match.py`: `align_sequence` |
| Fusion đa mô thức | `trake_match.fuse_scores` (weighted / RRF), `cosine_sim_matrix` |
| Ghim apex chuyển động | `trake_match.snap_to_apex` |
| Rải đáp án theo ngân sách | `trake_match.build_submission` |
| Đầu-cuối (chữ→frame) | `trake_match.localize_phases` |
| Con trỏ event→embedding | `pipeline.finalize_output` → `events.jsonl`: `rep_frame`, `rep_keyframe_id` |
| **Gán nhãn action từng pha (CÁCH 2)** | `caption.event_labels: true`; chạy riêng: `scripts/label_events.py` |
| Test thuật toán (đáp án biết trước) | `scripts/test_trake_match.py` (20 test, có DP==brute-force) |
| Verify dữ liệu thật | `scripts/eval_trake.py` |
| Đo tầng dày | `scripts/measure_dense_layer.py` |

## J6. CÁCH 2 — gán nhãn ACTION từng pha (ĐÃ BẬT + ĐÃ ĐO)

**Phân công lao động** (khác hẳn "caption kể chuyện": nhồi 8 frame ép VLM tường thuật cả shot):
- **"KHI NÀO"** (start/end/anchor_frame từng pha) ← `event_segmentation` (optical flow) cắt, chính
  xác tới frame. Caption kể chuyện KHÔNG cho ra được frame này.
- **"LÀM GÌ"** (nhãn ngữ nghĩa từng pha) ← VLM gán, **1 ảnh đại diện/sự kiện**, ngắn (<25 từ),
  gom theo shot nên số request = số shot (không nở theo số sự kiện). `_EVENT_PROMPT` cấm ghi
  giây/tả camera; `_parse_event_lines` chịu model lệch định dạng + trả rỗng khi lỗi API (không
  nhiễm rác vào `events.jsonl`).

**Đo trên L21_V001 (subset 20 shot nhiều pha, `scripts/label_events.py`):**
- Chất lượng nhãn ĐẠT — phân biệt PHA rõ ở shot có hành động thật:
  - shot 18: *"nhắm mắt"* → *"mở mắt và đang nói"*
  - shot 23 (3 pha): *"đứng dậy nhìn xuống"* → *"quay mặt sang phải nói chuyện với người phụ nữ áo
    hồng"* → *"đang nói chuyện khi người phụ nữ đứng phía sau"* (chuỗi pha + tương tác + trang phục).
  - Lệnh cấm camera GIỮ được: trường `motion` (flow) có "camera lia" nhưng `action` (VLM) thì không.
- **Độ phủ:** chỉ **~57%/lượt** (33/58) do throttle Gemma multimodal → `label_events.py` có **RESUME**
  (bỏ shot đã đủ nhãn), chạy lại nhiều lượt lấp dần như `caption_waves`; hoặc dùng **Qwen local**
  (không throttle) để phủ 100%.
- Nhãn này nạp vào **kênh e5 chữ-chữ** của `localize_phases` → phân biệt 2 khung GIỐNG NHAU bằng
  NGỮ NGHĨA hành động (đòn bẩy cho 15% trượt ở J3, nơi CLIP/SigLIP bó tay).

## J7. Còn tồn đọng (TRAKE)

- Độ phủ nhãn action phụ thuộc quota — chạy nốt bằng waves hoặc Qwen local.
- Chưa dò `λ`, trọng số fusion (đang mặc định trực giác) — cần bộ truy vấn TRAKE mẫu để dò.
- Chưa đo LẠI `localize_phases` khi CÓ kênh e5 (cần nhãn phủ đủ trước).
- Drop-DTW (bỏ pha/nhiễu) chưa cần, ghi để dành.

# ============================================================================
# K. BỘ ĐO KIS TỰ CHẤM — 100 mẫu ground truth từ keyframe BTC (2026-08-11)
# ============================================================================

## K1. Vấn đề giải quyết

Trước mục này, **mọi lựa chọn trong hệ retrieval đều là đoán**: không dò được `w` của RRF,
không biết ASR hay CLIP mạnh hơn trên kho này, không biết đổi SigLIP2 có đáng không. Lý do:
BTC **chưa phát bộ truy vấn mẫu** nào, nên không có gì để chấm.

Mục này dựng bộ đo KIS 100 câu **tự tạo từ chính dữ liệu BTC gửi**, kèm bộ chấm hiện thực
đúng công thức thể lệ. Từ đây mọi thay đổi đều có số liệu trước/sau.

## K2. Tiêu chí chấm CHÍNH THỨC của BTC (đọc `docs/Thong tin vong So tuyen AIC2026.pdf`)

Nguồn: *Thông tin vòng Sơ tuyển AIC 2026*, 6 trang. Không phải paper — là thể lệ BTC.

**R-Score cho KIS** (trang 3, mục 2.1.1) — **NHỊ PHÂN**, không có điểm một phần:

```
R-Score(rᵢ) = I( vᵢ = GTᵥ  ∧  idᵢ ∈ [s, e] )
```

**Final Score** (trang 4-5, mục 2.2) — mỗi truy vấn nộp **tối đa 100 câu**:

```
R@k         = max{ R-Score(r₁) ... R-Score(r_k) },   k ∈ {1, 5, 20, 50, 100}
Final Score = (1/5) · Σ R@k
```

Chữ **max** (không phải tổng) là mấu chốt. Điểm chỉ phụ thuộc **hạng của câu đúng ĐẦU TIÊN**:

| hạng câu đúng đầu tiên | 1 | 2–5 | 6–20 | 21–50 | 51–100 | trượt |
|---|---|---|---|---|---|---|
| **Final Score** | 1.00 | 0.80 | 0.60 | 0.40 | 0.20 | 0 |

Ba hệ quả trực tiếp cho bộ rải đáp án (§Q6a của `PIPELINE.md`):

1. **Nộp trùng là vứt đi.** 100 câu đúng hết cũng bằng đúng 1 câu ở hạng 1. Mọi frame nộp
   thêm trong CÙNG một shot đều vô giá trị → khử trùng theo **shot**, không theo frame.
2. **Luôn nộp đủ 100.** Bậc 51–100 vẫn được 0.2, chi phí bằng 0. Bỏ trống là tự mất điểm.
3. **Chỉ 4 ngưỡng có ý nghĩa: 1, 5, 20, 50.** Đẩy câu đúng từ hạng 6 lên 5 được +0.2; từ
   hạng 20 lên 6 được **+0.0**. Tinh chỉnh rerank trong cùng một bậc là công cốc.

## K3. ⚠️ Bề rộng `[s, e]` — BTC KHÔNG công bố con số cố định

Đây là ẩn số lớn nhất và phải thừa nhận thẳng, không được đoán bừa rồi tối ưu theo:

- **TRAKE** — thể lệ nói rõ (trang 4): *"đoạn ứng với khoảnh khắc ngữ nghĩa này thường rất
  ngắn, thông thường là dưới 10 frame"*; ví dụ dùng `[95,105]`, `[145,155]` → cỡ **±5 frame**.
- **KIS** — chỉ có ví dụ (trang 3): `L01_V001, [500, 510]` → **11 frame**.
- **Q&A** — chỉ có ví dụ (trang 3): `L05_V005, [800, 900]` → **101 frame**.

Hai ví dụ KIS và Q&A lệch nhau **10 lần** → không suy ra được quy tắc. Thêm nữa, phần mô tả
nhiệm vụ KIS (trang 1) nói *"chỉ ra một khung hình **bất kỳ** thuộc đoạn video đó"*, nghe như
`[s,e]` là cả đoạn sự kiện — **mâu thuẫn** với ví dụ 11 frame. Thể lệ mơ hồ ở chỗ này.

→ **Cách xử lý: không cược vào một mức.** Ground truth ghi ra 4 mức song song, bộ chấm báo cáo
cả 4 cột. Nếu hệ chỉ ăn điểm ở `kf_span` mà rớt ở `tol5` thì biết ngay là thiếu tầng định vị dày.

| mức | bề rộng | ý nghĩa |
|---|---|---|
| `tol5` | ±5 frame | sát ví dụ KIS trong thể lệ — khắt khe nhất |
| `tol30` | ±30 frame | ~1 giây ở 30fps |
| `tol90` | ±90 frame | ~3 giây, cỡ khoảng cách 2 keyframe BTC |
| `kf_span` | nửa đường tới keyframe trước/sau | cách hiểu "frame bất kỳ thuộc đoạn đó" |

**Hệ quả nghiêm trọng cho TRAKE (số đo, không phải cảm tính):** keyframe BTC cách nhau
~90–170 frame (đọc `map-keyframes`: `frame_idx = 0, 90, 261, 351` ở fps=30, tức 3–5.7 giây).
Cửa sổ TRAKE là ~10 frame. **Xác suất một keyframe BTC rơi trúng cửa sổ TRAKE ≈ 10/90 ≈ 11%.**
Nộp thẳng vị trí keyframe BTC cho TRAKE gần như chắc chắn trượt → **bắt buộc** giải mã video ở
fps gốc quanh vùng ứng viên (đúng việc `refine_boundaries()` ở §J4 đang làm).

## K4. Cách tạo 100 mẫu — LÀM NGƯỢC CHIỀU

Tự thiết kế (không lấy từ paper). Ý tưởng: viết truy vấn trước rồi đi tìm đáp án trong 873
video là mò kim đáy bể. Làm ngược lại thì ground truth **có sẵn theo cách xây dựng**:

1. Chọn ngẫu nhiên 100 video (seed cố định), **mỗi video tối đa 1 mẫu** → hai truy vấn không
   bao giờ đụng nhau, giảm ca "hệ trả về đoạn KHÁC cũng đúng nhưng bị tính sai".
2. Trong mỗi video, bốc ngẫu nhiên 1 keyframe ở khoảng **10%–90%** → né intro/outro/logo.
3. **Lọc ảnh xấu**: bỏ frame có `std < 28` hoặc `mean` ngoài `[25, 235]` (đen, trắng, fade) —
   ảnh một màu thì không viết nổi truy vấn phân biệt được. Đo thực tế: loại 5 ảnh.
4. Tra `map-keyframes/<video>.csv` → `frame_idx` thật + `fps` → sinh 4 khoảng `[s,e]` ở §K3.
5. Đưa **ẢNH** cho một VLM viết truy vấn tiếng Việt theo văn phong ví dụ của BTC.

**Kết quả thực tế (2026-08-11):** 100/100 mẫu, 100 video khác nhau, 0 truy vấn trùng,
độ dài trung bình 35 từ. Nguồn: **30 câu sinh qua API** + **70 câu VLM đọc ảnh trực tiếp**.

Vì sao chia hai nguồn: chạy Gemma free tier được 30 câu thì **cạn quota** (429
`RESOURCE_EXHAUSTED` liên tục, tốc độ rơi xuống ~1 mẫu/4 phút → 70 mẫu còn lại mất ~5 giờ).
Hai chỉnh sửa rút ra:
- `api_max_retries: 3` trong `config.yaml` là **quá ít** cho batch bắn dồn — 3 lần 429 liên
  tiếp là bỏ cuộc, trả `[hết quota]` dù key vẫn sống. Đã thêm cờ `--retries/--cooldown`.
- Với bộ ĐO thì rớt mẫu = **thủng bộ đo**, khác caption theo shot (rớt vài shot vẫn chấp
  nhận được). Nên đường sinh truy vấn phải kiên nhẫn hơn hẳn đường caption.

70 câu còn lại do VLM **đọc thẳng ảnh** (7 lô × ~10 ảnh) rồi nạp qua
`scripts/add_kis_queries.py` — không tốn quota, chất lượng cao hơn, và tự chấm `distinct`
sát thực tế hơn (VLM nhìn thấy 3 tập *Món Ngon Mỗi Ngày* dùng CHUNG trường quay nên hạ
điểm phân biệt của cả 3, việc mà model chấm từng ảnh riêng lẻ không làm được).

**Điểm chống thổi phồng điểm — quan trọng nhất cả mục:** truy vấn sinh từ **ảnh**, TUYỆT ĐỐI
không cho model nhìn caption/ASR/metadata. Nếu sinh từ caption, model dùng lại đúng từ ngữ
trong caption → kênh caption ăn điểm giả, đo xong tưởng hệ mạnh nhưng thi thật rớt.
Prompt còn cấm: nhắc logo đài (cảnh nào cũng có → vô ích), chép nguyên văn chữ chạy màn hình
(biến bài toán thành tra OCR), dùng từ "ảnh/khung hình/keyframe", bịa tên riêng/ngày tháng.

**Tự chấm độ phân biệt.** Kho này là bản tin thời sự: hàng nghìn cảnh "người áo trắng trả lời
phỏng vấn" giống hệt nhau. Truy vấn chung chung làm bộ chấm tự động **vô nghĩa** — hệ trả về
một đoạn khác cũng đúng nhưng bị tính sai. Nên model tự chấm `distinct` 1–5 ngay trong cùng
một lần gọi (0 token thêm); mẫu `distinct < 3` bị đánh dấu `needs_review` và **mặc định bị
loại** khỏi phép chấm.

Phân bố đo được trên 100 mẫu: `{1: 2, 2: 3, 3: 11, 4: 48, 5: 36}` → **5 mẫu bị loại**, còn
**95 mẫu dùng chấm được**. Năm mẫu bị loại và lý do — đều là thứ bộ lọc ảnh KHÔNG bắt được:

| mẫu | vấn đề |
|---|---|
| `kis_071` | **frame HỎNG** — lỗi giải mã, toàn ảnh là vệt nhiễu ngang. `std`/`mean` vẫn "bình thường" nên lọt lưới |
| `kis_057` | mặt nước loá nắng, không có vật thể nào |
| `kis_077` | suối chảy qua đá, cảnh thiên nhiên thuần tuý |
| `kis_040` | cận cảnh bếp gas trống — tập nấu ăn nào cũng có |
| `kis_088` | trường quay *Món Ngon Mỗi Ngày*, hai người đứng chào — **trùng khít** nhiều tập khác |

Hai nhóm nguyên nhân đáng ghi nhớ: (a) **frame hỏng / thuần texture** — cần thêm bộ lọc
gradient hoặc phát hiện nhiễu, không chỉ `std`; (b) **trường quay lặp lại** — kho có nguyên
một series game show/nấu ăn dùng chung bối cảnh, đây là nguồn nhiễu lớn nhất cho KIS.

## K5. Vì sao chọn cách này

| Phương án | Đánh đổi |
|---|---|
| Chờ BTC phát truy vấn mẫu | Miễn phí nhưng **không biết bao giờ có** → đứng hình vô thời hạn |
| Viết tay 100 truy vấn | Chuẩn nhất, không lệch — nhưng ~4–6 giờ người, và vẫn phải tự dò đáp án |
| **VLM nhìn ảnh (đang dùng)** | ~1 giờ; **rẻ nhất để có số đo đầu tiên** |

Nhược điểm phải thừa nhận: truy vấn VLM sinh **dễ hơn** truy vấn giám khảo (nó tả đúng thứ
trong ảnh, dùng từ phổ thông; giám khảo viết trừu tượng hơn, hay dùng từ mà kênh thị giác
không bắt được). → Trường `source` tách `gemma` / `vlm` / `manual`, bộ chấm **báo cáo số
riêng cho từng nguồn**. Cần bổ sung 20–30 câu viết tay làm đối chứng trước khi tin con số
tuyệt đối.

Với 100 mẫu, sai số quanh mức 60% ≈ **±10%**. Đủ phân biệt hệ A/B khi chênh ≥10 điểm phần trăm,
**không** đủ để tin một thay đổi làm tăng 2–3%.

## K6. File / hàm

| File | Việc |
|---|---|
| `scripts/make_kis_gt.py` | Lấy mẫu 100 keyframe, lọc ảnh xấu, sinh 4 khoảng `[s,e]` → `data/eval/kis_gt.json` |
| `scripts/gen_kis_queries.py` | Gemma nhìn ảnh → truy vấn + `distinct`; JSONL ghi tăng dần (resume) |
| `scripts/add_kis_queries.py` | Nạp truy vấn viết sẵn (VLM đọc ảnh, hoặc **người viết tay**) vào bộ |
| `scripts/eval_kis.py` | `r_score()`, `final_score()` — đúng công thức thể lệ; chấm 4 mức song song |
| `data/eval/kis_gt.json` | Bộ ground truth 100 mẫu (đầu ra chính) |
| `data/eval/kis_queries.jsonl` | Kết quả sinh thô (để chạy lại không mất việc) |

Quy trình chạy lại từ đầu:

```bash
python -u scripts/make_kis_gt.py                                    # 100 mẫu, seed 42
python -u scripts/gen_kis_queries.py --only-missing > logs/gen.log 2>&1
python -u scripts/add_kis_queries.py batch.json --merge             # nạp câu viết tay
python -u scripts/eval_kis.py --sub ket_qua_he_thong.json > logs/eval.log 2>&1
```

Định dạng file nộp cho `eval_kis.py` — mỗi truy vấn tối đa 100 câu, xếp theo thứ hạng:

```json
{ "kis_001": [["L21_V001", 1500], ["L21_V003", 220], "..."],
  "kis_002": ["..."] }
```

## K7. Kiểm chứng bộ chấm (ĐÃ CHẠY, khớp 100%)

Không tin công thức tự gõ — đối chiếu **từng ví dụ nguyên văn trong thể lệ**:

- Bảng hạng → Final Score (1→1.00, 2→0.80, 6→0.60, 21→0.40, 51→0.20, trượt→0): **khớp cả 10 ca**.
- 3 ví dụ R-Score trang 3 (`[L01_V001,505]`→1, `[L01_V001,600]`→0, `[L02_V003,505]`→0): **khớp**.
- Câu đúng ở hạng 101 → 0 điểm (thể lệ cắt ở 100 câu): **khớp**.
- Ví dụ điểm lẻ trang 5 (r₁=0.5, r₃=0.8, r₁₅=0.6 → **0.74**): **khớp**.

Đã chạy thử đầu-cuối bằng một file nộp giả (cắm câu đúng ở hạng ngẫu nhiên, lệch frame ngẫu
nhiên): bốn mức dung sai tách ra đúng thứ tự kỳ vọng `tol5 < kf_span < tol30 < tol90`,
chứng tỏ cả bốn cột đều sống và phân biệt được độ chính xác định vị.

⚠️ Một điểm yếu của `kf_span`: khi hai keyframe BTC nằm sát nhau, khoảng này co lại còn
**1–2 frame** (đo được: `kis_057` có `kf_span = [1649, 1650]`). Ở các mẫu đó `kf_span`
**khắt khe hơn cả `tol5`**. Đừng đọc `kf_span` như "mức dễ nhất" — nó không đơn điệu.

## K8. Còn tồn đọng

- **Chưa có 20–30 câu viết tay** làm đối chứng — chưa tin được con số tuyệt đối, chỉ dùng để
  so A/B. Nạp bằng `add_kis_queries.py` với `source: "manual"`.
- Bề rộng `[s,e]` thật vẫn là ẩn số (§K3) — nên hỏi BTC nếu có kênh hỏi đáp.
- **Phân bố mẫu lệch**: L26 chiếm 35/100 (kho có nhiều video L26 nhất), L22/L23 chỉ 3 mẫu.
  Chưa phân tầng theo bộ sưu tập — nếu cần cân bằng thì phải sửa `make_kis_gt.py`.
- **Bộ lọc ảnh xấu còn thủng**: `std`/`mean` không bắt được frame lỗi giải mã (`kis_071`).
  Nên thêm kiểm tra gradient/nhiễu ngang.
- Chưa có bộ đo tương ứng cho **Q&A** và **TRAKE** (TRAKE khó hơn hẳn: cần ground truth
  khoảnh khắc, mà chấm tay ở mức <10 frame rất tốn công).
- Chưa lọc ca "truy vấn trúng nhiều đoạn" bằng máy — hiện chỉ dựa vào `distinct` do model tự
  chấm, chưa đối chiếu chéo toàn kho. Nguồn nhiễu lớn nhất đã lộ diện: các series dùng
  **chung một trường quay** (xem bảng ở §K4).
