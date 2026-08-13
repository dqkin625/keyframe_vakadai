# PIPELINE ĐỀ XUẤT — AIC 2026 Vòng sơ tuyển

> Viết 2026-08-04, sau khi đọc `docs/Thong tin vong So tuyen AIC2026.pdf` và khảo sát
> toàn bộ `data/`. Đây là tài liệu **thiết kế**, chưa phải mô tả code đã chạy.
> Mọi phương pháp khi được implement thật phải ghi vào `docs/METHODS.md` theo RULE.
>
> Số liệu trong tài liệu này chia 2 loại và luôn ghi rõ:
> **[ĐO]** = đo thật trên máy này · **[ƯỚC]** = suy ra/ngoại suy, chưa đo.

---

## 0. TÓM TẮT KẾT LUẬN

Ba điều quan trọng nhất rút ra được:

1. **Kho dữ liệu là BẢN TIN TRUYỀN HÌNH ("60 Giây" — HTV).** Nghĩa là mỗi video có
   *lời thuyết minh tiếng Việt liên tục* và *chữ chạy trên màn hình* mô tả **đúng sự kiện
   mà giám khảo sẽ hỏi**, **bằng đúng ngôn ngữ của truy vấn**. Pipeline hiện tại **chưa dùng
   cả hai**. Đây là nguồn tín hiệu rẻ nhất và mạnh nhất đang bị bỏ trống.

2. **Caption VLM cho MỌI shot đang nằm trên đường găng, và nó không đáng nằm ở đó.**
   Batch 1 ≈ **118.000 shot** [ƯỚC]. Với nhịp đo thật (142 shot / 13.4 phút [ĐO], H4) thì
   riêng caption đã là ~195 giờ chạy liên tục và đang mất trắng hàng chục %. Trong khi đó
   **độ phủ toàn kho không đến từ caption — nó đến từ CLIP**, thứ đã có sẵn miễn phí cho cả
   873 video. Caption phải bị hạ cấp từ "bắt buộc toàn kho" xuống "công cụ re-rank có chọn lọc".

3. **Điểm được sinh ra ở tầng TRUY VẤN, mà tầng đó gần như chưa tồn tại.**
   `src/db/indexer.py` mới là skeleton 131 dòng; không có module search / re-rank / sinh đáp án.
   Luật chấm `R@k = max` thưởng cực nặng cho vị trí 1 và 5 → **bộ xếp hạng và bộ rải đáp án
   quyết định điểm nhiều hơn việc index dày thêm vài %**.

---

## 1. ĐỌC LUẬT — 6 điều ràng buộc thiết kế

Nguồn: PDF mục 1–2.

### 1.1. Ba dạng truy vấn, ba đường ra khác nhau

| Dạng | Nộp gì | Điều kiện đúng | Chốt chặn |
|---|---|---|---|
| **Textual KIS** | `video_id, frame_id` | đúng video **và** `frame_id ∈ [s,e]` | định vị shot |
| **Q&A** | `video_id, frame_id, answer` | như trên **và** answer khớp **ngữ nghĩa** | thêm tầng VQA |
| **TRAKE** | `video_id, frame_id₁..frame_idₙ` | **sai video → 0 ngay**; đúng video → điểm = tỉ lệ khoảnh khắc trúng | căn chỉnh thời gian **rất mịn** |

### 1.2. `Final Score = ⅕ Σ R@k`, `k ∈ {1,5,20,50,100}`, `R@k = max` trong k đáp án đầu

Quy đổi ra giá trị biên của một đáp án đúng theo vị trí nó nằm:

| Đáp án đúng đầu tiên nằm ở hạng | Final Score |
|---|---|
| 1 | **1.00** |
| 2–5 | 0.80 |
| 6–20 | 0.60 |
| 21–50 | 0.40 |
| 51–100 | 0.20 |
| >100 | 0 |

→ **Hệ quả 1:** nhảy từ hạng 6 lên hạng 1 đáng giá **+0.4 điểm** — bằng đúng việc tìm thêm
được một truy vấn mới từ con số 0 lên 0.4. **Re-rank chất lượng cao ở top-20 là khoản đầu tư
lãi nhất trong toàn hệ thống.**

→ **Hệ quả 2:** cái đuôi 21→100 vẫn còn 0.2–0.4 điểm. **Không bao giờ nộp thiếu 100 đáp án.**
Đáp án thứ 100 là miễn phí, không có phạt sai.

### 1.3. `frame_id` KHÔNG bắt buộc phải là keyframe đã index

Đã kết luận ở `METHODS.md` G6c và tôi xác nhận lại từ PDF: luật chỉ quy định định dạng
`<video_id>, <frame_id>`, không hề bắt frame đó phải nằm trong tập keyframe đã trích.

→ Với **KIS/Q&A**: khoanh đúng shot rồi thì **rải frame_id dày lúc NỘP** là phủ được cửa sổ đáp án.
Mật độ index gần như không phải trần điểm.
→ Với **TRAKE**: đáp án là **một BỘ N frame nộp cùng lúc**, nên không rải bừa được (xem 1.5).

### 1.4. Cửa sổ đáp án hẹp, và PDF tự mâu thuẫn về độ rộng

- TRAKE nói rõ: *"đoạn ứng với khoảnh khắc ngữ nghĩa này thường rất ngắn, thông thường là
  dưới 10 frame"*.
- Mọi ví dụ trong PDF đều dùng cửa sổ **11 frame** (`[500,510]`, `[95,105]`, `[145,155]`...).
- Với KIS/Q&A, PDF nói cửa sổ theo *"cùng nguyên tắc"* nhưng không nêu con số.

→ **Thiết kế theo giả định bi quan**: cửa sổ ~10 frame. Nếu thực tế cửa sổ rộng hơn (cả sự
kiện, vài giây) thì phương án bi quan vẫn trúng — nó bao hàm. Ngược lại thì không.
→ Ở 30fps, 10 frame = **1/3 giây**. Một shot 5s = 150 frame cần **~14 đáp án cách đều 10 frame**
mới phủ kín. **100 đáp án ≈ phủ trọn 7 shot.** Đây là ngân sách thật, phải lập kế hoạch.

### 1.5. TRAKE là bài toán tổ hợp, không phải bài toán rải

N khoảnh khắc, mỗi khoảnh khắc cần một frame trúng cửa sổ ~10 frame. Nếu mỗi khoảnh khắc ta
định vị được với sai số ±15 frame → cần ~3 phương án/khoảnh khắc → `3⁴ = 81 ≤ 100` **vừa đủ**.
Nếu sai số ±1.5 giây → `9⁴ = 6561` **bất khả thi**.

→ **Chỉ tiêu sống còn của TRAKE là `P(sai số ≤ 15 frame)`**, không phải recall thô.
→ Nhưng lưu ý điểm TRAKE là **từng phần** (`⅟N × số khoảnh khắc trúng`), nên đáp án sai một
nửa vẫn có điểm. Bộ sinh tổ hợp nên tối ưu *kỳ vọng max*, không phải xác suất trúng cả bộ.

### 1.6. Dữ liệu thi chính thức là VIDEO

PDF mục 3 nói rõ: *"Dữ liệu thi chính thức là Video; các thành phần còn lại (Keyframes,
Objects, CLIP features, Metadata) chỉ nhằm mục đích cung cấp thêm thông tin"*, và **không cam
kết batch 2 sẽ kèm chúng.**

→ **Kiến trúc phải chạy được chỉ từ `.mp4`.** Mọi asset BTC phát coi là quà: dùng để đi nhanh
ở batch 1 và làm tập validation, **không được là phụ thuộc cứng**.

---

## 2. ĐỌC DATA — kho này thực sự là gì

### 2.1. Quy mô [ĐO]

| Hạng mục | Số liệu |
|---|---|
| Video batch 1 | **873** video, **130.7 giờ**, trung bình 9.0 phút/video |
| Keyframe BTC phát | **177.321** ảnh (873 video) → ~203 kf/video → **1 keyframe / 2.66 giây** |
| CLIP features | `float16 (n, 512)`, clip-ViT-B-32, đã chuẩn hoá L2, 1 file `.npy`/video |
| Objects | Faster R-CNN / OpenImages V4, **100 box/ảnh** |
| Metadata | `title, description, keywords, publish_date, author, length` — **tiếng Việt** |
| Shot [ƯỚC] | ~15 shot/phút [ĐO trên 4 video L21] × 7842 phút ≈ **118.000 shot** batch 1 |

### 2.2. Kho này là bản tin "60 Giây" của HTV — và điều đó thay đổi mọi thứ

`media-info` cho thấy toàn bộ kho đến từ kênh **"60 Giây Official" (HTV)**. Đây không phải
video câm, không phải CCTV. Hệ quả trực tiếp:

**(a) Có LỜI THUYẾT MINH tiếng Việt gần như liên tục 130.7 giờ.**
Phát thanh viên/phóng viên *đọc ra bằng lời* chính xác sự kiện đang chiếu: ai, làm gì, ở đâu,
ngày nào. Truy vấn của giám khảo cũng bằng tiếng Việt, mô tả cùng sự kiện đó.
→ **ASR là kênh khớp gần như trực tiếp query ↔ nội dung, không qua trung gian thị giác.**
Hiện **pipeline chưa hề động đến audio.**

**(b) Có CHỮ CHÁY TRÊN HÌNH (lower-third, headline, tên nhân vật, địa danh, ngày).**
Truyền hình luôn đóng chữ tiêu đề tin. Đây là văn bản sạch, ngắn, đúng trọng tâm, **chứa tên
riêng và con số** — đúng thứ mà CLIP dở nhất và BM25 giỏi nhất.
→ Hiện OCR chỉ được lấy *kèm* trong caption VLM (trường `ocr`), tức là **bị nhốt sau nút cổ chai
đắt nhất**. Tách OCR ra chạy độc lập bằng PaddleOCR là rẻ hơn nhiều bậc.

**(c) Video đã dựng, mật độ shot cao (15 shot/phút), nhiều shot studio lặp lại.**
Rất nhiều shot là người dẫn chương trình trong trường quay — gần như giống hệt nhau qua hàng
trăm video. Caption chúng là lãng phí thuần tuý. → cơ sở cho việc **lọc shot trước khi caption** (§4.6).

**(d) `publish_date` có sẵn** → lọc theo thời gian là gần như miễn phí, rất mạnh khi truy vấn
có mốc thời gian ("tháng 8/2024", "dịp lễ 2/9").

### 2.3. Hai cái bẫy đã biết, phải giữ nguyên cảnh báo

- **`001.jpg` KHÔNG phải frame 1.** Tên ảnh chỉ là số thứ tự; frame thật nằm ở cột `frame_idx`
  của `map-keyframes/<vid>.csv`. Nộp nhầm số thứ tự → R-Score = 0 dù tìm đúng cảnh.
- **fps không đồng nhất** (L21_V003 = 25fps, còn lại 30fps). Mọi quy đổi giây ↔ frame phải đọc
  fps **theo từng video**.

### 2.4. Tình trạng tải về và một rào cản vật lý thật [ĐO]

| Nhóm | Video | Keyframes | map/clip/objects/media-info |
|---|---|---|---|
| L21–L24 | **128** ✅ (14.5 GB) | 128 ✅ | 128 ✅ |
| L25–L30 | **0** ❌ | 446 (L26 mới 199/498) | 745 ✅ |
| **Tổng** | **128 / 873** | 574 / 873 | **873 / 873** ✅ |

**Rào cản:** 128 video = 14.5 GB → 873 video ≈ **99 GB** [ƯỚC]. Ổ E: chỉ còn **66 GB**.
Keyframe còn thiếu ≈ +12 GB. **Chưa tính batch 2.** → sẽ hết đĩa trước khi tải xong.

Ổ C: còn **121 GB** trống. Ba cách xử lý, chọn 1:
1. **Rẻ nhất:** vòng lặp *tải → rút audio + keyframe + embedding → xoá `.mp4` → giữ audio*.
   130.7h audio Opus 16kHz mono ≈ **1.9 GB** [ƯỚC] — không đáng kể. Chỉ giữ `.mp4` của các video
   đang cần decode dày cho TRAKE.
2. Đẩy phần đã xử lý sang C: hoặc F: (38 GB).
3. Ổ cứng ngoài 2TB — dứt điểm, nên làm nếu batch 2 lớn.

⚠️ Cách 1 có đánh đổi: TRAKE cần decode dày **video gốc** lúc trả lời truy vấn. Phải giữ được
khả năng lấy lại `.mp4` (giữ link/script tải lại theo `video_id`).

---

## 3. CHẨN ĐOÁN PIPELINE HIỆN TẠI

### Điểm mạnh — giữ nguyên
- Shot detection (AutoShot) + trích keyframe theo optical flow đã được **đo đối chiếu với
  keyframe BTC theo đúng luật chấm** (G6b). Đây là công việc tốt, hiếm đội làm.
- Tốc độ decode 14–17x realtime [ĐO] → index thô toàn kho ~9.1 giờ. Không phải nút cổ chai.
- Đã xuất sẵn 2 loại embedding + `id` chung → đúng kiến trúc 3 kho.
- Có bộ chấm tái hiện đúng cả 3 ví dụ trong PDF (`scratchpad/eval_pdf.py`). Rất giá trị.

### Ba vấn đề cấu trúc

**V1 — Caption VLM toàn kho nằm sai chỗ trong kiến trúc.**
Thực trạng: 6 video đã cắt shot, nhưng captioned **48 / 1612 shot** [ĐO] — 3%. Toàn bộ công
sức 2 ngày qua (load balancer, 429, retry storm, wave scheduling) là để chống đỡ một khối
lượng công việc **mà bài toán không đòi hỏi phải làm hết**. CLIP mới là thứ cho độ phủ,
và CLIP cho 873 video **đã có sẵn, miễn phí, ngay bây giờ**.

**V2 — Bỏ trống hai kênh text đúng ngôn ngữ truy vấn: ASR và OCR.**
Xem §2.2. Đây là kênh khớp query tiếng Việt trực tiếp nhất, chi phí thấp hơn caption 1–2 bậc.

**V3 — Không có tầng truy vấn.**
Không có: dịch/phân tích query, retrieval đa kênh, RRF, gộp keyframe → sự kiện, re-rank,
bộ rải đáp án, VQA, căn chỉnh TRAKE, bộ ghi file nộp. **Đây là toàn bộ nơi sinh điểm.**
Với luật `R@k = max`, một hệ index trung bình + re-rank tốt sẽ **thắng đứt** hệ index xuất sắc
+ không re-rank.

---

## 4. PIPELINE ĐỀ XUẤT

Nguyên tắc xuyên suốt: **kênh RẺ lo ĐỘ PHỦ (100% kho) — kênh ĐẮT lo ĐỘ CHÍNH XÁC (chỉ top-K).**

```
════════ OFFLINE — chạy 1 lần cho cả kho ════════
                                      ┌──────────────────┐
  .mp4 ──► S0 ingest/chuẩn hoá ──────►│  MANIFEST duy    │
            (fps thật, video_id)      │  nhất: video_id, │
                 │                    │  fps, duration   │
                 ├──► S1 shot + keyframe  ─────────────┐  └──────────────────┘
                 │      (có kf BTC → dùng luôn)        │
                 │                                     ▼
                 ├──► S2 VISUAL EMBED  ────────────►  ⬛ Kênh 1: dense ảnh
                 │      SigLIP2 + CLIP B/32           (100% kho — XƯƠNG SỐNG)
                 │
                 ├──► S3 ASR (Whisper VI)  ────────►  ⬛ Kênh 2: lời thoại
                 │      segment + timestamp            (100% kho)
                 │
                 ├──► S4 OCR (PaddleOCR)  ─────────►  ⬛ Kênh 3: chữ trên hình
                 │      trên keyframe                  (100% kho)
                 │
                 ├──► S5 media-info  ──────────────►  ⬛ Kênh 4: tiêu đề/ngày
                 │                                     (100% kho, 0 compute)
                 │
                 └──► S6 CAPTION VLM  ─────────────►  ⬛ Kênh 5: caption
                        CÓ CHỌN LỌC (~20-30% shot)     (phủ một phần)
                                     │
                                     ▼
                        S7 INDEX: Milvus (dense) + BM25 (sparse)

════════ ONLINE — mỗi truy vấn ════════
  query (VI) ──► Q1 phân tích: loại truy vấn / dịch EN / từ khoá / tách sub-event
                     │
                     ▼
                 Q2 truy 5 kênh song song ──► Q3 RRF ──► Q4 gộp thành SỰ KIỆN (đoạn)
                     │
                     ▼
                 Q5 RE-RANK bằng VLM trên top-50..100 sự kiện   ◄── ngân sách VLM ở ĐÂY
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
      KIS          Q&A          TRAKE
   Q6a rải      Q6b VQA +    Q6c decode DÀY đoạn
   frame_id     chuẩn hoá    + DP căn chỉnh đơn điệu
        └────────────┼────────────┘
                     ▼
             Q7 ghi file nộp (đủ 100 dòng) + chấm offline
```

### S0 — Ingest & chuẩn hoá
Một **manifest duy nhất** (`data/index/manifest.parquet`): `video_id, path, fps_thật,
n_frames, duration, có_keyframe_BTC?, có_video?, publish_date`.
Bắt buộc vì fps không đồng nhất (§2.3) và vì tài sản có/không theo từng video.
`fps` lấy từ `map-keyframes` nếu có, ngược lại `ffprobe`.

### S1 — Shot + keyframe
Giữ nguyên module hiện tại (AutoShot + motion sampling). Bổ sung một nhánh rẽ:

- Video **có keyframe BTC** → **dùng luôn**, không decode. Tiết kiệm toàn bộ 9.1 giờ cho
  574/873 video, và có `frame_idx` thật sẵn trong CSV.
- Video **không có** (batch 2, và L26 còn thiếu) → chạy pipeline hiện tại.

Nguyên tắc: **đơn vị index là KEYFRAME, đơn vị hiển thị/tính điểm là SHOT.** Mỗi keyframe mang
`shot_id` để gộp ngược lên.

### S2 — Visual embedding (XƯƠNG SỐNG, ưu tiên số 1)

Đây là kênh duy nhất bảo đảm **100% độ phủ ngay lập tức**.

**Ngày 0 (0 GPU):** nạp thẳng 873 file `.npy` BTC phát vào Milvus → **có baseline KIS chạy trên
toàn kho trong vòng một buổi.** Không cần chờ gì hết. Đây là mốc "có thứ chạy được end-to-end"
mà dự án đang thiếu.

**Sau đó (nâng chất):** clip-ViT-B-32 là model 2021, yếu. Re-embed 177k keyframe bằng model mạnh hơn:
- **SigLIP2 (`ViT-B-16-SigLIP2`, webli)** — đã có trong config, mạnh hơn hẳn B/32.
- Cân nhắc thêm **ViT-L-14 / DFN2B** (Vortex dùng) nếu VRAM cho phép ở batch nhỏ.
- **RRF fusion** giữa 2–3 model (Vortex tr.7): `score = Σ 1/(60 + rank_i)`.
- Chi phí [ƯỚC]: SigLIP-B/16 fp16 trên 4050 ≈ 100–200 ảnh/s → 177k ảnh ≈ **20–30 phút**.
  **Rẻ đến mức không cần bàn.** So sánh: caption toàn kho ≈ 195 giờ.

**Vấn đề ngôn ngữ — quyết định quan trọng:** query là tiếng Việt, CLIP/SigLIP mạnh nhất là
tiếng Anh. Hai hướng:

| Hướng | Ưu | Nhược |
|---|---|---|
| **Dịch query VI→EN bằng LLM rồi encode bằng CLIP tiếng Anh** ⭐ | dùng được model thị giác mạnh nhất; chi phí 1 call LLM / truy vấn (vài chục call cả vòng thi) | thêm 1 điểm hỏng; mất sắc thái tên riêng tiếng Việt |
| CLIP đa ngôn ngữ (jina-clip-v2, mSigLIP) | không cần dịch | yếu hơn rõ rệt ở phần thị giác |

→ **Đề xuất: dịch (hướng 1) làm chính**, sinh **2–4 biến thể dịch** cho mỗi query rồi trung
bình vector (query expansion) — rẻ và chống được rủi ro dịch lệch. Tên riêng/địa danh tiếng Việt
**không dựa vào CLIP** mà để BM25 trên ASR/OCR lo (§S3, S4) — đó chính là lý do phải có đủ kênh.

### S3 — ASR (kênh đang bỏ trống, giá trị cao nhất trên mỗi giờ công)

- Model: **PhoWhisper-medium (VinAI)** — chuyên tiếng Việt; hoặc **faster-whisper large-v3
  int8_float16** + VAD (Silero) để bỏ khoảng lặng.
- Output: `(video_id, start_sec, end_sec, text)` → gán về shot/keyframe theo timestamp.
- Chi phí [ƯỚC]: ~10–15x realtime trên 4050 với VAD → **130.7h / ~12 ≈ 11 giờ GPU**, chạy 1 lần,
  chạy nền qua đêm. So với 195 giờ caption thì đây là món hời.
- Chỉ cần **audio**, không cần giữ `.mp4` → giải quyết luôn bài toán đĩa (§2.4).
- Index 2 dạng: **BM25** (bắt tên riêng, số, địa danh chính xác) + **dense e5-multilingual**
  (bắt diễn đạt khác từ).

⚠️ Cảnh báo trung thực: lời bình **không phải lúc nào cũng đồng bộ với hình** — bản tin hay
đọc trước/sau cảnh vài giây, và có cảnh minh hoạ không liên quan lời. → ASR dùng để **khoanh
vùng ở mức video/đoạn 10–30s**, rồi để kênh thị giác chốt frame. Đừng để ASR quyết định frame.

### S4 — OCR (kênh đang bị nhốt sau caption)

- **PaddleOCR v5** (VIREO dùng, MMM 2026 tr.183-189) hoặc VietOCR.
- Chạy trên keyframe đã có. Ưu tiên **vùng 1/3 dưới khung hình** (lower-third truyền hình) để
  giảm chi phí và nhiễu; có thể chạy đủ khung ở tầng 2 cho các video được retrieval khoanh.
- Chi phí [ƯỚC]: 10–20 ảnh/s → 177k ảnh ≈ **3–5 giờ**.
- Index BM25. Đây là kênh **duy nhất** bắt được chính xác tên người/tên tỉnh/ngày tháng.

### S5 — media-info (0 compute, làm ngay trong 1 giờ)
`title + description + keywords + publish_date` → BM25 ở **mức video**, dùng để
**lọc/boost trước** khi CLIP phải làm việc. `publish_date` cho phép lọc thời gian.
Đã được ghi trong `METHODS.md` G7 là còn tồn đọng — nên làm sớm, chi phí gần bằng 0.

### S6 — Caption VLM: hạ cấp thành CÓ CHỌN LỌC

Vẫn giữ, vì caption paragraph chi tiết là kênh tốt cho truy vấn mô tả cảnh (Cheng MMM 2025
tr.318). Nhưng **không caption tất cả**. Bộ lọc trước khi caption:

1. **Bỏ shot quá ngắn** (< 1.0s) — thường là chuyển cảnh/hiệu ứng.
2. **Bỏ shot studio lặp lại**: cụm CLIP embedding toàn kho, những cụm khổng lồ xuyên video
   (người dẫn chương trình, hình hiệu, bảng quảng cáo) → caption **1 lần cho cả cụm**.
3. **Gộp shot liền kề gần trùng thành SCENE**, caption theo scene chứ không theo shot.
4. **Ưu tiên theo lượng thông tin**: shot có nhiều người/vật thể (dùng `objects` BTC đã phát —
   miễn phí), shot có OCR mới, shot có ASR đang mô tả sự kiện.

[ƯỚC] giảm 3–5 lần → 118k shot còn **~25–40k** đơn vị caption. Với nhịp đã đo là đi từ
~195 giờ xuống **~40–65 giờ**, và có thể chạy dần trong lúc các kênh khác đã phục vụ được.

**Nếu vẫn không kịp:** thứ tự ưu tiên caption theo **video được retrieval chạm nhiều nhất**
trên tập truy vấn thử — caption trở thành *lazy*, sinh theo nhu cầu.

### S7 — Index

Đề xuất **gọn còn 1 backend: Milvus** (dense + **Sparse-BM25** trong cùng hệ, VIREO MMM 2026
tr.198) thay vì dựng thêm Elasticsearch. Ít việc vận hành hơn, đủ dùng.

| Collection | Vector/Field | Đơn vị |
|---|---|---|
| `kf_visual` | dense SigLIP2 (+ CLIP B/32 làm kênh 2) | keyframe |
| `kf_caption` | dense e5-multilingual | keyframe/scene |
| `text_sparse` | BM25 trên `asr ⊕ ocr ⊕ caption ⊕ title` | đoạn 10–30s |
| `video_meta` | title/keywords/publish_date | video |

Khoá nối: `id` toàn cục (đã có) + `video_id, frame_idx thật, shot_id`.
**Bắt buộc lưu `frame_idx` THẬT ngay trong index** — chống bẫy §2.3.

---

## 5. TẦNG TRUY VẤN (nơi sinh điểm — hiện chưa có gì)

### Q1 — Phân tích truy vấn (1 call LLM)
Xuất ra JSON: `{loại: KIS|QA|TRAKE, bản_dịch_EN[2-4 biến thể], từ_khoá_VI[], thực_thể[],
mốc_thời_gian?, sub_events[] (cho TRAKE), câu_hỏi (cho Q&A)}`.

### Q2+Q3 — Truy 5 kênh song song → RRF
`score(kf) = Σ_kênh w_kênh / (60 + rank_kênh(kf))`. Trọng số `w` **phải hiệu chỉnh trên tập
truy vấn tự tạo** (§6), không đoán.

### Q4 — Gộp keyframe → SỰ KIỆN (bước hay bị bỏ qua nhưng rất quan trọng)
Sắp kết quả theo `(video, thời gian)`, quét 2 con trỏ gom keyframe cách nhau ≤ T giây thành
một **đoạn**; điểm đoạn = tổng/max điểm thành viên, **cộng thưởng khi đoạn phủ nhiều sub-query**
(U-CESE Algorithm 2, tr.9).
Lý do: 20 keyframe liên tiếp cùng một cảnh đang **chiếm hết 20 chỗ trong top-20** — đúng chỗ
đắt nhất của bảng điểm. Gộp lại giải phóng slot cho các giả thuyết khác nhau → tăng thẳng R@5, R@20.

### Q5 — RE-RANK bằng VLM (khoản đầu tư lãi nhất, xem §1.2)
Lấy top-50..100 **sự kiện** (không phải frame), gửi VLM 3–5 ảnh đại diện + query, hỏi
"cảnh này có khớp mô tả không, 0–10, vì sao". Sắp lại theo điểm này.
Ngân sách: **~100 call/truy vấn** thay vì 118.000 call/kho. Cùng hạ tầng Gemma/Gemini đã có,
nhưng lượng nhỏ hơn 3 bậc → hết sạch bài toán 429 đang vật lộn.

### Q6a — KIS: bộ RẢI ĐÁP ÁN (answer planner)
Đây là module **chuyển xác suất tìm đúng thành điểm**, và nó phải được tối ưu tường minh.

Đầu vào: danh sách sự kiện đã xếp hạng kèm điểm tin cậy `p_i`.
Ràng buộc: 100 dòng; `R@k = max` tại `k ∈ {1,5,20,50,100}`; cửa sổ đáp án ~10 frame (§1.4).

Chiến lược đề xuất (cần thực nghiệm để chốt):

| Hạng | Phân bổ | Lý do |
|---|---|---|
| 1 | frame tốt nhất của sự kiện #1 | R@1 nặng 0.2 điểm cuối |
| 2–5 | 3 frame nữa **trong** sự kiện #1 (cách 10 frame) + frame tốt nhất sự kiện #2 | cân giữa "đúng shot, lệch frame" và "sai shot" |
| 6–20 | phủ dày sự kiện #1–#3 | |
| 21–100 | mở rộng sang sự kiện #4–#12 | đuôi vẫn còn 0.2 điểm, và nộp thiếu là phí |

Cân bằng cốt lõi: **rải trong shot** (phòng đúng shot sai frame) vs **rải qua shot** (phòng sai shot).
Tỉ lệ tối ưu phụ thuộc độ rộng cửa sổ thật và độ tin cậy của re-rank → **chốt bằng đo trên
tập tự tạo (§6), không chốt bằng cảm tính.**

### Q6b — Q&A
Sau khi có sự kiện top-1: gửi VLM nhiều frame của đoạn + câu hỏi → đáp án.
- Câu trả lời **phải ngắn** ("5", "màu xanh") — luật chấm khớp ngữ nghĩa, câu dài dễ lệch.
- Sinh đáp án ở **nhiều sự kiện top** để 100 dòng không dùng chung một answer sai.
- Chuẩn hoá số/màu/đơn vị; ưu tiên tiếng Việt (PDF cho phép cả 2).

### Q6c — TRAKE: tầng lấy mẫu DÀY + căn chỉnh đơn điệu
1. Retrieval bằng **toàn chuỗi** sự kiện (cộng điểm nếu một video chứa đủ các sub-event theo
   đúng thứ tự — temporal re-ranking, Vortex tr.8). Sai video = 0 nên bước này phải chắc.
2. Với video thắng: **decode DÀY** đoạn nghi vấn — *mọi frame*, không phải keyframe. Một đoạn
   60s ở 30fps = 1800 frame; embed hết bằng SigLIP mất vài giây. Đây chính là "tầng DÀY" đã
   ghi nhận thiếu ở `METHODS.md` G6c/G7.
3. **Căn chỉnh bằng quy hoạch động đơn điệu**: cho N mô tả sub-event và chuỗi T frame, tìm
   `t₁ < t₂ < ... < t_N` cực đại hoá `Σ sim(mô_tả_j, frame_{t_j})` — DP `O(N·T)`, ép đúng thứ tự
   thời gian. (Giống DTW một chiều; tự thiết kế cho bài này, không phải trích từ paper.)
4. Sinh tổ hợp: quanh mỗi `t_j` lấy 3 phương án cách ~10 frame → `3^N` bộ, sắp theo tổng điểm,
   cắt 100. Vì điểm là **từng phần**, ưu tiên đa dạng hoá **mỗi vị trí j độc lập** (không dồn
   biến thiên vào một j) → tối đa hoá số khoảnh khắc trúng.

### Q7 — Bộ ghi file nộp
- Luôn đủ 100 dòng, không trùng `(video_id, frame_id)`.
- `frame_id` là **frame thật**, đã quy đổi theo fps của chính video đó.
- Kẹp `frame_id` vào `[0, n_frames-1]`.
- Chạy `scratchpad/eval_pdf.py` trên tập tự tạo trước mỗi lần nộp.

---

## 6. ĐO ĐẠC — không có tập truy vấn thì mọi lựa chọn ở trên đều là đoán

**Đây là việc cần làm sớm, không phải cuối.** Không có nó thì không thể chỉnh `w` của RRF,
không thể chốt chiến lược rải đáp án, không biết ASR hay CLIP mạnh hơn ở kho này.

Cách tự tạo tập kiểm thử, chi phí thấp:
1. Chọn ~60 đoạn ngẫu nhiên đã có caption/ASR trong L21–L24.
2. Dùng LLM sinh truy vấn tiếng Việt **theo đúng văn phong PDF** từ nội dung đoạn đó
   (KIS + Q&A + TRAKE), tự động có ground truth `(video_id, [s,e])`.
3. Người rà lại nhanh để loại truy vấn mơ hồ / trúng nhiều đoạn.
4. Chấm bằng bộ chấm đã có (`scratchpad/eval_pdf.py`), báo cáo Final Score + tách theo kênh.

⚠️ Điểm yếu phải thừa nhận: truy vấn do LLM sinh **dễ hơn** truy vấn của giám khảo (nó nhìn
thấy caption nên hay dùng lại từ ngữ trong caption → thổi phồng kênh caption). → Bổ sung một
nhóm truy vấn **viết tay hoàn toàn** làm đối chứng, và luôn báo cáo 2 con số riêng.

---

## 7. THỨ TỰ LÀM (đề xuất)

| # | Việc | Chi phí [ƯỚC] | Được gì |
|---|---|---|---|
| **1** | Nạp 873 `.npy` CLIP BTC + `map-keyframes` vào Milvus, viết search CLI đơn giản | **1 ngày, 0 GPU** | **Hệ chạy được end-to-end trên TOÀN KHO.** Chấm dứt tình trạng chưa có gì đo được |
| **2** | Bộ truy vấn kiểm thử + nối bộ chấm (§6) | 1 ngày | Từ đây mọi thay đổi đều đo được |
| **3** | media-info vào BM25 (§S5) | vài giờ | Lọc chủ đề/ngày, gần như miễn phí |
| **4** | Bộ rải đáp án + ghi file nộp (§Q6a, Q7) | 1 ngày | **Nhân điểm cho mọi thứ đã có** |
| **5** | Re-embed SigLIP2 + RRF + dịch query (§S2) | 1 ngày (+30 phút GPU) | Cú nhảy lớn nhất của kênh thị giác |
| **6** | Gộp keyframe → sự kiện (§Q4) | 1 ngày | Giải phóng slot top-20, tăng thẳng R@5/R@20 |
| **7** | **ASR toàn kho** (§S3) | 1 ngày code + 11h GPU nền | Mở kênh mạnh nhất còn bỏ trống |
| **8** | OCR toàn kho (§S4) | 1 ngày + 3–5h GPU | Tên riêng, số, địa danh |
| **9** | VLM re-rank top-100 (§Q5) | 1–2 ngày | Đẩy đáp án đúng lên hạng 1–5 |
| **10** | TRAKE: decode dày + DP căn chỉnh (§Q6c) | 2 ngày | Mở khoá dạng truy vấn thứ 3 |
| **11** | Caption có chọn lọc (§S6), chạy nền | liên tục | Bù kênh mô tả cảnh |
| **∞** | Tải nốt 745 video + xử lý bài toán đĩa (§2.4) | chạy song song từ đầu | Điều kiện cần cho mọi thứ |

**Điểm mấu chốt của thứ tự này:** việc số 1 cho ra hệ thống chạy trên **toàn bộ 873 video**
ngay ngày đầu, trong khi cách đang làm (caption trước) sau 2 ngày mới phủ được **3% của 1 video**.

---

## 8. NHỮNG THỨ CÒN PHẢI CHỐT

1. **BTC có cho dùng API ngoài (Gemini/Gemma) lúc thi không?** Nếu KHÔNG, toàn bộ nhánh
   caption/re-rank/VQA phải chạy local (Qwen3-VL-4B 4-bit trên 6GB) → phải thiết kế lại ngân sách
   ngay từ bây giờ. **Đây là câu hỏi rủi ro cao nhất, nên hỏi BTC sớm.**
2. **Batch 2 lớn cỡ nào, có kèm keyframe/CLIP không?** Quyết định việc có phải tự trích toàn bộ hay không.
3. **Đĩa** — chọn phương án ở §2.4 trước khi tải tiếp.
4. **Cửa sổ `[s,e]` của KIS/Q&A rộng bao nhiêu thật?** Đang phải giả định bi quan ~10 frame.
   Nếu BTC có mẫu truy vấn/đáp án thì con số này chỉnh lại chiến lược rải đáp án đáng kể.
