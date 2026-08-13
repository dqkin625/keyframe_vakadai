# Lưu trữ & truy vấn — Milvus + Elasticsearch (theo giải pháp vô địch)

Kiến trúc này bám đúng 2 giải pháp top HCMC AI Challenge 2025:
- **U-CESE** (arxiv 2605.23274) tr.7: *"indexed into the VisualDB by **Milvus** ... Text data are indexed in raw form with [**Elasticsearch**]"*
- **Vortex** (arxiv 2606.19682) tr.1: *"Built on **Milvus and Elasticsearch**"*

## Hai DB, hai vai trò

| DB | Lưu gì | Đơn vị | Truy vấn |
|---|---|---|---|
| **Milvus** | vector CLIP của keyframe | keyframe | ANN ngữ nghĩa (ảnh/ý nghĩa) |
| **Elasticsearch** | caption + OCR + metadata shot | shot | full-text + filter field |

Hai bên **chung khoá `id`** (id global của keyframe) để join.

## Luồng truy vấn (hybrid)

```
Câu hỏi người dùng: "người đàn ông áo đỏ chạy trên cầu"
   │
   ├─ nhánh NGỮ NGHĨA:  text -> CLIP text-encoder -> vector -> Milvus (top-K id keyframe)
   │
   └─ nhánh TEXT:       tìm "áo đỏ", "cầu" trong caption/OCR -> Elasticsearch (BM25)
                                   │
        gộp (fusion) 2 danh sách id  ->  fetch metadata shot từ Elasticsearch
                                   ->  trả về video + timestamp để chấm điểm
```

## Nạp dữ liệu

```bash
# 1) Metadata shot -> Elasticsearch (cần ES chạy ở :9200)
python -m src.db.indexer data/output/shots.jsonl

# 2) Vector CLIP -> Milvus  (sau khi tầng CLIP sinh embedding cho từng keyframe)
#    init_milvus(dim=512) rồi index_milvus(client, coll, ids, embeddings)
```

## Vì sao KHÔNG dùng MongoDB?

Cả 2 đội vô địch **không dùng MongoDB**. Vai trò text/metadata họ giao cho **Elasticsearch** vì
nó mạnh về **full-text search tiếng Việt (BM25) + filter theo field + fuzzy/typo** — đúng nhu cầu
tìm theo caption và OCR. MongoDB vẫn chạy được làm metadata store, nhưng full-text yếu hơn ES.
Xem `indexer.py` để biết mapping index và các hàm search/join.
