"""
Nạp output pipeline vào DB theo đúng kiến trúc 2 đội vô địch HCMC AI Challenge 2025:

    Milvus          <- vector CLIP của KEYFRAME   (tìm bằng NGỮ NGHĨA / ảnh)
    Elasticsearch   <- caption + OCR + metadata SHOT (tìm bằng TEXT / lọc field)

Tham chiếu:
  - U-CESE (arxiv 2605.23274) tr.7: "indexed into the VisualDB by Milvus ...
    Text data are indexed in raw form with [Elasticsearch]"
  - Vortex (arxiv 2606.19682) tr.1: "Built on Milvus and Elasticsearch"

Hai DB chia sẻ chung một khoá `id` (id global của keyframe) để join kết quả:
  text query -> CLIP text-encoder -> vector -> Milvus top-K -> lấy id
             -> tra id trong Elasticsearch -> ra caption/timestamp/OCR để hiển thị.
"""
from __future__ import annotations

import json
from typing import List, Optional


def init_milvus(dim=512, uri="http://localhost:19530", coll="keyframe_clip"):
    """Tạo collection Milvus (HNSW/COSINE). dim khớp CLIP: 512 (ViT-B) / 768 (ViT-L)."""
    from pymilvus import MilvusClient, DataType

    client = MilvusClient(uri=uri)
    if client.has_collection(coll):
        return client, coll

    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id", DataType.INT64, is_primary=True)       # id global keyframe
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
    client.create_collection(coll, schema=schema)

    idx = client.prepare_index_params()
    idx.add_index(field_name="embedding", index_type="HNSW", metric_type="COSINE",
                  params={"M": 16, "efConstruction": 200})
    client.create_index(coll, idx)
    print(f"[milvus] tạo collection {coll} (dim={dim}, HNSW/COSINE)")
    return client, coll


def index_milvus(client, coll, ids: List[int], embeddings: List[list]):
    """embeddings do tầng CLIP sinh (mỗi keyframe 1 vector). Ở đây chỉ nạp."""
    client.insert(coll, [{"id": i, "embedding": e} for i, e in zip(ids, embeddings)])
    client.flush(coll)
    print(f"[milvus] nạp {len(ids)} vector")


def search_milvus(client, coll, query_vec: list, topk=100):
    """Trả về list id keyframe gần nhất về ngữ nghĩa."""
    res = client.search(coll, data=[query_vec], limit=topk, output_fields=["id"])
    return [hit["id"] for hit in res[0]]


def init_es(url="http://localhost:9200", index="shots"):
    """Tạo index với mapping tối ưu cho tiếng Việt: caption/ocr full-text, còn lại để filter."""
    from elasticsearch import Elasticsearch

    es = Elasticsearch(url)
    if es.indices.exists(index=index):
        return es, index

    mapping = {
        "mappings": {
            "properties": {
                "id":          {"type": "long"},        # khớp id keyframe bên Milvus
                "video_id":    {"type": "keyword"},     # lọc chính xác
                "shot_index":  {"type": "integer"},
                "start_time":  {"type": "float"},
                "end_time":    {"type": "float"},
                "caption":     {"type": "text", "analyzer": "standard"},  # full-text
                "ocr":         {"type": "text", "analyzer": "standard"},  # full-text (chữ trên hình)
                "keyframe_path": {"type": "keyword"},
            }
        }
    }
    es.indices.create(index=index, body=mapping)
    print(f"[es] tạo index {index}")
    return es, index


def index_es(es, index, docs: List[dict]):
    """docs = record cấp shot (từ shots.jsonl), có thể thêm trường 'ocr'."""
    from elasticsearch.helpers import bulk

    actions = [{"_index": index, "_id": d["id"], "_source": d} for d in docs]
    bulk(es, actions)
    es.indices.refresh(index=index)
    print(f"[es] nạp {len(docs)} shot")


def search_es_text(es, index, query: str, size=100):
    """Tìm bằng text trên caption + OCR (BM25)."""
    body = {"query": {"multi_match": {"query": query, "fields": ["caption", "ocr"]}}}
    res = es.search(index=index, body=body, size=size)
    return [h["_source"] for h in res["hits"]["hits"]]


def fetch_es_by_ids(es, index, ids: List[int]) -> dict:
    """Sau khi Milvus trả id keyframe -> lấy metadata shot tương ứng để hiển thị."""
    res = es.mget(index=index, body={"ids": ids})
    return {d["_id"]: d.get("_source") for d in res["docs"] if d.get("found")}


# --------------------------------------------------------------------------- #
def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Nạp shots.jsonl vào Elasticsearch (metadata).")
    ap.add_argument("shots_jsonl", help="đường dẫn shots.jsonl")
    ap.add_argument("--es-url", default="http://localhost:9200")
    args = ap.parse_args()

    shots = load_jsonl(args.shots_jsonl)
    for i, s in enumerate(shots):
        s.setdefault("id", i)          # id global; tầng CLIP dùng chung id này cho Milvus
        s.setdefault("ocr", "")        # điền OCR nếu có (VietOCR/PARSeq) — chỗ này để trống
    es, index = init_es(url=args.es_url)
    index_es(es, index, shots)
    print("Xong metadata (Elasticsearch). Vector CLIP nạp qua init_milvus()/index_milvus() sau khi có embedding.")
