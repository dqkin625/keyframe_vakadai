"""
Sinh EMBEDDING cho keyframe (2 loại), biến output thành gói SẴN-SÀNG-INDEX cho Milvus.
Bám thiết kế 3 kho của U-CESE (tr.7) và kênh text-to-caption của MMM 2025 (Cheng, tr.318).

  1. CLIP image embedding của keyframe   -> VisualDB (tìm bằng ảnh/ngữ nghĩa)
  2. Caption embedding (Sentence Transformer đa ngôn ngữ) -> TextualDB embedding
     (kênh "text-to-caption retrieval" — mã hoá caption tiếng Việt để tìm theo nghĩa)

Lưu ý: CLIP embedding thường ĐÃ được tính khi dedup keyframe (reduce.py) -> tái dùng,
gần như MIỄN PHÍ. Chỉ caption embedding là bước thêm (nhẹ, batch trên GPU).

Dùng model e5 qua transformers TRỰC TIẾP (không qua sentence-transformers) vì máy này
chặn DLL scipy mà sentence-transformers/sklearn kéo theo.
"""
from __future__ import annotations

from typing import List

import numpy as np


def embed_keyframes_clip(images: List[np.ndarray], cfg: dict) -> np.ndarray:
    """CLIP image embedding cho keyframe. Tái dùng hàm trong reduce.py."""
    from ..captioning.reduce import clip_embed
    return clip_embed(images, cfg)


def embed_keyframes_siglip(images: List[np.ndarray], cfg: dict) -> np.ndarray:
    """SigLIP2 image embedding — embedding ảnh THỨ 2 (fine-grained) để RRF fusion với CLIP
    (Vortex/VIREO dùng CLIP + SigLIP2). Dùng lại open_clip với model SigLIP2."""
    from ..captioning.reduce import clip_embed
    scfg = {**cfg,
            "dedup_clip_model": cfg.get("siglip_model", "ViT-B-16-SigLIP2"),
            "dedup_clip_pretrained": cfg.get("siglip_pretrained", "webli")}
    return clip_embed(images, scfg)


class CaptionEmbedder:
    """Mã hoá caption (tiếng Việt) bằng multilingual-e5 để tìm text-to-caption.
    e5 cần tiền tố 'passage:' cho tài liệu và 'query:' cho truy vấn."""

    _CACHE = {}

    def __init__(self, cfg: dict):
        import torch
        from transformers import AutoTokenizer, AutoModel

        self.model_id = cfg.get("caption_emb_model", "intfloat/multilingual-e5-base")
        self.device = cfg.get("device", "cuda")
        self.batch = int(cfg.get("caption_emb_batch", 64))
        key = (self.model_id, self.device)
        if key not in CaptionEmbedder._CACHE:
            tok = AutoTokenizer.from_pretrained(self.model_id)
            mdl = AutoModel.from_pretrained(self.model_id).to(self.device).eval()
            CaptionEmbedder._CACHE[key] = (tok, mdl, torch)
        self.tok, self.mdl, self.torch = CaptionEmbedder._CACHE[key]

    def _encode(self, texts: List[str]) -> np.ndarray:
        torch = self.torch
        import torch.nn.functional as F
        out = []
        for i in range(0, len(texts), self.batch):
            b = self.tok(texts[i:i + self.batch], padding=True, truncation=True,
                         max_length=256, return_tensors="pt").to(self.device)
            with torch.no_grad():
                o = self.mdl(**b)
            m = b.attention_mask.unsqueeze(-1).float()
            emb = (o.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1e-9)  # mean pool
            out.append(F.normalize(emb, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 768), np.float32)

    def encode_captions(self, captions: List[str]) -> np.ndarray:
        return self._encode(["passage: " + (c or "") for c in captions])

    def encode_query(self, query: str) -> np.ndarray:
        return self._encode(["query: " + query])[0]
