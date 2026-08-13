"""
Giảm số keyframe CẦN caption bằng dedup nội dung (Vortex tr.5-6), giữ nguyên độ phủ.

Ý tưởng: nhiều keyframe DAKE liền nhau gần như trùng nhau -> tốn caption vô ích.
  1. CLIP embed mọi keyframe  (embedding này DÙNG LẠI được cho Milvus - Vortex/U-CESE)
  2. Gom cụm greedy theo rel_diff: keyframe mới mở CỤM mới nếu khác cụm hiện tại
     > ngưỡng (cosine distance); nếu không, thuộc cùng cụm với đại diện.
  3. Chỉ caption ĐẠI DIỆN mỗi cụm (batch), rồi GÁN caption đó cho mọi keyframe trong cụm.

Kết quả: mọi keyframe vẫn có caption + vector (đủ cho retrieval), nhưng số lần gọi
model giảm mạnh (POV3: 143 -> ~61 ở ngưỡng 0.1). Bám đúng tinh thần "prioritize
computational efficiency ... reduce redundancy" của Vortex tr.5.
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

_CLIP = {}


def _load_clip(model_name: str, pretrained: str, device: str):
    key = (model_name, pretrained, device)
    if key not in _CLIP:
        import open_clip
        import torch
        m, _, pre = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device)
        m.eval()
        _CLIP[key] = (m, pre, torch)
    return _CLIP[key]


def clip_embed(images: List[np.ndarray], cfg: dict) -> np.ndarray:
    """Trả về ma trận embedding CLIP đã chuẩn hoá [N, D]. Batch trên GPU."""
    from PIL import Image
    m, pre, torch = _load_clip(cfg.get("dedup_clip_model", "ViT-B-32"),
                               cfg.get("dedup_clip_pretrained", "laion2b_s34b_b79k"),
                               cfg.get("device", "cuda"))
    device = cfg.get("device", "cuda")
    bs = 64
    embs = []
    with torch.no_grad():
        for i in range(0, len(images), bs):
            batch = torch.stack([
                pre(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
                for im in images[i:i + bs]]).to(device)
            e = m.encode_image(batch)
            e = e / e.norm(dim=-1, keepdim=True)
            embs.append(e.cpu().numpy())
    return np.concatenate(embs) if embs else np.zeros((0, 512), np.float32)


def cluster_by_reldiff(embs: np.ndarray, threshold: float) -> List[int]:
    """Gom cụm greedy tuần tự. Trả về nhãn cụm cho mỗi keyframe (đại diện = keyframe
    đầu mỗi cụm). rel_diff = 1 - cosine so với ĐẠI DIỆN cụm hiện tại (Vortex Eq.1)."""
    if len(embs) == 0:
        return []
    labels = [0]
    rep = embs[0]
    cur = 0
    for i in range(1, len(embs)):
        if 1.0 - float(embs[i] @ rep) > threshold:   # đủ khác -> mở cụm mới
            cur += 1
            rep = embs[i]
        labels.append(cur)
    return labels


def caption_keyframes_dedup(keyframes, captioner, cfg: dict, log=print):
    """Caption tất cả keyframe nhưng chỉ gọi model trên đại diện mỗi cụm.
    Trả về (captions theo thứ tự keyframe, embeddings, số lần gọi model)."""
    images = [kf.image for kf in keyframes]
    if not images:
        return [], np.zeros((0, 512), np.float32), 0

    embs = clip_embed(images, cfg)
    thr = float(cfg.get("dedup_threshold", 0.1))
    labels = cluster_by_reldiff(embs, thr)
    n_clusters = (max(labels) + 1) if labels else 0

    # đại diện = keyframe ĐẦU mỗi cụm
    reps = {}
    for i, lb in enumerate(labels):
        if lb not in reps:
            reps[lb] = i
    rep_idx = [reps[lb] for lb in range(n_clusters)]

    log(f"      [dedup] {len(images)} keyframe -> {n_clusters} cụm (rel_diff>{thr}) "
        f"-> chỉ caption {n_clusters} đại diện")

    rep_caps = captioner.caption_batch([images[i] for i in rep_idx])
    cap_by_cluster = {lb: rep_caps[lb] for lb in range(n_clusters)}
    captions = [cap_by_cluster[lb] for lb in labels]      # gán về từng keyframe
    return captions, embs, n_clusters
