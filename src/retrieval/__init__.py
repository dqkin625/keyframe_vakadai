"""Tầng TRUY XUẤT — phần giao với module tiền xử lý keyframe.

Chỉ chứa MẢNH truy xuất thuộc phần việc này: KHỚP CHUỖI cho TRAKE (đọc `events.jsonl`
+ embedding keyframe, không phụ thuộc hạ tầng index của bạn cùng nhóm).

Xem `trake_match.py` và docs/METHODS.md mục J.
"""
from .trake_match import (
    align_sequence,
    cosine_sim_matrix,
    fuse_scores,
    localize_phases,
    snap_to_apex,
    build_submission,
)

__all__ = [
    "align_sequence",
    "cosine_sim_matrix",
    "fuse_scores",
    "localize_phases",
    "snap_to_apex",
    "build_submission",
]
