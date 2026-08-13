"""
save_shot_cache.py — Lưu CACHE (spans + keyframe) của 1 video để test caption nhanh về sau.

Copy shots.jsonl + keyframes.jsonl + ảnh keyframe từ data/output/<video_id> sang
data/cache/<video_id>. Shot detection + keyframe là TẤT ĐỊNH nên cache dùng lại được mãi
-> test_caption.py / caption_waves.py bỏ qua được khâu tốn ~130s này.

ĐIỀU KIỆN: đã chạy full pipeline 1 lần cho video đó (python main.py <...>).

VÍ DỤ:  python scripts/save_shot_cache.py L21_V001
"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def main():
    vid = sys.argv[1] if len(sys.argv) > 1 else "L21_V001"
    out = os.path.join("data", "output", vid)
    kf_src = os.path.join("data", "output", "keyframes", vid)
    if not os.path.exists(os.path.join(out, "shots.jsonl")):
        print(f"[LỖI] chưa có {out}/shots.jsonl. Chạy full 1 lần trước: python main.py <...>/{vid}.mp4")
        sys.exit(1)
    cache = os.path.join("data", "cache", vid)
    os.makedirs(cache, exist_ok=True)
    for f in ("shots.jsonl", "keyframes.jsonl"):
        shutil.copy2(os.path.join(out, f), os.path.join(cache, f))
    dst_kf = os.path.join(cache, "keyframes")
    if os.path.isdir(kf_src):
        if os.path.isdir(dst_kf):
            shutil.rmtree(dst_kf)
        shutil.copytree(kf_src, dst_kf)
    n_img = len(os.listdir(dst_kf)) if os.path.isdir(dst_kf) else 0
    n_shot = sum(1 for _ in open(os.path.join(cache, "shots.jsonl"), encoding="utf-8"))
    print(f"✅ cache {vid}: {n_shot} shot, {n_img} ảnh keyframe -> {cache}")
    print(f"   giờ test nhanh:  python scripts/test_caption.py {vid} --shots 40")
    print(f"   hoặc chạy đợt:   python scripts/caption_waves.py {vid}")


if __name__ == "__main__":
    main()
