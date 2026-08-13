"""Thêm key Gemma vào .env AN TOÀN: check trùng -> test sống -> mới thêm.

    python scripts/add_gemma_key.py <KEY> [<KEY2> ...]
    python scripts/add_gemma_key.py --check-only        # chỉ kiểm tra các key ĐANG có

Quy trình cho mỗi key:
  1. TRÙNG với key đã có trong .env?  -> bỏ qua, KHÔNG tốn request
  2. Test THẬT: 1 request chữ + 1 request VIDEO (tải thật của pipeline), trên CẢ 2 model
  3. Sống + không trùng -> append vào .env

Vì sao phải test bằng VIDEO chứ không chỉ prompt chữ: pipeline gửi clip mp4 kèm
VideoMetadata; đã gặp trường hợp key trả lời chữ bình thường nhưng hỏng ở tải thật.
"""
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
ENV = os.path.join(ROOT, ".env")

from dotenv import load_dotenv                      # noqa: E402
load_dotenv(ENV, override=True)
import yaml                                         # noqa: E402
from google import genai                            # noqa: E402
from google.genai import types                      # noqa: E402

MODELS = yaml.safe_load(open("config.yaml", encoding="utf-8"))["caption"]["gemma_models"]
SAMPLE = "data/video/Videos_L21_a/L21_V002.mp4"


def existing():
    """{tên biến: giá trị} các key hiện có trong môi trường (đã nạp từ .env)."""
    return {a: v for a, v in os.environ.items() if a.startswith("GEMMA_API_KEY") and v}


def next_index(names):
    idx = [1]
    for n in names:
        p = n.split("_")[-1]
        if p.isdigit():
            idx.append(int(p))
    return max(idx) + 1


_clip = None


def clip_bytes():
    """Cắt 1 clip nhỏ ĐÚNG như pipeline (fps/scale lấy từ config) để test tải thật."""
    global _clip
    if _clip is not None:
        return _clip
    if not os.path.exists(SAMPLE):
        _clip = b""
        return _clip
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    t = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    t.close()
    subprocess.run([ff, "-y", "-ss", "100", "-to", "104", "-i", SAMPLE,
                    "-vf", "fps=0.5,scale=320:-2", "-an", "-loglevel", "error", t.name],
                   capture_output=True, timeout=60)
    _clip = open(t.name, "rb").read()
    os.unlink(t.name)
    return _clip


def test(key):
    """Trả (sống?, mô tả). Test chữ + video trên cả 2 model."""
    c = genai.Client(api_key=key)
    out = []
    for m in MODELS:
        t = time.time()
        try:
            c.models.generate_content(model=m, contents="Trả lời một từ: xin chào")
            out.append(f"{m[:14]} chữ {time.time()-t:.1f}s OK")
        except Exception as e:
            return False, f"{m[:14]} chữ LỖI {type(e).__name__}: {str(e)[:90]}"
    b = clip_bytes()
    if b:
        for m in MODELS:
            t = time.time()
            try:
                p = [types.Part(inline_data=types.Blob(data=b, mime_type="video/mp4"),
                                video_metadata=types.VideoMetadata(fps=0.5)),
                     "Mô tả ngắn gọn bằng tiếng Việt."]
                r = c.models.generate_content(model=m, contents=p)
                out.append(f"{m[:14]} video {time.time()-t:.1f}s OK "
                           f"({r.usage_metadata.prompt_token_count} tok)")
            except Exception as e:
                return False, f"{m[:14]} VIDEO LỖI {type(e).__name__}: {str(e)[:90]}"
    return True, " | ".join(out)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cur = existing()
    print(f".env đang có {len(cur)} key")

    import collections
    dups = [(v, [a for a, vv in cur.items() if vv == v])
            for v, n in collections.Counter(cur.values()).items() if n > 1]
    if dups:
        print(f"⚠️  TRÙNG LẶP NỘI BỘ trong .env: {len(dups)} cặp")
        for v, names in dups:
            print(f"     ...{v[-6:]} -> {names}")
    else:
        print("✅ không có key nào trùng nhau trong .env")

    if "--check-only" in sys.argv:
        for a, v in sorted(cur.items()):
            ok, msg = test(v)
            print(f"  {a:<20} ...{v[-6:]}  {'SỐNG' if ok else 'CHẾT'}  {msg[:100]}")
        return
    if not args:
        print("\nDùng: python scripts/add_gemma_key.py <KEY> [<KEY2> ...]")
        return

    added = 0
    for k in args:
        k = k.strip()
        print(f"\n--- ...{k[-6:]} ({len(k)} ký tự) ---")
        dup = [a for a, v in cur.items() if v == k]
        if dup:
            print(f"  ❌ TRÙNG {', '.join(dup)} -> bỏ qua (không tốn request)")
            continue
        if not k.startswith("AIzaSy"):
            print(f"  ⚠️  KHÔNG phải API key AI Studio (tiền tố {k[:9]!r}). "
                  f"Nhiều khả năng là token OAuth -> CÓ THỂ HẾT HẠN theo giờ.")
        ok, msg = test(k)
        print(f"  {'✅ SỐNG' if ok else '❌ CHẾT'}: {msg}")
        if not ok:
            continue
        name = f"GEMMA_API_KEY_{next_index(cur)}"
        with open(ENV, "a", encoding="utf-8") as f:
            f.write(f"\n{name}={k}\n")
        cur[name] = k
        added += 1
        print(f"  -> đã thêm {name}")
    print(f"\nXong: thêm {added} key, .env còn {len(cur)} key")


if __name__ == "__main__":
    main()
