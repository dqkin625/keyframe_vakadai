"""
Quản lý API key trong .env (validate qua Gemma API).

    python scripts/keys.py list          # liệt kê + trạng thái sống/chết từng key
    python scripts/keys.py add <KEY>     # thêm key nếu sống & chưa có
    python scripts/keys.py add <K1> <K2> # thêm nhiều key
    python scripts/keys.py clean         # xoá mọi key chết, giữ key sống, đánh số lại

Key đọc/ghi ở .env (biến GEMMA_API_KEY, GEMMA_API_KEY_2, ...).
"""
import os
import subprocess
import sys
import warnings

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# tự chuyển sang Python trong .venv (nơi có google-genai)
_VENV_PY = os.path.join(_ROOT, ".venv", "Scripts", "python.exe")
if not os.path.exists(_VENV_PY):
    _VENV_PY = os.path.join(_ROOT, ".venv", "bin", "python")
if (os.path.exists(_VENV_PY)
        and os.path.normcase(sys.executable) != os.path.normcase(_VENV_PY)
        and not os.environ.get("_HCMAI_KEYS_RELAUNCHED")):
    env = dict(os.environ, _HCMAI_KEYS_RELAUNCHED="1", PYTHONIOENCODING="utf-8")
    sys.exit(subprocess.call([_VENV_PY, os.path.abspath(__file__)] + sys.argv[1:], env=env))

import time  # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.captioning.captioner import GemmaAPICaptioner  # noqa: E402

MODEL = "gemma-4-26b-a4b-it"


def _alive(key: str) -> bool:
    """True nếu key gọi được (retry lỗi transient 'client has been closed')."""
    from google import genai
    c = genai.Client(api_key=key)
    for _ in range(3):
        try:
            c.models.generate_content(model=MODEL, contents=["ok"])
            return True
        except Exception as e:
            if "client has been closed" in str(e):
                time.sleep(1); continue
            return False
    return False


def _write(keys):
    lines = ["# API keys Gemini/Gemma - mỗi key 1 dòng (KHÔNG commit)", ""]
    for i, k in enumerate(keys, 1):
        lines.append(("GEMMA_API_KEY" if i == 1 else f"GEMMA_API_KEY_{i}") + f'="{k}"')
    open(".env", "w", encoding="utf-8").write("\n".join(lines) + "\n")


def _load():
    try:
        return GemmaAPICaptioner._load_keys()
    except Exception:
        return []


def cmd_list():
    keys = _load()
    print(f"{len(keys)} key trong .env:")
    for i, k in enumerate(keys, 1):
        print(f"  {i}. ...{k[-4:]}  {'✓ sống' if _alive(k) else '✗ CHẾT'}")


def cmd_add(new_keys):
    keys = _load()
    added = 0
    for nk in new_keys:
        tail = nk[-4:]
        if nk in keys:
            print(f"  ...{tail}: đã có sẵn"); continue
        if _alive(nk):
            keys.append(nk); added += 1
            print(f"  ...{tail}: SỐNG → thêm")
        else:
            print(f"  ...{tail}: CHẾT → bỏ")
    if added:
        _write(keys)
    print(f"Tổng key sống trong .env: {len(keys)}")


def cmd_clean():
    keys = _load()
    alive = [k for k in keys if _alive(k)]
    dead = [k for k in keys if k not in alive]
    _write(alive)
    print(f"Giữ {len(alive)} key sống, xoá {len(dead)} key chết"
          + (": " + ", ".join("..." + k[-4:] for k in dead) if dead else ""))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit()
    cmd = sys.argv[1]
    if cmd == "list":
        cmd_list()
    elif cmd == "add":
        cmd_add(sys.argv[2:])
    elif cmd == "clean":
        cmd_clean()
    else:
        print(__doc__)
