"""
Phát hiện ranh giới shot (shot boundary detection).

Hai backend:
  - "pyscenedetect": ContentDetector, chạy ngay không cần tải weights. Rất ổn định.
  - "transnetv2":    mạng deep chuyên detect chuyển cảnh (kể cả dissolve/fade),
                     là detector team đã chọn. Cần cài transnetv2-pytorch.

Cả hai trả về danh sách Shot(start_frame, end_frame) theo chỉ số frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Shot:
    index: int
    start_frame: int
    end_frame: int          # inclusive
    fps: float
    motion_hint: str = ""   # tóm tắt CHUYỂN ĐỘNG đo được trong shot (đưa vào prompt caption)
    # Chuỗi SỰ KIỆN có thứ tự bên trong shot (event_segmenter.Event). Rỗng nếu chưa cắt
    # (vd keyframe_signal=histogram, hoặc event_segmentation: false).
    events: list = field(default_factory=list)
    scene_id: int = -1      # id SCENE (gom các shot liền kề cùng bối cảnh) - điền sau

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def start_time(self) -> float:
        return self.start_frame / self.fps

    @property
    def end_time(self) -> float:
        return (self.end_frame + 1) / self.fps


# --------------------------------------------------------------------------- #
# Backend 1: PySceneDetect (mặc định, không cần weights)
# --------------------------------------------------------------------------- #
def _detect_pyscenedetect(video_path: str, threshold: float, min_scene_len: int,
                          show_progress: bool = True) -> List[Shot]:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    video = open_video(video_path)
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    sm.detect_scenes(video, show_progress=show_progress)
    scenes = sm.get_scene_list()
    fps = float(video.frame_rate)

    shots: List[Shot] = []
    for i, (start, end) in enumerate(scenes):
        s = start.get_frames()
        e = end.get_frames() - 1          # PySceneDetect end là exclusive
        if e < s:
            e = s
        shots.append(Shot(index=i, start_frame=s, end_frame=e, fps=fps))

    # Nếu video không có cut nào (1 shot), get_scene_list trả rỗng -> tự tạo 1 shot.
    if not shots:
        import cv2
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        shots = [Shot(index=0, start_frame=0, end_frame=max(0, total - 1), fps=fps)]
    return shots


# --------------------------------------------------------------------------- #
# Backend 2: TransNetV2 (detector của team)
# --------------------------------------------------------------------------- #
def _ensure_ffmpeg() -> None:
    """TransNetV2 giải mã video qua ffmpeg-python -> cần binary `ffmpeg`.
    Ta dùng ffmpeg ĐÓNG GÓI TRONG VENV (imageio-ffmpeg), không cài gì vào máy:
    tạo alias tên `ffmpeg.exe` cạnh binary rồi thêm thư mục đó vào PATH của
    tiến trình hiện tại (process-local, không sửa PATH hệ thống)."""
    import os
    import shutil

    try:
        import imageio_ffmpeg
    except ImportError:
        return  # nếu máy đã có ffmpeg trên PATH thì vẫn chạy được

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    d = os.path.dirname(exe)
    alias = os.path.join(d, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not os.path.exists(alias):
        shutil.copy2(exe, alias)
    if d not in os.environ.get("PATH", ""):
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def _detect_transnetv2(video_path: str, prob_threshold: float,
                       quiet: bool = False) -> List[Shot]:
    """Dùng `analyze_video` của transnetv2-pytorch (thay cho `detect_scenes`) vì nó
    nhận tham số quiet=False -> HIỆN THANH TIẾN ĐỘ khi xử lý frame, và trả về luôn
    cả `scenes` lẫn `fps` trong một lần chạy.
    scenes: list dict {shot_id, start_frame, end_frame, probability, start_time, end_time}"""
    from transnetv2_pytorch import TransNetV2  # pip install transnetv2-pytorch

    _ensure_ffmpeg()

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    model = TransNetV2(device=device)
    model.eval()

    res = model.analyze_video(video_path, threshold=prob_threshold, quiet=quiet)
    scenes = res["scenes"]
    fps = float(res.get("fps") or 25.0)

    return [
        Shot(index=i,
             start_frame=int(s["start_frame"]),
             end_frame=int(s["end_frame"]),
             fps=fps)
        for i, s in enumerate(scenes)
    ]


# --------------------------------------------------------------------------- #
# Backend 3: AutoShot — detector mà CẢ HAI đội vô địch HCMC 2025 dùng
#   U-CESE (arxiv 2605.23274) tr.7 : "We utilize AutoShot [28] for this task"
#   Vortex (arxiv 2606.19682) tr.5 : "we utilize AutoShot [18] to segment the video"
# Paper AutoShot (arxiv 2304.06116): vượt TransNetV2 +4.2% F1 trên tập SHOT.
# Kiến trúc là biến thể NAS của TransNetV2 -> tiền xử lý/cửa sổ y hệt TransNetV2.
# --------------------------------------------------------------------------- #
_AUTOSHOT_CKPT_URL = ("https://huggingface.co/backseollgi/AutoShot/"
                      "resolve/main/ckpt_0_200_0.pth")
_AUTOSHOT_CACHE = {}


def _autoshot_weights_path() -> str:
    """Trả về đường dẫn weights, tự tải về models/ nếu chưa có."""
    import os
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    d = os.path.join(root, "models")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "autoshot_ckpt_0_200_0.pth")
    if not os.path.exists(p):
        import ssl
        import urllib.request
        print(f"[AutoShot] tải weights lần đầu (~57MB) -> {p}")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(_AUTOSHOT_CKPT_URL,
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=600) as r:
            open(p, "wb").write(r.read())
    return p


def _load_autoshot(device: str):
    if device in _AUTOSHOT_CACHE:
        return _AUTOSHOT_CACHE[device]
    import torch
    from .autoshot_arch import TransNetV2Supernet

    model = TransNetV2Supernet().eval()
    sd = torch.load(_autoshot_weights_path(), map_location="cpu", weights_only=False)
    # checkpoint gốc bọc state_dict trong key 'net'
    if isinstance(sd, dict):
        sd = sd.get("net", sd.get("state_dict", sd))
    # bỏ tiền tố 'module.' nếu ckpt lưu từ DataParallel; chỉ nạp khoá khớp
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    own = model.state_dict()
    ok = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(ok, strict=False)
    print(f"[AutoShot] nạp {len(ok)}/{len(own)} tensor trọng số")
    model = model.to(device)
    _AUTOSHOT_CACHE[device] = model
    return model


def autoshot_infer(frames: np.ndarray, prob_threshold: float, fps: float,
                   quiet: bool = False) -> List[Shot]:
    """AutoShot inference trên mảng frame 48x27 RGB ĐÃ giải mã -> List[Shot].
    Tách khỏi phần decode để có thể GỘP decode với DAKE (dùng chung 1 lượt giải mã)."""
    import torch
    from tqdm import tqdm
    from transnetv2_pytorch import TransNetV2 as _T   # dùng lại predictions_to_scenes

    n = len(frames)
    if n == 0:                     # video rỗng/hỏng (0 frame giải mã được) -> KHÔNG có shot
        return []                  # caller (process_video) sẽ bỏ qua video này, không crash

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_autoshot(device)

    # --- cửa sổ trượt 100 frame, bước 50, đệm 25 đầu/cuối ---
    win, step, pad = 100, 50, 25
    tail = pad + step - (n % step if n % step else step)
    padded = np.concatenate(
        [frames[:1]] * pad + [frames] + [frames[-1:]] * tail, axis=0)

    preds = []
    bar = None if quiet else tqdm(total=n, desc="Processing frames", unit="frame")
    with torch.no_grad():
        ptr = 0
        while ptr + win <= len(padded):
            # [T,H,W,C] -> [1,C,T,H,W] (Conv3d của AutoShot nhận channels-first)
            batch = torch.from_numpy(padded[ptr:ptr + win].copy()).float()
            batch = batch.permute(3, 0, 1, 2).unsqueeze(0).to(device)
            out = model(batch)
            one_hot = out[0] if isinstance(out, tuple) else out
            p = torch.sigmoid(one_hot).reshape(-1).cpu().numpy()[pad:pad + step]
            preds.append(p)
            ptr += step
            if bar:
                bar.update(min(step, n - bar.n))
    if bar:
        bar.close()

    single = np.concatenate(preds)[:n]
    scenes = _T.predictions_to_scenes((single > prob_threshold).astype(np.uint8))
    return [Shot(index=i, start_frame=int(s), end_frame=int(e), fps=fps)
            for i, (s, e) in enumerate(scenes)]


def _detect_autoshot(video_path: str, prob_threshold: float,
                     quiet: bool = False, use_gpu_decode: bool = True) -> List[Shot]:
    import cv2
    from .video_io import extract_frames

    if not quiet:
        print(f"[AutoShot] trích frame từ {video_path}")
    frames = extract_frames(video_path, 48, 27, use_gpu=use_gpu_decode)  # 48x27 RGB
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()
    return autoshot_infer(frames, prob_threshold, fps, quiet)


def detect_shots(video_path: str, cfg: dict) -> List[Shot]:
    """cfg = block `shot_detection` trong config.yaml."""
    detector = cfg.get("detector", "pyscenedetect").lower()
    if detector == "autoshot":
        return _detect_autoshot(video_path,
                                cfg.get("autoshot_prob_threshold", 0.5),
                                quiet=bool(cfg.get("quiet", False)),
                                use_gpu_decode=bool(cfg.get("gpu_decode", True)))
    if detector == "transnetv2":
        return _detect_transnetv2(video_path,
                                  cfg.get("transnet_prob_threshold", 0.5),
                                  quiet=bool(cfg.get("quiet", False)))
    return _detect_pyscenedetect(
        video_path,
        threshold=cfg.get("threshold", 27.0),
        min_scene_len=cfg.get("min_scene_len", 15),
        show_progress=not bool(cfg.get("quiet", False)),
    )
