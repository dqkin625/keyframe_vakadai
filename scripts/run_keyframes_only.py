"""
Chạy pipeline ĐẾN HẾT bước keyframe, DỪNG TRƯỚC caption — cho cả một thư mục video.

    python -u scripts/run_keyframes_only.py data/video/Videos_L21_a > logs/kf_L21.log 2>&1
    python -u scripts/run_keyframes_only.py data/video/Videos_L21_a --force   # làm lại video đã xong

LÀM GÌ (đúng thứ tự trong src/pipeline.py):
  1. Cắt shot (AutoShot) + trích keyframe (strategy `action`) — gộp 1 lượt decode.
  2. Lọc keyframe trùng (`dedup_keyframes`, dùng CLIP ViT-B-32).
  3. Ghi ảnh keyframe + metadata.
  4. Dựng chuỗi sự kiện TRAKE (`build_event_records`) — thuần tín hiệu, KHÔNG gọi model.

KHÔNG LÀM:
  - **caption**: dùng `MockCaptioner` -> không gọi API, không nạp VLM. Trường `caption`
    trong output là chuỗi giả, ĐỪNG dùng.
  - **embedding** (CLIP/SigLIP keyframe + e5 caption): tầng này nằm SAU caption trong
    pipeline. Nếu chạy bây giờ thì `caption_emb.npy` sẽ là embedding của caption GIẢ —
    dữ liệu rác dễ bị nhầm là thật. Bật lại bằng `embedding.enabled` khi đã có caption thật.
  - **gán nhãn sự kiện bằng VLM**: `MockCaptioner` không có `label_events_batch` nên bị bỏ qua.

CÓ RESUME: video nào đã có `keyframes.jsonl` thì bỏ qua (trừ khi `--force`).
"""
import argparse
import os
import sys
import time
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import logging          # noqa: E402
import warnings         # noqa: E402
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

VIDEO_EXT = (".mp4", ".mkv", ".avi", ".mov", ".webm")


def _mute_tqdm() -> None:
    """Tắt MỌI thanh tqdm. Chạy 29 video thì tqdm ghi \\r ra stderr làm log phình lên hàng MB
    và không đọc lại được (đo thực tế: 52 KB cho MỘT video). Tiến độ đã có mốc timestamp rồi.
    Vá thẳng `__init__` của lớp -> ăn cả những chỗ đã `from tqdm import tqdm` từ trước."""
    import tqdm as _t
    classes = [_t.tqdm]
    try:
        from tqdm.auto import tqdm as _auto
        if _auto is not _t.tqdm:
            classes.append(_auto)
    except Exception:
        pass
    for cls in classes:
        _orig = cls.__init__

        def _quiet(self, *a, __orig=_orig, **k):
            k["disable"] = True
            __orig(self, *a, **k)

        cls.__init__ = _quiet


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def _is_done(out_root: str, kf_subdir: str, vid: str):
    """Video đã xong THẬT chưa? Trả (bool, lý_do_nếu_chưa).

    KHÔNG chỉ kiểm tra `keyframes.jsonl` tồn tại. `finalize_output` ghi 3 file NỐI TIẾP
    (keyframes -> shots -> events); bị giết giữa chừng (hibernate, mất điện, Ctrl-C) sẽ để
    lại keyframes.jsonl mà thiếu events.jsonl -> lần chạy sau BỎ QUA, thủng dữ liệu âm thầm.
    Nên đòi đủ: 3 file có và khác rỗng, + số ảnh khớp số dòng keyframes.jsonl."""
    d = os.path.join(out_root, vid)
    need = ("keyframes.jsonl", "shots.jsonl", "events.jsonl")
    for f in need:
        p = os.path.join(d, f)
        if not os.path.exists(p):
            return False, f"thiếu {f}"
        if os.path.getsize(p) == 0:
            return False, f"{f} rỗng"
    with open(os.path.join(d, "keyframes.jsonl"), encoding="utf-8") as fh:
        n_row = sum(1 for line in fh if line.strip())
    # Ảnh có thể nằm ở HAI bố cục: phẳng (pipeline vừa ghi ra) hoặc đã gom theo bộ
    # (sau khi chạy group_keyframes_by_collection.py). Phải nhận cả hai, nếu không thì
    # mọi video đã gom sẽ bị coi là CHƯA CHẠY và bị làm lại từ đầu.
    coll = vid.split("_")[0]
    img_dir = None
    for cand in (os.path.join(out_root, kf_subdir, vid),
                 os.path.join(out_root, kf_subdir, coll, vid)):
        if os.path.isdir(cand):
            img_dir = cand
            break
    if img_dir is None:
        return False, "thiếu thư mục ảnh"
    n_img = len([f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")])
    if n_img != n_row:
        return False, f"lệch ảnh/metadata ({n_img} ảnh vs {n_row} dòng)"
    return True, ""


def _fmt(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="thư mục chứa video, vd data/video/Videos_L21_a")
    ap.add_argument("--detector", default="autoshot", choices=("autoshot", "transnetv2", "pyscenedetect"))
    ap.add_argument("--strategy", default="action", choices=("action", "dake", "adaptive", "middle", "clip_reldiff"))
    ap.add_argument("--force", action="store_true", help="làm lại cả video đã có keyframes.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N video đầu (để thử)")
    ap.add_argument("--only", default=None,
                    help="chỉ chạy đúng 1 video theo tên, vd L23_V001 (không cần đuôi .mp4)")
    ap.add_argument("--progress", action="store_true",
                    help="giữ thanh tqdm (mặc định TẮT vì làm log phình, khó đọc lại)")
    ap.add_argument("--dedup-threshold", type=float, default=None,
                    help="ghi đè keyframe_dedup_threshold. LƯU Ý: đây là ngưỡng ĐỘ KHÁC "
                         "(1 - Pearson), giữ frame khi khác > ngưỡng. Càng CAO càng loại nhiều.")
    args = ap.parse_args()

    if not args.progress:
        _mute_tqdm()

    import yaml
    from src.pipeline import process_video, finalize_output
    from src.captioning import build_captioner

    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["shot_detection"]["detector"] = args.detector
    cfg["keyframe"]["strategy"] = args.strategy
    cfg["caption"]["backend"] = "mock"          # KHÔNG caption
    cfg["caption"]["level"] = "shot"
    cfg["caption"]["event_labels"] = False      # không gán nhãn sự kiện bằng VLM
    cfg["embedding"]["enabled"] = False         # tầng embedding nằm SAU caption -> tắt
    if args.dedup_threshold is not None:
        cfg["keyframe"]["keyframe_dedup_threshold"] = args.dedup_threshold

    out_root = cfg["paths"]["output_dir"]
    os.makedirs(out_root, exist_ok=True)

    videos = sorted(f for f in os.listdir(args.folder) if f.lower().endswith(VIDEO_EXT))
    if args.only:
        want = os.path.splitext(args.only)[0]
        videos = [v for v in videos if os.path.splitext(v)[0] == want]
        if not videos:
            _log(f"KHÔNG tìm thấy video '{want}' trong {args.folder}")
            return
    if args.limit:
        videos = videos[:args.limit]

    _log(f"thư mục: {args.folder}")
    _log(f"tìm thấy {len(videos)} video | detector={args.detector} strategy={args.strategy}")
    _log(f"keyframe_dedup_threshold={cfg['keyframe']['keyframe_dedup_threshold']} "
         f"| action_max_gap_sec={cfg['keyframe'].get('action_max_gap_sec')} "
         f"| action_max_gap_min_diff={cfg['keyframe'].get('action_max_gap_min_diff')}")
    _log("caption=MOCK (KHÔNG gọi API) | embedding=TẮT | nhãn sự kiện=TẮT")

    todo = []
    for v in videos:
        vid = os.path.splitext(v)[0]
        done, why = _is_done(out_root, cfg["paths"]["keyframe_subdir"], vid)
        if done and not args.force:
            _log(f"  bỏ qua {vid} (đã xong đủ)")
            continue
        if os.path.isdir(os.path.join(out_root, vid)) and not done:
            _log(f"  làm LẠI {vid} (dở dang: {why})")
        todo.append(v)
    _log(f"cần chạy {len(todo)}/{len(videos)} video")
    if not todo:
        return

    captioner = build_captioner(cfg["caption"])     # MockCaptioner, dựng 1 lần

    t_all = time.time()
    tot_shot = tot_kf = tot_ev = 0
    ok, failed = [], []

    for i, v in enumerate(todo, 1):
        vid = os.path.splitext(v)[0]
        path = os.path.join(args.folder, v)
        size_mb = os.path.getsize(path) / 1e6
        _log(f"[{i}/{len(todo)}] {vid}  ({size_mb:.0f} MB)")
        t0 = time.time()
        try:
            recs, embs, evs = process_video(path, cfg, captioner)
            if not recs:
                _log(f"      ⚠️  {vid}: không ra keyframe nào -> BỎ QUA")
                failed.append((vid, "0 keyframe"))
                continue
            finalize_output(recs, {k: [embs.get(k)] for k in ("clip", "siglip", "caption")},
                            out_root, subdir=vid, all_events=evs)
            n_shot = len({r["shot_index"] for r in recs})
            tot_shot += n_shot
            tot_kf += len(recs)
            tot_ev += len(evs)
            ok.append(vid)
            el = time.time() - t0
            done = i
            avg = (time.time() - t_all) / done
            eta = avg * (len(todo) - done)
            _log(f"      ✓ {n_shot} shot, {len(recs)} keyframe, {len(evs)} sự kiện "
                 f"({_fmt(el)}) | còn {len(todo)-done} video, ETA ~{_fmt(eta)}")
        except KeyboardInterrupt:
            _log("DỪNG theo yêu cầu người dùng"); raise
        except Exception as e:
            failed.append((vid, str(e)[:120]))
            _log(f"      ✗ LỖI {vid}: {e}")
            traceback.print_exc()

    _log("=" * 62)
    _log(f"XONG {len(ok)}/{len(todo)} video trong {_fmt(time.time()-t_all)}")
    _log(f"  tổng: {tot_shot} shot, {tot_kf} keyframe, {tot_ev} sự kiện")
    _log(f"  ảnh : {out_root}/{cfg['paths']['keyframe_subdir']}/<video_id>/")
    _log(f"  meta: {out_root}/<video_id>/keyframes.jsonl | shots.jsonl | events.jsonl")
    if failed:
        _log(f"  ✗ {len(failed)} video LỖI:")
        for vid, err in failed:
            _log(f"      {vid}: {err}")
    _log("NHẮC: trường `caption` trong output là GIẢ (mock). Chạy bước caption riêng khi cần.")


if __name__ == "__main__":
    main()
