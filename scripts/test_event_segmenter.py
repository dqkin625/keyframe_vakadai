"""Kiểm thử event_segmenter bằng TÍN HIỆU TỔNG HỢP (biết trước đáp án).

    python scripts/test_event_segmenter.py        (thoát 0 = đạt, 1 = có test hỏng)

Vì sao dùng tín hiệu tổng hợp: chạy trên video thật thì KHÔNG BIẾT ranh giới đúng nằm đâu
-> không kết luận được thuật toán đúng hay sai. Ở đây ta TỰ DỰNG chuỗi có ranh giới đã biết
rồi đo xem có tìm ra đúng chỗ không.

⚠️ Tín hiệu tổng hợp KHÔNG thay thế được chạy thật: mục [4] (shot tĩnh = ĐÚNG BẰNG 0) từng
cho ĐẠT trong khi video thật (tĩnh = 0 + nhiễu) hỏng nặng — xem METHODS I7-b. Luôn chạy CẢ
`scripts/check_events.py` trên output thật.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.frame_sampling.event_segmenter import (
    _pa_curve, _turn_curve, _smooth1d, _norm01, _local_maxima,
    segment_shot_events, group_scenes, events_to_records)
from src.frame_sampling.shot_detector import Shot

FAIL = []
def check(name, cond, extra=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {name}{'   ' + extra if extra else ''}")
    if not cond:
        FAIL.append(name)

CFG = dict(action_candidate_stride=5, event_min_sec=0.4, event_max_per_shot=8,
           event_pa_weight=0.5, event_boundary_threshold=0.25, event_smooth_sigma=1.0,
           event_window_sec=0.5, action_still_level=0.5)

print("\n[1] PA — phát hiện ĐỔI NGOẠI HÌNH (GEBD ICCV 2021)")
# 20 mẫu: 10 mẫu 'tối' rồi 10 mẫu 'sáng' -> ranh giới đúng ở t=9/10
feats = np.vstack([np.zeros((10, 64), np.float32), np.ones((10, 64), np.float32)])
pa = _pa_curve(feats, w=3)
peak = int(np.argmax(pa))
check("PA đỉnh đúng chỗ đổi ngoại hình (t=9..10)", peak in (9, 10), f"peak={peak}")
check("PA gần 0 ở vùng đồng nhất", pa[3] < 1e-6 and pa[16] < 1e-6)

print("\n[2] TURN — phát hiện ĐỔI ĐỘNG HỌC khi ngoại hình KHÔNG đổi")
# chuyển động sang PHẢI 10 bước rồi ĐẢO CHIỀU sang TRÁI 10 bước (mag không đổi)
per = [(0.0, 0.0, 0.0, 0.0)]
per += [(1.0, 0.0, 1.0, 0.9)] * 10
per += [(-1.0, 0.0, 1.0, 0.9)] * 10
turn = _turn_curve(per, w=3)
peak = int(np.argmax(turn))
check("TURN đỉnh đúng chỗ đảo chiều (t≈11)", 9 <= peak <= 12, f"peak={peak}")

# DỪNG LẠI đột ngột (mag 2.0 -> 0.0), hướng giữ nguyên
per2 = [(0.0, 0.0, 0.0, 0.0)] + [(2.0, 0.0, 2.0, 0.9)] * 10 + [(0.0, 0.0, 0.0, 0.0)] * 10
turn2 = _turn_curve(per2, w=3)
peak2 = int(np.argmax(turn2))
check("TURN đỉnh đúng chỗ DỪNG LẠI (t≈11)", 9 <= peak2 <= 12, f"peak={peak2}")

# cảnh TĨNH + nhiễu hướng ngẫu nhiên -> KHÔNG được sinh đỉnh mạnh
rng = np.random.default_rng(0)
per3 = [(0.0, 0.0, 0.0, 0.0)]
for _ in range(30):
    a = rng.uniform(0, 6.28)
    per3.append((0.01*np.cos(a), 0.01*np.sin(a), 0.01, 0.5))
turn3 = _turn_curve(per3, w=3)
check("TURN KHÔNG nổ ở cảnh tĩnh (nhiễu hướng)", float(np.max(turn3)) < 0.05,
      f"max={float(np.max(turn3)):.4f}")

print("\n[3] segment_shot_events — frame THẬT, thứ tự, phủ kín shot")
fps = 30.0
shot = Shot(index=7, start_frame=1000, end_frame=1000 + 40*5 - 1, fps=fps)
fidxs = [1000 + i*5 for i in range(40)]          # stride 5, 40 mẫu = 200 frame ≈ 6.7s
grays = ([np.zeros((27, 48), np.uint8) for _ in range(20)] +
         [np.full((27, 48), 200, np.uint8) for _ in range(20)])
per_s = [(0.0, 0.0, 0.0, 0.0)] + [(1.0, 0.0, 1.0, 0.2)]*19 + [(-1.0, 0.0, 1.0, 0.2)]*20
evs = segment_shot_events("VID", shot, fidxs, grays, per_s, CFG)
check("cắt được >1 sự kiện", len(evs) > 1, f"n={len(evs)}")
check("ord_in_shot tăng dần 0..n-1", [e.ord_in_shot for e in evs] == list(range(len(evs))))
check("thứ tự thời gian không đảo",
      all(evs[i].end_frame < evs[i+1].start_frame for i in range(len(evs)-1)))
check("phủ TRỌN shot (đầu=start, cuối=end, không hở)",
      evs[0].start_frame == shot.start_frame and evs[-1].end_frame == shot.end_frame
      and all(evs[i+1].start_frame == evs[i].end_frame + 1 for i in range(len(evs)-1)))
check("anchor nằm TRONG đoạn của nó",
      all(e.start_frame <= e.anchor_frame <= e.end_frame for e in evs))
check("frame là frame THẬT (>=1000, không phải chỉ số mẫu)",
      all(e.start_frame >= 1000 for e in evs))
b = evs[1].start_frame
check("ranh giới rơi gần chỗ đổi thật (frame 1100, sai số <= stride)",
      abs(b - 1100) <= 5, f"boundary={b}")

print("\n[4] Trường hợp biên")
short = Shot(index=0, start_frame=0, end_frame=5, fps=fps)
e1 = segment_shot_events("V", short, [0, 5], [np.zeros((27,48),np.uint8)]*2,
                         [(0,0,0,0)]*2, CFG)
check("shot quá ngắn -> đúng 1 sự kiện phủ trọn shot",
      len(e1) == 1 and e1[0].start_frame == 0 and e1[0].end_frame == 5)

flat = Shot(index=1, start_frame=0, end_frame=199, fps=fps)
e2 = segment_shot_events("V", flat, [i*5 for i in range(40)],
                         [np.zeros((27,48),np.uint8)]*40, [(0,0,0,0)]*40, CFG)
check("shot hoàn toàn TĨNH -> 1 sự kiện (không cắt bừa)", len(e2) == 1, f"n={len(e2)}")

print("\n[4b] HỒI QUY — shot TĨNH CÓ NHIỄU không được cắt (bug _norm01 khuếch đại nhiễu)")
# shot 'đứng yên' thực tế: ảnh gần như đồng nhất + nhiễu cảm biến nhỏ, flow ~0.05 (dưới sàn)
rng2 = np.random.default_rng(7)
noisy_grays = [np.clip(120 + rng2.normal(0, 2, (27, 48)), 0, 255).astype(np.uint8)
               for _ in range(40)]
noisy_per = [(0.0, 0.0, 0.0, 0.0)]
for _ in range(39):
    a = rng2.uniform(0, 6.28)
    noisy_per.append((0.05*np.cos(a), 0.05*np.sin(a), 0.05, 0.4))
shot_n = Shot(index=3, start_frame=0, end_frame=40*5-1, fps=fps)
en = segment_shot_events("V", shot_n, [i*5 for i in range(40)], noisy_grays, noisy_per, CFG)
check("shot TĨNH có nhiễu -> 1 sự kiện (không bịa hành động)", len(en) == 1, f"n={len(en)}")

print("\n[4c] HỒI QUY — không có đoạn nào ngắn hơn event_min_sec (kể cả 2 mép shot)")
viol = [e for e in evs if (e.end_time - e.start_time) < CFG["event_min_sec"] - 1e-6]
check("mọi sự kiện >= event_min_sec", not viol,
      f"vi phạm={[(e.ord_in_shot, round(e.end_time-e.start_time,3)) for e in viol]}")

print("\n[5] fps 25 vs 30 — quy đổi giây<->frame theo TỪNG video")
shot25 = Shot(index=0, start_frame=0, end_frame=40*5-1, fps=25.0)
e25 = segment_shot_events("V", shot25, [i*5 for i in range(40)], grays, per_s, CFG)
check("fps=25 ghi đúng vào Event.fps", all(e.fps == 25.0 for e in e25))
check("time = frame/fps đúng với 25fps",
      abs(e25[0].end_time - (e25[0].end_frame + 1)/25.0) < 1e-6)

print("\n[6] group_scenes + events_to_records")
s0 = Shot(index=0, start_frame=0,   end_frame=99,  fps=fps)
s1 = Shot(index=1, start_frame=100, end_frame=199, fps=fps)   # GIỐNG s0
s2 = Shot(index=2, start_frame=200, end_frame=299, fps=fps)   # KHÁC hẳn
dark = [(0, np.zeros((27,48), np.uint8))]
brgt = [(0, np.full((27,48), 255, np.uint8))]
scene = group_scenes([s0, s1, s2], {0: dark, 1: dark, 2: brgt},
                     {**CFG, "event_scene_grouping": True, "event_scene_threshold": 0.12,
                      "event_scene_max_sec": 60.0})
check("2 shot giống nhau -> CÙNG scene", scene[0] == scene[1], f"{scene}")
check("shot khác hẳn -> scene KHÁC", scene[2] != scene[1], f"{scene}")

ev_by_shot = {0: segment_shot_events("V", s0, [i*5 for i in range(20)],
                                     [np.zeros((27,48),np.uint8)]*20, [(0,0,0,0)]*20, CFG),
              1: segment_shot_events("V", s1, [100+i*5 for i in range(20)],
                                     [np.zeros((27,48),np.uint8)]*20, [(0,0,0,0)]*20, CFG)}
rows = events_to_records(ev_by_shot, scene)
check("event_ord đánh theo SCENE (liên tục qua 2 shot)",
      [r["event_ord"] for r in rows] == list(range(len(rows))), str([r["event_ord"] for r in rows]))
check("rows sắp theo thời gian",
      all(rows[i]["start_frame"] <= rows[i+1]["start_frame"] for i in range(len(rows)-1)))
check("có đủ trường TRAKE cần", all(
    k in rows[0] for k in ("video_id","scene_id","shot_index","event_ord",
                           "start_frame","end_frame","anchor_frame","fps","action")))

print("\n[7] parser nhãn sự kiện (chịu model lệch định dạng)")
from src.captioning.captioner import _parse_event_lines
r = _parse_event_lines("1 | người áo vàng | đi tới | ô tô trắng\n2 | người đó | mở cửa xe | -", 2)
check("parse dạng chuẩn", r[0].startswith("người áo vàng") and "mở cửa" in r[1], str(r))
r = _parse_event_lines("Đây là mô tả:\n1. người A chạy\n2. người A dừng lại\n3. người A quay đầu", 3)
check("parse dạng '1.' + có mở bài", r[2] == "người A quay đầu", str(r))
r = _parse_event_lines("người A chạy\nngười A dừng", 3)
check("model bỏ đánh số -> gán theo thứ tự, thiếu thì rỗng",
      r == ["người A chạy", "người A dừng", ""], str(r))
r = _parse_event_lines("[hết quota]", 3)
check("lỗi API -> KHÔNG nhiễm vào dữ liệu", r == ["", "", ""], str(r))
r = _parse_event_lines("1 | a\n5 | b", 3)
check("số ngoài phạm vi bị bỏ qua", r == ["a", "", ""], str(r))

print("\n" + "="*60)
print(f"KẾT QUẢ: {'TẤT CẢ ĐẠT' if not FAIL else str(len(FAIL)) + ' TEST HỎNG: ' + ', '.join(FAIL)}")
print("="*60)
sys.exit(1 if FAIL else 0)
