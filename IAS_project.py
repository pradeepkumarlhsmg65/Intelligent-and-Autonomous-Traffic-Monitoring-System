"""
IAS_final_dashboard.py
Final IAS traffic controller + digital twin dashboard (Windows friendly)

Dependencies:
  pip install ultralytics opencv-python pyserial

How to run:
  python IAS_final_dashboard.py

Controls (press key in console window):
  n/s/e/w  -> Emergency for North/South/East/West lane
  a        -> Ambulance (emergency) (defaults to lane 0)
  p        -> Police (emergency) (defaults to lane 0)
  q        -> Quit program
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
import serial
import msvcrt
import os

# ----------------------- USER CONFIG -----------------------
VIDEO_PATHS = [
    r"D:\projects\north_lane.mp4",   # lane 0 (North)
    r"D:\projects\south_lane.mp4",   # lane 1 (South)
    r"D:\projects\east_lane.mp4",    # lane 2 (East)
    r"D:\projects\west_lane.mp4"     # lane 3 (West)
]

LANE_NAMES = ["NORTH", "SOUTH", "EAST", "WEST"]


HIGH_THRESHOLD = 10       # vehicles >= this → HIGH density
GREEN_HIGH = 15.0         # seconds for HIGH density
GREEN_LOW = 8.0           # seconds for LOW density
EMERGENCY_GREEN = 15.0    # seconds to hold emergency green
YELLOW_TIME = 2.0         # used only for display / explanatory 
STARVATION_LIMIT = 30.0   # if lane waits > this, prioritized

# Confidence threshold for YOLO
YOLO_CONF = 0.35

# Vehicle class indices in COCO (using Ultralytics naming): car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES_IDX = {2, 3, 5, 7}

# Visual layout
FRAME_W = 640
FRAME_H = 360
GRID_COLS = 2
GRID_ROWS = 2
PANEL_W = 400   # right-side stats panel width
FONT = cv2.FONT_HERSHEY_SIMPLEX

# -----------------------------------------------------------

def safe_open_videos(paths):
    caps = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Video file not found: {p}")
        caps.append(cv2.VideoCapture(p))
    return caps

def nonblocking_key():
    """Return single-char lower string if key pressed on Windows console; else None."""
    if msvcrt.kbhit():
        ch = msvcrt.getch()
        try:
            s = ch.decode('utf-8').lower()
        except:
            return None
        return s
    return None


def draw_info_panel(panel, stats):
    """Draw a right-hand stats dashboard panel (panel size: PANEL_H x PANEL_W)."""
    h, w = panel.shape[:2]
    x = 10
    y = 30
    line_h = 26
    cv2.putText(panel, "IAS TRAFFIC DASHBOARD", (x, y), FONT, 0.7, (255,255,255), 2)
    y += line_h*1.5
    cv2.putText(panel, f"Current Lane: {stats['current_name']}", (x, int(y)), FONT, 0.6, (0,255,0), 2)
    y += line_h
    cv2.putText(panel, f"Mode: {stats['mode']}", (x, int(y)), FONT, 0.6, (0,255,255), 2)
    y += line_h
    cv2.putText(panel, f"Emergency: {stats['emergency']}", (x, int(y)), FONT, 0.6, (0,0,255) if stats['emergency'] else (200,200,200), 2)
    y += line_h*1.2
    cv2.putText(panel, "Per-Lane Summary:", (x, int(y)), FONT, 0.6, (255,255,255), 2)
    y += int(line_h*1.2)

    for i, name in enumerate(LANE_NAMES):
        color = (0,255,0) if i == stats['current'] else (200,200,200)
        text = f"{name[:6]:6} | Vehicles: {stats['counts'][i]:2d} | Wait: {int(stats['waits'][i]):3d}s"
        cv2.putText(panel, text, (x, int(y)), FONT, 0.55, color, 1)
        y += int(line_h*0.95)

    y += int(line_h*0.6)
    cv2.putText(panel, "Timing:", (x, int(y)), FONT, 0.6, (255,255,255), 2)
    y += int(line_h*1.1)
    cv2.putText(panel, f"High green: {GREEN_HIGH}s", (x, int(y)), FONT, 0.55, (255,255,255), 1)
    y += int(line_h*0.9)
    cv2.putText(panel, f"Low green : {GREEN_LOW}s", (x, int(y)), FONT, 0.55, (255,255,255), 1)
    y += int(line_h*0.9)
    cv2.putText(panel, f"Emergency  : {EMERGENCY_GREEN}s", (x, int(y)), FONT, 0.55, (255,255,255), 1)

    return panel

def annotate_frame(frame, results, count, lane_name, is_active, wait_time):
    # draw bounding boxes and labels for vehicle detections
    for r in results:
        boxes = r.boxes
        if boxes is None: continue
        for b in boxes:
            cls_idx = int(b.cls[0].item()) if hasattr(b.cls[0], 'item') else int(b.cls[0])
            if cls_idx not in VEHICLE_CLASSES_IDX:
                continue
            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    # head info
    title = f"{lane_name} | Vehicles: {count} | Wait: {int(wait_time)}s"
    color = (0,255,0) if is_active else (0,0,255)
    cv2.rectangle(frame, (0,0), (frame.shape[1], 30), (0,0,0), -1)
    cv2.putText(frame, title, (8,20), FONT, 0.6, color, 2)
    return frame

def stitch_grid(frames):
    # expects frames list of 4 frames (FRAME_H x FRAME_W)
    top = np.hstack((frames[0], frames[1]))
    bottom = np.hstack((frames[2], frames[3]))
    grid = np.vstack((top, bottom))
    return grid

# ---------------------- Main -------------------------
def main():
    # load model
    print("Loading YOLOv8 model (this may take a second)...")
    model = YOLO("yolov8n.pt")

    # open videos
    caps = safe_open_videos(VIDEO_PATHS)

  

    # state
    lane_counts = [0]*4
    lane_waits = [0.0]*4
    last_update_time = time.time()
    current_lane = None
    mode = "NORMAL"  # NORMAL or EMERGENCY
    emergency_active = False
    emergency_lane = None
    emergency_start = 0.0

    # initial population of densities (one frame each)
    print("Reading initial frames to compute start densities...")
    for i,cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        if not ret:
            frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        res = model(frame, conf=YOLO_CONF, verbose=False)
        # count vehicles
        cnt = 0
        for r in res:
            for c in r.boxes.cls:
                idx = int(c) if not hasattr(c, 'item') else int(c.item())
                if idx in VEHICLE_CLASSES_IDX:
                    cnt += 1
        lane_counts[i] = cnt

    # choose starting lane by highest density (tie-breaker: index)
    current_lane = int(np.argmax(np.array(lane_counts)))
    print("Initial lane selected (highest density):", LANE_NAMES[current_lane])
    last_switch_time = time.time()

    # main loop
    while True:
        t0 = time.time()
        # read frames & detect for each lane
        frames_out = []
        results_list = []
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            if not ret:
                frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

            frame = cv2.resize(frame, (FRAME_W, FRAME_H))

            # run detection on this frame (we keep results object to draw boxes)
            res = model(frame, conf=YOLO_CONF, verbose=False)
            # count vehicles
            cnt = 0
            for r in res:
                for c in r.boxes.cls:
                    idx = int(c) if not hasattr(c, 'item') else int(c.item())
                    if idx in VEHICLE_CLASSES_IDX:
                        cnt += 1
            lane_counts[i] = cnt
            results_list.append(res)
            frames_out.append(frame)

        # update waits
        now = time.time()
        dt = now - last_update_time
        last_update_time = now
        for i in range(4):
            if i == current_lane and not emergency_active:
                lane_waits[i] = 0.0
            else:
                lane_waits[i] += dt

        # check for keyboard input (Windows)
        k = nonblocking_key()
        if k == 'q':
            print("Quit requested. Exiting.")
            break
        if k in ('n','s','e','w','a','p'):
            emergency_active = True
            if k == 'n': emergency_lane = 0
            elif k == 's': emergency_lane = 1
            elif k == 'e': emergency_lane = 2
            elif k == 'w': emergency_lane = 3
            elif k == 'a': emergency_lane = 0
            elif k == 'p': emergency_lane = 0
            emergency_start = time.time()
            mode = "EMERGENCY"
            print(">>> Emergency triggered for", LANE_NAMES[emergency_lane])
            # immediate switch
            if current_lane != emergency_lane:
                current_lane = emergency_lane
                last_switch_time = time.time()

        # emergency handling: hold for EMERGENCY_GREEN then resume
        if emergency_active:
            if time.time() - emergency_start >= EMERGENCY_GREEN:
                emergency_active = False
                mode = "NORMAL"
                print(">>> Emergency cleared, resuming normal scheduling")
            # still draw dashboard and skip scheduling changes
        else:
            # Normal scheduling: determine green duration for current lane
            density = lane_counts[current_lane]
            if density >= HIGH_THRESHOLD:
                green_time = GREEN_HIGH
            else:
                green_time = GREEN_LOW

            # if current green elapsed, select next lane
            if time.time() - last_switch_time >= green_time:
                # starvation check: if any lane waited > STARVATION_LIMIT, pick it
                starvation_candidates = [i for i,w in enumerate(lane_waits) if w >= STARVATION_LIMIT]
                if starvation_candidates:
                    next_lane = int(max(starvation_candidates, key=lambda i: lane_waits[i]))
                else:
                    # choose highest density among lanes; tie: highest wait
                    dens_arr = np.array(lane_counts, dtype=float)
                    max_dens = dens_arr.max()
                    high_lanes = [i for i,v in enumerate(lane_counts) if v == max_dens]
                    if len(high_lanes) > 1:
                        # break ties by wait time
                        next_lane = int(max(high_lanes, key=lambda i: lane_waits[i]))
                    else:
                        # if only one top density lane, pick it; else fall back to wait-based
                        next_lane = int(high_lanes[0]) if high_lanes else int(np.argmax(lane_waits))

                if next_lane != current_lane:
                    current_lane = next_lane
                last_switch_time = time.time()
                # reset waits for the selected lane below (in update loop)

        # prepare annotated frames
        annotated = []
        for i, frame in enumerate(frames_out):
            is_active = (i == current_lane)
            f = annotate_frame(frame.copy(), results_list[i], lane_counts[i], LANE_NAMES[i], is_active, lane_waits[i])
            # highlight border for active lane
            border_color = (0,255,0) if is_active else (0,0,255)
            cv2.rectangle(f, (0,0), (f.shape[1]-1, f.shape[0]-1), border_color, 3)
            annotated.append(f)

        # compose grid and dashboard
        grid = stitch_grid(annotated)  # shape (2*FRAME_H, 2*FRAME_W)
        panel_h = grid.shape[0]
        panel = np.zeros((panel_h, PANEL_W, 3), dtype=np.uint8)

        stats = {
            "current": current_lane,
            "current_name": LANE_NAMES[current_lane],
            "mode": mode,
            "counts": lane_counts.copy(),
            "waits": lane_waits.copy(),
            "emergency": (emergency_active and emergency_lane == current_lane)
        }
        panel = draw_info_panel(panel, stats)

        # final combined image
        canvas = np.hstack((grid, panel))
        cv2.imshow("CPS Digital Twin Dashboard", canvas)

        # small sleep to reduce CPU usage
        if cv2.waitKey(1) & 0xFF == 27:
            print("ESC pressed. Exiting.")
            break

    # cleanup
    for cap in caps: cap.release()
    cv2.destroyAllWindows()
    if ser: ser.close()


if __name__ == "__main__":
    main()
