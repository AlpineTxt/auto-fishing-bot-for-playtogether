import time

import cv2
import keyboard
import mss
import numpy as np
import pyautogui

TEMPLATE_PATHS = [
    r"C:\Users\Lenovo\THEBot\fishing bot\2nd\detectbait.png",
    r"C:\Users\Lenovo\THEBot\fishing bot\2nd\bait.png",
    r"C:\Users\Lenovo\THEBot\fishing bot\2nd\bait.new.png",
]
# Large window for visualization.
VIS_ROI = {"left": 1083, "top": 156, "width": 747, "height": 354}
# Small area used for real detection (absolute screen coordinates).
DETECT_ROI = {"left": 1649, "top": 329, "width": 74, "height": 58}

MAX_DIFF = 0.32
SCALES = [0.80, 0.90, 1.00, 1.10, 1.20]
LOG_COOLDOWN = 0.15
LOOP_SLEEP = 0.01

templates = []
for path in TEMPLATE_PATHS:
    t = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if t is None:
        print(f"[INFO] Missing template skipped: {path}")
        continue
    templates.append(cv2.equalizeHist(t))
if not templates:
    raise FileNotFoundError("No valid bait templates loaded.")


def best_match_single(gray_eq, base_template):
    best_diff = 999.0
    best_loc = (0, 0)
    best_size = (0, 0)
    best_scale = 1.0

    for scale in SCALES:
        tw = max(8, int(base_template.shape[1] * scale))
        th = max(8, int(base_template.shape[0] * scale))
        if tw >= gray_eq.shape[1] or th >= gray_eq.shape[0]:
            continue

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        t = cv2.resize(base_template, (tw, th), interpolation=interp)

        # SQDIFF: lower is better (more similar).
        res_diff = cv2.matchTemplate(gray_eq, t, cv2.TM_SQDIFF_NORMED)
        diff_min, _, diff_loc, _ = cv2.minMaxLoc(res_diff)

        if float(diff_min) < best_diff:
            best_diff = float(diff_min)
            best_loc = diff_loc
            best_size = (tw, th)
            best_scale = scale

    return best_diff, best_loc, best_size, best_scale


def best_match(gray_eq, template_list):
    best_diff = 999.0
    best_loc = (0, 0)
    best_size = (0, 0)
    best_scale = 1.0
    for t in template_list:
        diff, loc, size, scale = best_match_single(gray_eq, t)
        if diff < best_diff:
            best_diff = diff
            best_loc = loc
            best_size = size
            best_scale = scale
    return best_diff, best_loc, best_size, best_scale

print("Detectbait bot running. Press F8 or ESC to stop.")

last_log_at = 0.0
clicked_this_detection = False

with mss.mss() as sct:
    cv2.namedWindow("DetectBait Bot", cv2.WINDOW_NORMAL)
    cv2.moveWindow("DetectBait Bot", 60, 80)
    cv2.resizeWindow("DetectBait Bot", 900, 500)

    while True:
        if keyboard.is_pressed("f8"):
            print("Stopped by F8.")
            break

        frame = np.array(sct.grab(VIS_ROI))[:, :, :3]

        # Convert detection ROI from absolute screen coords to local coords inside VIS_ROI.
        dx1 = DETECT_ROI["left"] - VIS_ROI["left"]
        dy1 = DETECT_ROI["top"] - VIS_ROI["top"]
        dx2 = dx1 + DETECT_ROI["width"]
        dy2 = dy1 + DETECT_ROI["height"]

        # Clamp to visible frame bounds.
        dx1 = max(0, dx1)
        dy1 = max(0, dy1)
        dx2 = min(frame.shape[1], dx2)
        dy2 = min(frame.shape[0], dy2)

        detect_patch = frame[dy1:dy2, dx1:dx2]
        if detect_patch.size == 0:
            cv2.imshow("DetectBait Bot", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                print("Stopped by ESC.")
                break
            time.sleep(LOOP_SLEEP)
            continue

        gray = cv2.cvtColor(detect_patch, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        diff, loc, (tw, th), scale = best_match(gray, templates)
        x, y = loc
        detected = diff <= MAX_DIFF
        if not detected:
            clicked_this_detection = False

        now = time.time()
        if detected and (now - last_log_at) >= LOG_COOLDOWN:
            print(
                f"[{time.strftime('%H:%M:%S')}] DETECTED diff={diff:.3f} "
                f"loc=({DETECT_ROI['left'] + x},{DETECT_ROI['top'] + y})",
                flush=True,
            )
            last_log_at = now

        # Single click: one click when detection starts, then wait until detection drops.
        if detected and not clicked_this_detection:
            click_x = DETECT_ROI["left"] + x + (tw // 2)
            click_y = DETECT_ROI["top"] + y + (th // 2)
            pyautogui.click(click_x, click_y)
            print(f"[{time.strftime('%H:%M:%S')}] CLICK at ({click_x},{click_y})", flush=True)
            clicked_this_detection = True

        view = frame.copy()
        color = (0, 255, 0) if detected else (0, 255, 255)
        # Draw detection sub-area in blue on the large visualization window.
        cv2.rectangle(view, (dx1, dy1), (dx2, dy2), (255, 150, 0), 2)
        # Draw matched object inside the detection sub-area.
        cv2.rectangle(view, (dx1 + x, dy1 + y), (dx1 + x + tw, dy1 + y + th), color, 2)
        cv2.putText(
            view,
            f"{'DETECTED' if detected else 'WAIT'} diff={diff:.3f}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )
        cv2.putText(
            view,
            f"max_diff={MAX_DIFF:.2f} scale={scale:.2f}",
            (8, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
        cv2.putText(
            view,
            f"detectROI=({DETECT_ROI['left']},{DETECT_ROI['top']}) "
            f"{DETECT_ROI['width']}x{DETECT_ROI['height']}",
            (8, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 150, 0),
            2,
        )
        cv2.imshow("DetectBait Bot", view)

        if cv2.waitKey(1) & 0xFF == 27:
            print("Stopped by ESC.")
            break

        time.sleep(LOOP_SLEEP)

cv2.destroyAllWindows()
