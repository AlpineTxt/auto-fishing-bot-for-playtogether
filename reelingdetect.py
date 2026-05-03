import time

import cv2
import keyboard
import mss
import numpy as np
import pyautogui

TEMPLATE_PATHS = [
    r"C:\Users\Lenovo\THEBot\fishing bot\2nd\reeling.png",
]

VIS_ROI = {"left": 1083, "top": 156, "width": 747, "height": 354}
DETECT_ROI = {"left": 1705, "top": 403, "width": 115, "height": 88}

MAX_DIFF = 0.32
SCALES = [0.80, 0.90, 1.00, 1.10, 1.20]
LOOP_SLEEP = 0.01
CLICK_ENABLED = True

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
        res = cv2.matchTemplate(gray_eq, t, cv2.TM_SQDIFF_NORMED)
        diff_min, _, min_loc, _ = cv2.minMaxLoc(res)
        if float(diff_min) < best_diff:
            best_diff = float(diff_min)
            best_loc = min_loc
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


print("Detectbait (info-diff) running. Press F8 or ESC to stop.")
clicked_this_detection = False
prev_diff = None

with mss.mss() as sct:
    cv2.namedWindow("DetectBait InfoDiff", cv2.WINDOW_NORMAL)
    cv2.moveWindow("DetectBait InfoDiff", 60, 80)
    cv2.resizeWindow("DetectBait InfoDiff", 900, 500)

    while True:
        if keyboard.is_pressed("f8"):
            print("Stopped by F8.")
            break

        frame = np.array(sct.grab(VIS_ROI))[:, :, :3]

        dx1 = DETECT_ROI["left"] - VIS_ROI["left"]
        dy1 = DETECT_ROI["top"] - VIS_ROI["top"]
        dx2 = dx1 + DETECT_ROI["width"]
        dy2 = dy1 + DETECT_ROI["height"]
        dx1 = max(0, dx1)
        dy1 = max(0, dy1)
        dx2 = min(frame.shape[1], dx2)
        dy2 = min(frame.shape[0], dy2)

        detect_patch = frame[dy1:dy2, dx1:dx2]
        if detect_patch.size == 0:
            cv2.imshow("DetectBait InfoDiff", frame)
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

        if detected and CLICK_ENABLED and not clicked_this_detection:
            click_x = DETECT_ROI["left"] + x + (tw // 2)
            click_y = DETECT_ROI["top"] + y + (th // 2)
            pyautogui.click(click_x, click_y)
            print(f"[{time.strftime('%H:%M:%S')}] CLICK ({click_x},{click_y}) diff={diff:.3f}", flush=True)
            clicked_this_detection = True

        diff_delta = 0.0 if prev_diff is None else (diff - prev_diff)
        prev_diff = diff

        view = frame.copy()
        color = (0, 255, 0) if detected else (0, 255, 255)
        cv2.rectangle(view, (dx1, dy1), (dx2, dy2), (255, 150, 0), 2)
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
            f"max_diff={MAX_DIFF:.2f} delta={diff_delta:+.3f} scale={scale:.2f}",
            (8, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
        )
        cv2.putText(
            view,
            f"detectROI=({DETECT_ROI['left']},{DETECT_ROI['top']}) {DETECT_ROI['width']}x{DETECT_ROI['height']}",
            (8, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 150, 0),
            2,
        )
        cv2.imshow("DetectBait InfoDiff", view)

        if cv2.waitKey(1) & 0xFF == 27:
            print("Stopped by ESC.")
            break

        time.sleep(LOOP_SLEEP)

cv2.destroyAllWindows()
