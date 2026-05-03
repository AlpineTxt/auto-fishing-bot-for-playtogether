import time

import cv2
import keyboard
import mss
import numpy as np

TEMPLATE_PATHS = [
    r"C:\Users\Lenovo\THEBot\fishing bot\2nd\dayalert.png",
    r"C:\Users\Lenovo\THEBot\fishing bot\2nd\alert 1.png",
    r"C:\Users\Lenovo\THEBot\fishing bot\2nd\alert 2.png",
    r"C:\Users\Lenovo\THEBot\fishing bot\2nd\alert3.png",
    r"C:\Users\Lenovo\THEBot\fishing bot\2nd\alert 4.png",
    r"C:\Users\Lenovo\THEBot\fishing bot\2nd\alert5.png",
]
# Large window for visualization.
VIS_ROI = {"left": 1083, "top": 156, "width": 747, "height": 354}
# Small area used for real detection (absolute screen coordinates).
DETECT_ROI = {"left": 1336, "top": 153, "width": 110, "height": 103}

MATCH_THRESHOLD = 0.65
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
    raise FileNotFoundError("No valid alert templates loaded.")


def best_match_single(gray_eq, base_template):
    best_score = -1.0
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

        # Direct matching: higher score means better match.
        res = cv2.matchTemplate(gray_eq, t, cv2.TM_CCOEFF_NORMED)
        _, max_score, _, max_loc = cv2.minMaxLoc(res)

        if float(max_score) > best_score:
            best_score = float(max_score)
            best_loc = max_loc
            best_size = (tw, th)
            best_scale = scale

    return best_score, best_loc, best_size, best_scale


def best_match(gray_eq, template_list):
    best_score = -1.0
    best_loc = (0, 0)
    best_size = (0, 0)
    best_scale = 1.0
    for t in template_list:
        score, loc, size, scale = best_match_single(gray_eq, t)
        if score > best_score:
            best_score = score
            best_loc = loc
            best_size = size
            best_scale = scale
    return best_score, best_loc, best_size, best_scale

print("Detectalert bot running. Press F8 or ESC to stop.")

last_log_at = 0.0

with mss.mss() as sct:
    cv2.namedWindow("DetectAlert Bot", cv2.WINDOW_NORMAL)
    cv2.moveWindow("DetectAlert Bot", 60, 80)
    cv2.resizeWindow("DetectAlert Bot", 900, 500)

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
            cv2.imshow("DetectAlert Bot", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                print("Stopped by ESC.")
                break
            time.sleep(LOOP_SLEEP)
            continue

        gray = cv2.cvtColor(detect_patch, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        score, loc, (tw, th), scale = best_match(gray, templates)
        x, y = loc
        detected = score >= MATCH_THRESHOLD

        now = time.time()
        if detected and (now - last_log_at) >= LOG_COOLDOWN:
            print(
                f"[{time.strftime('%H:%M:%S')}] ALERT DETECTED score={score:.3f} "
                f"loc=({DETECT_ROI['left'] + x},{DETECT_ROI['top'] + y})",
                flush=True,
            )
            last_log_at = now

        view = frame.copy()
        color = (0, 255, 0) if detected else (0, 255, 255)
        # Draw detection sub-area in blue on the large visualization window.
        cv2.rectangle(view, (dx1, dy1), (dx2, dy2), (255, 150, 0), 2)
        # Draw matched object inside the detection sub-area.
        cv2.rectangle(view, (dx1 + x, dy1 + y), (dx1 + x + tw, dy1 + y + th), color, 2)
        cv2.putText(
            view,
            f"{'DETECTED' if detected else 'WAIT'} score={score:.3f}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )
        cv2.putText(
            view,
            f"thr={MATCH_THRESHOLD:.2f} scale={scale:.2f}",
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
        cv2.imshow("DetectAlert Bot", view)

        if cv2.waitKey(1) & 0xFF == 27:
            print("Stopped by ESC.")
            break

        time.sleep(LOOP_SLEEP)

cv2.destroyAllWindows()
