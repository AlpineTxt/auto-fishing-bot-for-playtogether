import time

import cv2
import keyboard
import mss
import numpy as np

ALERT_TEMPLATE_PATHS = [
    r"C:\Users\Lenovo\THEBot\fishing bot\2nd\Screenshot 2026-04-02 220339.png",
    r"C:\Users\Lenovo\THEBot\fishing bot\2nd\Screenshot 2026-04-02 203026.png",
    r"C:\Users\Lenovo\THEBot\fishing bot\2nd\Screenshot 2025-10-20 005703.png",
    r"C:\Users\Lenovo\THEBot\fishing bot\2nd\Screenshot 2025-10-20 012643.png",
    r"C:\Users\Lenovo\THEBot\fishing bot\2nd\alert 1.png",
    r"C:\Users\Lenovo\THEBot\fishing bot\2nd\alert 2.png",
    r"C:\Users\Lenovo\THEBot\fishing bot\2nd\alert3.png",
    r"C:\Users\Lenovo\THEBot\fishing bot\2nd\alert 4.png",
    r"C:\Users\Lenovo\THEBot\fishing bot\2nd\alert5.png",
]

ALERT_ROI = {"left": 1410, "top": 190, "width": 229, "height": 124}
THRESHOLD = 0.82
REQUIRED_HITS = 2
LOG_COOLDOWN = 0.1
LOOP_SLEEP = 0.01
SCALES = [0.85, 0.95, 1.00, 1.05, 1.15]


def load_templates(paths):
    loaded = []
    for path in paths:
        tpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if tpl is None:
            print(f"[INFO] Missing template skipped: {path}")
            continue
        loaded.append(cv2.equalizeHist(tpl))
    return loaded


def best_match(gray_eq, templates):
    best_score = -1.0
    best_loc = (0, 0)
    best_wh = (0, 0)
    for tpl in templates:
        for scale in SCALES:
            w = max(8, int(tpl.shape[1] * scale))
            h = max(8, int(tpl.shape[0] * scale))
            if w > gray_eq.shape[1] or h > gray_eq.shape[0]:
                continue
            resized = cv2.resize(
                tpl,
                (w, h),
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
            )
            res = cv2.matchTemplate(gray_eq, resized, cv2.TM_SQDIFF_NORMED)
            min_val, _, min_loc, _ = cv2.minMaxLoc(res)
            score = 1.0 - float(min_val)
            if score > best_score:
                best_score = score
                best_loc = min_loc
                best_wh = (w, h)
    return best_score, best_loc, best_wh


templates = load_templates(ALERT_TEMPLATE_PATHS)
if not templates:
    raise FileNotFoundError("No valid alert templates were loaded.")

print("Exclamation-only live detector. Press F8 or ESC to stop.")
cv2.namedWindow("Alert Detect", cv2.WINDOW_NORMAL)
cv2.moveWindow("Alert Detect", 60, 80)
cv2.resizeWindow("Alert Detect", 460, 360)

hit_count = 0
last_candidate_log_at = 0.0
last_detected_log_at = 0.0

with mss.mss() as sct:
    while True:
        if keyboard.is_pressed("f8"):
            print("Stopped by F8.")
            break

        frame = np.array(sct.grab(ALERT_ROI))[:, :, :3]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)

        score, loc, (w, h) = best_match(gray_eq, templates)
        x, y = loc

        if score >= THRESHOLD and score < 0.999:
            hit_count += 1
        else:
            hit_count = 0

        detected = hit_count >= REQUIRED_HITS
        now = time.time()

        if score >= THRESHOLD and (now - last_candidate_log_at) >= LOG_COOLDOWN:
            print(
                f"[{time.strftime('%H:%M:%S')}] CANDIDATE score={score:.3f} hit={hit_count}/{REQUIRED_HITS}",
                flush=True,
            )
            last_candidate_log_at = now

        if detected and (now - last_detected_log_at) >= LOG_COOLDOWN:
            print(
                f"[{time.strftime('%H:%M:%S')}] ALERT DETECTED "
                f"score={score:.3f} loc=({ALERT_ROI['left'] + x},{ALERT_ROI['top'] + y})",
                flush=True,
            )
            last_detected_log_at = now

        view = frame.copy()
        color = (0, 255, 0) if detected else (0, 255, 255)
        cv2.rectangle(view, (x, y), (x + w, y + h), color, 2)
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
            f"thr={THRESHOLD:.2f} hit={hit_count}",
            (8, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

        cv2.imshow("Alert Detect", view)
        if cv2.waitKey(1) & 0xFF == 27:
            print("Stopped by ESC.")
            break

        time.sleep(LOOP_SLEEP)

cv2.destroyAllWindows()
