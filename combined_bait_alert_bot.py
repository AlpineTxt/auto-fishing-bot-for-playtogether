import time
import ctypes

import cv2
import keyboard
import mss
import numpy as np
import pyautogui

VIS_ROI = {"left": 1083, "top": 156, "width": 747, "height": 354}

BAIT_ROI = {"left": 1649, "top": 329, "width": 74, "height": 58}
ALERT_ROI = {"left": 1360, "top": 153, "width": 110, "height": 103}

BAIT_TEMPLATE_PATHS = [
    "detectbait.png",
    "bait.png",
    "bait.new.png",
]
ALERT_TEMPLATE_PATHS = [
    "dayalert.png",
    "alert 1.png",
    "alert 2.png",
    "alert3.png",
    "alert 4.png",
    "alert5.png",
]

SCALES = [0.80, 0.90, 1.00, 1.10, 1.20]

# Bait uses SQDIFF logic: lower is better.
BAIT_MAX_DIFF = 0.38
BAIT_CLICK_ENABLED = True
BAIT_REQUIRED_HITS = 1

# Alert uses direct matching: higher is better.
ALERT_MATCH_THRESHOLD = 0.62
REEL_CLICK_POINT = (1762, 447)  # center of reel area (1705,403) to (1820,491)
REEL_CLICK_BURST = 2
REEL_BURST_GAP_SECONDS = 0.04
REEL_CLICK_USE_BOTH_METHODS = True
FOLLOWUP_CLICK_POINT = (1619, 449)
FOLLOWUP_DELAY_SECONDS = 3.0
POST_FOLLOWUP_BAIT_DELAY_SECONDS = 2.0
DIRECT_BAIT_CLICK_POINT = (1686, 358)
CLICK_METHOD = "winapi"  # "winapi" or "pyautogui"

LOG_COOLDOWN = 0.15
LOOP_SLEEP = 0.01


def load_templates(paths, name):
    out = []
    for p in paths:
        t = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if t is None:
            print(f"[INFO] Missing {name} template skipped: {p}")
            continue
        out.append(cv2.equalizeHist(t))
    if not out:
        raise FileNotFoundError(f"No valid {name} templates loaded.")
    return out


def roi_local_in_vis(abs_roi, vis_roi, frame_shape):
    x1 = abs_roi["left"] - vis_roi["left"]
    y1 = abs_roi["top"] - vis_roi["top"]
    x2 = x1 + abs_roi["width"]
    y2 = y1 + abs_roi["height"]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_shape[1], x2)
    y2 = min(frame_shape[0], y2)
    return x1, y1, x2, y2


def best_match_sqdiff(gray_eq, templates):
    best_diff = 999.0
    best_loc = (0, 0)
    best_wh = (0, 0)
    best_scale = 1.0
    for tpl in templates:
        for s in SCALES:
            tw = max(8, int(tpl.shape[1] * s))
            th = max(8, int(tpl.shape[0] * s))
            if tw >= gray_eq.shape[1] or th >= gray_eq.shape[0]:
                continue
            interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
            t = cv2.resize(tpl, (tw, th), interpolation=interp)
            res = cv2.matchTemplate(gray_eq, t, cv2.TM_SQDIFF_NORMED)
            diff_min, _, min_loc, _ = cv2.minMaxLoc(res)
            if float(diff_min) < best_diff:
                best_diff = float(diff_min)
                best_loc = min_loc
                best_wh = (tw, th)
                best_scale = s
    return best_diff, best_loc, best_wh, best_scale


def best_match_score(gray_eq, templates):
    best_score = -1.0
    best_loc = (0, 0)
    best_wh = (0, 0)
    best_scale = 1.0
    for tpl in templates:
        for s in SCALES:
            tw = max(8, int(tpl.shape[1] * s))
            th = max(8, int(tpl.shape[0] * s))
            if tw >= gray_eq.shape[1] or th >= gray_eq.shape[0]:
                continue
            interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
            t = cv2.resize(tpl, (tw, th), interpolation=interp)
            res = cv2.matchTemplate(gray_eq, t, cv2.TM_CCOEFF_NORMED)
            _, max_score, _, max_loc = cv2.minMaxLoc(res)
            if float(max_score) > best_score:
                best_score = float(max_score)
                best_loc = max_loc
                best_wh = (tw, th)
                best_scale = s
    return best_score, best_loc, best_wh, best_scale


def click_winapi(x, y):
    user32 = ctypes.windll.user32
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    user32.SetCursorPos(int(x), int(y))
    user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def safe_click(x, y):
    if CLICK_METHOD == "winapi":
        click_winapi(x, y)
    else:
        pyautogui.click(x, y)


def reel_click():
    # Reel tap reliability: use both click methods when enabled.
    if REEL_CLICK_USE_BOTH_METHODS:
        click_winapi(REEL_CLICK_POINT[0], REEL_CLICK_POINT[1])
        pyautogui.click(REEL_CLICK_POINT[0], REEL_CLICK_POINT[1])
    else:
        safe_click(REEL_CLICK_POINT[0], REEL_CLICK_POINT[1])


bait_templates = load_templates(BAIT_TEMPLATE_PATHS, "bait")
alert_templates = load_templates(ALERT_TEMPLATE_PATHS, "alert")

print("Combined bait+alert bot running. Press F8 or ESC to stop.")
print("Press F6 to test reel click manually.")

last_alert_log_at = 0.0
pending_followup_click_at = None
bait_reclick_at = None
bait_clicked_this_detection = False
bait_click_blocked_until = 0.0
bait_hits = 0
phase = "WAIT_BAIT"
fish_count = 0
phase_started_at = time.time()

with mss.mss() as sct:
    cv2.namedWindow("Combined Bot", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Combined Bot", 60, 80)
    cv2.resizeWindow("Combined Bot", 900, 500)

    while True:
        if keyboard.is_pressed("f8"):
            print("Stopped by F8.")
            break
        if keyboard.is_pressed("f6"):
            safe_click(REEL_CLICK_POINT[0], REEL_CLICK_POINT[1])
            print(
                f"[{time.strftime('%H:%M:%S')}] MANUAL REEL CLICK ({REEL_CLICK_POINT[0]},{REEL_CLICK_POINT[1]})",
                flush=True,
            )
            time.sleep(0.25)

        frame = np.array(sct.grab(VIS_ROI))[:, :, :3]
        view = frame.copy()

        # -------- BAIT --------
        bx1, by1, bx2, by2 = roi_local_in_vis(BAIT_ROI, VIS_ROI, frame.shape)
        bait_patch = frame[by1:by2, bx1:bx2]
        bait_detected = False
        bait_diff = 999.0
        bait_scale = 1.0
        bait_loc = (0, 0)
        bait_wh = (0, 0)
        if bait_patch.size > 0:
            bait_gray = cv2.cvtColor(bait_patch, cv2.COLOR_BGR2GRAY)
            bait_gray = cv2.equalizeHist(bait_gray)
            bait_diff, bait_loc, bait_wh, bait_scale = best_match_sqdiff(bait_gray, bait_templates)
            bait_detected = bait_diff <= BAIT_MAX_DIFF

        now = time.time()
        if not bait_detected:
            bait_clicked_this_detection = False
            bait_hits = 0
        else:
            bait_hits += 1

        bait_color = (0, 255, 0) if bait_detected else (0, 255, 255)
        cv2.rectangle(view, (bx1, by1), (bx2, by2), (255, 150, 0), 2)
        if bait_patch.size > 0:
            bx, by = bait_loc
            bw, bh = bait_wh
            cv2.rectangle(view, (bx1 + bx, by1 + by), (bx1 + bx + bw, by1 + by + bh), bait_color, 2)

        # -------- ALERT --------
        ax1, ay1, ax2, ay2 = roi_local_in_vis(ALERT_ROI, VIS_ROI, frame.shape)
        alert_patch = frame[ay1:ay2, ax1:ax2]
        alert_detected = False
        alert_score = -1.0
        alert_scale = 1.0
        alert_loc = (0, 0)
        alert_wh = (0, 0)
        if alert_patch.size > 0:
            alert_gray = cv2.cvtColor(alert_patch, cv2.COLOR_BGR2GRAY)
            alert_gray = cv2.equalizeHist(alert_gray)
            alert_score, alert_loc, alert_wh, alert_scale = best_match_score(alert_gray, alert_templates)
            alert_detected = alert_score >= ALERT_MATCH_THRESHOLD

        if alert_detected and (now - last_alert_log_at) >= LOG_COOLDOWN:
            print(
                f"[{time.strftime('%H:%M:%S')}] ALERT DETECTED score={alert_score:.3f}",
                flush=True,
            )
            last_alert_log_at = now

        # Strict cycle:
        # WAIT_BAIT -> WAIT_ALERT -> WAIT_FOLLOWUP -> WAIT_BAIT
        if phase == "WAIT_BAIT":
            if (
                BAIT_CLICK_ENABLED
                and bait_detected
                and not bait_clicked_this_detection
                and now >= bait_click_blocked_until
                and bait_hits >= BAIT_REQUIRED_HITS
            ):
                bx, by = bait_loc
                bw, bh = bait_wh
                bait_click_x = BAIT_ROI["left"] + bx + (bw // 2)
                bait_click_y = BAIT_ROI["top"] + by + (bh // 2)
                safe_click(bait_click_x, bait_click_y)
                print(
                    f"[{time.strftime('%H:%M:%S')}] BAIT CLICK ({bait_click_x},{bait_click_y})",
                    flush=True,
                )
                bait_clicked_this_detection = True
                phase = "WAIT_ALERT"
                phase_started_at = now

        elif phase == "WAIT_ALERT":
            # Reel click instantly when alert appears.
            if alert_detected:
                for _ in range(REEL_CLICK_BURST):
                    reel_click()
                    time.sleep(REEL_BURST_GAP_SECONDS)
                fish_count += 1
                print(
                    f"[{time.strftime('%H:%M:%S')}] REEL CLICK ({REEL_CLICK_POINT[0]},{REEL_CLICK_POINT[1]}) | "
                    f"FISH CAUGHT #{fish_count}",
                    flush=True,
                )
                pending_followup_click_at = now + FOLLOWUP_DELAY_SECONDS
                phase = "WAIT_FOLLOWUP"
                phase_started_at = now

        elif phase == "WAIT_FOLLOWUP":
            if pending_followup_click_at is not None and now >= pending_followup_click_at:
                safe_click(FOLLOWUP_CLICK_POINT[0], FOLLOWUP_CLICK_POINT[1])
                print(
                    f"[{time.strftime('%H:%M:%S')}] FOLLOWUP CLICK ({FOLLOWUP_CLICK_POINT[0]},{FOLLOWUP_CLICK_POINT[1]})",
                    flush=True,
                )
                pending_followup_click_at = None
                bait_reclick_at = now + POST_FOLLOWUP_BAIT_DELAY_SECONDS
                phase = "WAIT_REBAIT"
                phase_started_at = now

        elif phase == "WAIT_REBAIT":
            if bait_reclick_at is not None and now >= bait_reclick_at:
                safe_click(DIRECT_BAIT_CLICK_POINT[0], DIRECT_BAIT_CLICK_POINT[1])
                print(
                    f"[{time.strftime('%H:%M:%S')}] BAIT RECLICK ({DIRECT_BAIT_CLICK_POINT[0]},{DIRECT_BAIT_CLICK_POINT[1]})",
                    flush=True,
                )
                bait_reclick_at = None
                bait_clicked_this_detection = True
                bait_hits = 0
                phase = "WAIT_ALERT"
                phase_started_at = now

        alert_color = (0, 255, 0) if alert_detected else (0, 255, 255)
        cv2.rectangle(view, (ax1, ay1), (ax2, ay2), (255, 150, 0), 2)
        if alert_patch.size > 0:
            ax, ay = alert_loc
            aw, ah = alert_wh
            cv2.rectangle(view, (ax1 + ax, ay1 + ay), (ax1 + ax + aw, ay1 + ay + ah), alert_color, 2)

        # -------- Overlay text --------
        cv2.putText(
            view,
            f"MODE: {phase} | FISH={fish_count}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            view,
            f"BAIT: {'DETECTED' if bait_detected else 'WAIT'} diff={bait_diff:.3f}",
            (8, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            bait_color,
            2,
        )
        cv2.putText(
            view,
            f"ALERT: {'DETECTED' if alert_detected else 'WAIT'} score={alert_score:.3f}",
            (8, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            alert_color,
            2,
        )
        cv2.putText(
            view,
            "Blue boxes are detection ROIs | ESC/F8 to stop",
            (8, 102),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 150, 0),
            2,
        )

        cv2.imshow("Combined Bot", view)
        if cv2.waitKey(1) & 0xFF == 27:
            print("Stopped by ESC.")
            break

        time.sleep(LOOP_SLEEP)

cv2.destroyAllWindows()
