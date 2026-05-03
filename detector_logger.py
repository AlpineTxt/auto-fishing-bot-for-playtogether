import time
from pathlib import Path

import cv2
from mss import mss
from numpy import asarray

# ----------------- Config -----------------
ROI = {"left": 1292, "top": 418, "width": 1263, "height": 582}

TEMPLATE_PATHS = {
    "day": r"C:\Users\Lenovo\THEBot\fishing bot\2nd\alert 4.png",
    "night": r"C:\Users\Lenovo\THEBot\fishing bot\2nd\alert 1.png",
}

THRESHOLD = 0.82
LOG_FILE = Path(r"C:\Users\Lenovo\THEBot\fishing bot\2nd\detection_log.txt")
LOG_COOLDOWN_SECONDS = 0.2
DISPLAY_WINDOW = True
DISPLAY_SIZE = (760, 430)
LOOP_SLEEP_SECONDS = 0.01
# ------------------------------------------


def load_templates(paths: dict[str, str]) -> dict[str, tuple]:
    loaded = {}
    for mode, path in paths.items():
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Template not found for '{mode}': {path}")
        h, w = image.shape
        loaded[mode] = (image, w, h)
    return loaded


def match_best(gray_frame, templates):
    best = {
        "mode": None,
        "score": -1.0,
        "loc": (0, 0),
        "size": (0, 0),
    }
    for mode, (templ, w, h) in templates.items():
        result = cv2.matchTemplate(gray_frame, templ, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best["score"]:
            best["mode"] = mode
            best["score"] = float(max_val)
            best["loc"] = max_loc
            best["size"] = (w, h)
    return best


def append_log(line: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    templates = load_templates(TEMPLATE_PATHS)
    print("Detector started. Press ESC to stop.")
    print(f"Logging detections to: {LOG_FILE}")

    if DISPLAY_WINDOW:
        cv2.namedWindow("Detector Logger", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detector Logger", DISPLAY_SIZE[0], DISPLAY_SIZE[1])

    last_log_at = 0.0

    with mss() as sct:
        while True:
            frame_bgra = asarray(sct.grab(ROI))
            gray = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2GRAY)
            frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

            best = match_best(gray, templates)
            x, y = best["loc"]
            w, h = best["size"]
            detected = best["score"] >= THRESHOLD

            now = time.time()
            if detected and (now - last_log_at) >= LOG_COOLDOWN_SECONDS:
                screen_x = ROI["left"] + x
                screen_y = ROI["top"] + y
                line = (
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"DETECTED mode={best['mode']} score={best['score']:.3f} "
                    f"loc=({screen_x},{screen_y})"
                )
                print(line, flush=True)
                append_log(line)
                last_log_at = now

            if DISPLAY_WINDOW:
                color = (0, 255, 0) if detected else (0, 255, 255)
                cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame_bgr,
                    f"{'DETECTED' if detected else 'WAIT'} mode={best['mode']}",
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                )
                cv2.putText(
                    frame_bgr,
                    f"score={best['score']:.3f} thr={THRESHOLD:.2f}",
                    (8, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    color,
                    2,
                )
                cv2.imshow("Detector Logger", frame_bgr)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            time.sleep(LOOP_SLEEP_SECONDS)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
