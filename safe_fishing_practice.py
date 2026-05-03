import argparse
import sys
import time

import cv2
import keyboard
import mss
import numpy as np
import pyautogui


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safe local automation practice: detect bite color in ROI and react."
    )
    parser.add_argument("--left", type=int, default=300, help="ROI left screen coordinate")
    parser.add_argument("--top", type=int, default=200, help="ROI top screen coordinate")
    parser.add_argument("--width", type=int, default=500, help="ROI width")
    parser.add_argument("--height", type=int, default=350, help="ROI height")
    parser.add_argument(
        "--cooldown",
        type=float,
        default=1.2,
        help="Minimum seconds between reactions",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=850,
        help="Minimum yellow pixels to classify bite event",
    )
    parser.add_argument(
        "--show-debug",
        action="store_true",
        help="Show debug window with detected mask",
    )
    parser.add_argument(
        "--click",
        action="store_true",
        help="Perform mouse click on detection (default is dry-run)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log actions only (default behavior if --click not set)",
    )
    parser.add_argument(
        "--print-mouse",
        action="store_true",
        help="Print current mouse position repeatedly (for calibration)",
    )
    return parser.parse_args()


def print_mouse_loop() -> None:
    print("Move mouse to ROI corner(s). Press F8 to stop.")
    while not keyboard.is_pressed("F8"):
        x, y = pyautogui.position()
        sys.stdout.write(f"\rMouse: x={x:4d}, y={y:4d}")
        sys.stdout.flush()
        time.sleep(0.05)
    print("\nStopped.")


def detect_bite_yellow(frame_bgr: np.ndarray, min_pixels: int) -> tuple[bool, np.ndarray, int]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Range for the yellow bite signal in the demo app.
    lower = np.array([18, 110, 120], dtype=np.uint8)
    upper = np.array([42, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    pixel_count = int(cv2.countNonZero(mask))
    return pixel_count >= min_pixels, mask, pixel_count


def main() -> None:
    args = parse_args()

    if args.print_mouse:
        print_mouse_loop()
        return

    roi = {
        "left": args.left,
        "top": args.top,
        "width": args.width,
        "height": args.height,
    }
    simulate_only = not args.click or args.dry_run

    print("Safe Practice Automation")
    print(f"ROI: {roi}")
    print(f"Mode: {'DRY-RUN' if simulate_only else 'CLICK'}")
    print("Press F8 to stop.")

    with mss.mss() as sct:
        last_action_at = 0.0
        while not keyboard.is_pressed("F8"):
            raw = np.array(sct.grab(roi))
            frame = raw[:, :, :3]
            detected, mask, count = detect_bite_yellow(frame, args.threshold)
            now = time.time()

            if detected and (now - last_action_at) >= args.cooldown:
                if simulate_only:
                    print(f"[{time.strftime('%H:%M:%S')}] Bite detected (pixels={count}) -> REACT (simulated)")
                else:
                    pyautogui.click()
                    print(f"[{time.strftime('%H:%M:%S')}] Bite detected (pixels={count}) -> CLICK")
                last_action_at = now

            if args.show_debug:
                cv2.imshow("ROI", frame)
                cv2.imshow("YellowMask", mask)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            time.sleep(0.03)

    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()
