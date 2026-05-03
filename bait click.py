import cv2
import mss
import numpy as np
import pyautogui

TEMPLATE_PATH = r"C:\Users\Lenovo\THEBot\fishing bot\2nd\bait.new.png"
ROI = {"left": 1649, "top": 329, "width": 74, "height": 58}
CLICK_POINT = (1686, 358)
THRESHOLD = 0.70
DO_CLICK = True
SHOW_WINDOW = True

template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
if template is None:
    raise FileNotFoundError(TEMPLATE_PATH)

with mss.mss() as sct:
    frame = np.array(sct.grab(ROI))[:, :, :3]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

result = cv2.matchTemplate(gray, template, cv2.TM_SQDIFF_NORMED)
min_val, _, min_loc, _ = cv2.minMaxLoc(result)
score = 1.0 - float(min_val)

print(f"score={score:.3f}")

if score >= THRESHOLD:
    print("DETECTED")
    if DO_CLICK:
        pyautogui.click(CLICK_POINT[0], CLICK_POINT[1])
        print(f"clicked at {CLICK_POINT}")
else:
    print("NOT DETECTED")

if SHOW_WINDOW:
    h, w = template.shape[:2]
    x, y = min_loc
    view = frame.copy()
    color = (0, 255, 0) if score >= THRESHOLD else (0, 255, 255)
    cv2.rectangle(view, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        view,
        f"score={score:.3f} thr={THRESHOLD:.2f}",
        (4, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        2,
    )
    cv2.imshow("Bait Detect (Single Shot)", view)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
