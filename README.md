# Play Together Auto-Fishing Bot 🎣

A high-performance, automated fishing bot for the game **Play Together**, using computer vision to detect bites and reel in fish automatically.

## ✨ Features
- **Intelligent Detection**: Uses OpenCV template matching with histogram equalization to detect bait and alert icons even in varying lighting conditions (Day/Night).
- **Multi-Scale Support**: Scans at multiple scales (0.8x to 1.2x) to handle different screen resolutions and UI sizes.
- **State Machine Logic**: Orchestrates the full fishing cycle:
    1. **WAIT_BAIT**: Detects and clicks the bait to start fishing.
    2. **WAIT_ALERT**: Monitors the "!" alert icon with high-speed screen capture.
    3. **WAIT_FOLLOWUP**: Handles the catch confirmation and loot screen.
    4. **WAIT_REBAIT**: Automatically prepares the next cast.
- **Dual Input Methods**: Supports both `PyAutoGUI` and `WinAPI` (ctypes) for mouse events to ensure compatibility and bypass certain input restrictions.
- **Real-time Monitoring**: Provides a live visual overlay showing detection ROIs (Regions of Interest) and confidence scores.

## 🛠 Prerequisites
Install the required dependencies:
```bash
pip install opencv-python mss numpy pyautogui keyboard
```

## 🎮 How to Use
1. **Setup**: Open **Play Together** on your PC (works best with screen mirroring or emulators).
2. **Configuration**: Adjust the `VIS_ROI`, `BAIT_ROI`, and `ALERT_ROI` in `combined_bait_alert_bot.py` if your screen resolution differs from the default (1080p).
3. **Run**: Execute the bot:
   ```bash
   python combined_bait_alert_bot.py
   ```
4. **Controls**:
   - **F8**: Toggle Bot Start/Stop.
   - **F6**: Manual Reel Click Test.
   - **ESC**: Exit the application.

## 📄 License
This project is licensed under the MIT License.
