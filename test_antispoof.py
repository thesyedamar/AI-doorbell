"""
test_antispoof.py — Anti-Spoofing Live Diagnostic
===================================================
Tests Silent-Face models using correct scale-padded crops.

Controls:
    Q     — quit
    S     — save frame
    +/-   — raise/lower threshold
"""

import cv2
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from config import cfg
from vision.antispoof import AntiSpoofDetector

FONT         = cv2.FONT_HERSHEY_SIMPLEX
COLOR_GREEN  = (0, 220, 0)
COLOR_RED    = (0, 0, 220)
COLOR_YELLOW = (0, 220, 220)
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0, 0, 0)
COLOR_CYAN   = (220, 220, 0)
COLOR_GRAY   = (160, 160, 160)


def draw_bg_text(frame, text, pos, scale=0.65, color=COLOR_WHITE, thickness=2):
    x, y = pos
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, thickness)
    cv2.rectangle(frame, (x-4, y-th-4), (x+tw+4, y+bl+4), COLOR_BLACK, -1)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_score_bar(frame, score, threshold, x, y, width=300, height=25):
    cv2.rectangle(frame, (x, y), (x+width, y+height), (40, 40, 40), -1)
    fill  = int(min(score, 1.0) * width)
    color = COLOR_GREEN if score >= threshold else COLOR_RED
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x+fill, y+height), color, -1)
    tx = x + int(threshold * width)
    cv2.line(frame, (tx, y-5), (tx, y+height+5), COLOR_YELLOW, 2)
    cv2.rectangle(frame, (x, y), (x+width, y+height), COLOR_WHITE, 1)
    draw_bg_text(frame, "0.0", (x, y+height+18), 0.4, COLOR_GRAY, 1)
    draw_bg_text(frame, "1.0", (x+width-20, y+height+18), 0.4, COLOR_GRAY, 1)
    draw_bg_text(frame, f"T={threshold:.2f}", (tx-18, y-8), 0.4, COLOR_YELLOW, 1)


def main():
    import insightface

    print("\n" + "="*55)
    print("  AiDarwaza — Anti-Spoof Diagnostic (Silent-Face)")
    print("="*55)
    print("  REAL face  → confidence HIGH (green) >= 0.50")
    print("  FAKE/phone → confidence LOW  (red)  <  0.50")
    print("  +/- adjust threshold | S save frame | Q quit")
    print("="*55 + "\n")

    detector = AntiSpoofDetector()
    if not detector.enabled:
        print("ERROR: Models not loaded.")
        print("Run: python download_antispoof.py")
        return

    app = insightface.app.FaceAnalysis(
        name=cfg.vision.insightface_model,
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(960, 960))

    cap = cv2.VideoCapture(cfg.vision.camera_index)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    threshold     = cfg.vision.antispoof_real_threshold
    score_history = []
    debug_dir     = cfg.paths.debug_frames_dir
    debug_dir.mkdir(exist_ok=True)
    last_detail   = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        h, w    = display.shape[:2]
        faces   = app.get(frame)

        raw_conf = None
        is_real  = None

        if faces:
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            bbox = tuple(face.bbox)
            x1, y1, x2, y2 = [int(c) for c in bbox]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)

            # Use full frame + bbox for correct scale padding
            detail      = detector.predict_detailed_from_frame(frame, bbox)
            last_detail = detail
            raw_conf    = detail["confidence"]
            is_real     = detail["is_real"]

            score_history.append(raw_conf)
            if len(score_history) > 20:
                score_history.pop(0)
            smooth = float(np.mean(score_history))

            # Box color
            box_color = COLOR_GREEN if is_real else COLOR_RED
            cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 3)

            # REAL / FAKE label
            label = "REAL" if is_real else "FAKE"
            lx    = x1 + (x2-x1)//2 - 40
            draw_bg_text(display, label, (lx, min(h-10, y2+40)), 1.4, box_color, 3)

            # Scores above box
            draw_bg_text(display, f"Confidence: {raw_conf:.4f}", (x1, max(25, y1-55)), 0.65, COLOR_WHITE, 2)
            draw_bg_text(display, f"Smooth: {smooth:.4f}",       (x1, max(50, y1-28)), 0.65, COLOR_CYAN,  2)

            # Landmarks
            if face.kps is not None:
                for kp in face.kps:
                    cv2.circle(display, (int(kp[0]), int(kp[1])), 4, box_color, -1)

            # Score bar
            draw_score_bar(display, smooth, threshold, 20, h-70, width=350)

        else:
            score_history.clear()
            last_detail = {}
            draw_bg_text(display, "No face — move closer to camera", (20, h//2), 0.9, COLOR_YELLOW, 2)

        # Header
        cv2.rectangle(display, (0, 0), (w, 45), (20, 20, 20), -1)
        draw_bg_text(display, "AiDarwaza — Silent-Face Anti-Spoof", (10, 32), 0.75, COLOR_WHITE, 2)

        # Info panel
        px = w - 310
        cv2.rectangle(display, (px-8, 50), (w, 290), (20, 20, 20), -1)
        m1 = last_detail.get("model_1_score")
        m2 = last_detail.get("model_2_score")
        rows = [
            ("Threshold:", f"{threshold:.2f}", COLOR_YELLOW),
            ("Confidence:", f"{raw_conf:.4f}" if raw_conf is not None else "N/A",
             COLOR_GREEN if is_real else (COLOR_RED if is_real is False else COLOR_WHITE)),
            ("Model 1:", f"{m1:.4f}" if m1 is not None else "N/A", COLOR_CYAN),
            ("Model 2:", f"{m2:.4f}" if m2 is not None else "N/A", COLOR_CYAN),
            ("Result:", ("REAL" if is_real else "FAKE") if is_real is not None else "N/A",
             COLOR_GREEN if is_real else (COLOR_RED if is_real is False else COLOR_GRAY)),
            ("", "", COLOR_WHITE),
            ("[+/-]", "adjust threshold", COLOR_GRAY),
            ("[S]",   "save frame",       COLOR_GRAY),
        ]
        yi = 72
        for label, value, color in rows:
            if label:
                draw_bg_text(display, f"{label:<14}{value}", (px, yi), 0.55, color, 1)
            yi += 26

        # Bottom
        cv2.rectangle(display, (0, h-95), (500, h-55), (20,20,20), -1)
        draw_bg_text(display, "REAL face = HIGH confidence | FAKE/phone = LOW confidence",
                     (10, h-80), 0.48, COLOR_WHITE, 1)
        draw_bg_text(display, f"Current threshold: {threshold:.2f}  (REAL if confidence >= threshold)",
                     (10, h-58), 0.48, COLOR_YELLOW, 1)

        cv2.imshow("AiDarwaza — Anti-Spoof Diagnostic", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break
        elif key == ord("+") or key == ord("="):
            threshold = min(0.95, threshold + 0.05)
            print(f"Threshold → {threshold:.2f}")
        elif key == ord("-"):
            threshold = max(0.05, threshold - 0.05)
            print(f"Threshold → {threshold:.2f}")
        elif key == ord("s"):
            fname = debug_dir / f"spoof_{datetime.now().strftime('%H%M%S')}.jpg"
            cv2.imwrite(str(fname), display)
            print(f"Saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nFinal threshold: {threshold:.2f}")
    print(f"Add to .env: ANTISPOOF_THRESHOLD={threshold:.2f}")


if __name__ == "__main__":
    main()
