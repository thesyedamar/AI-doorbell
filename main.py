"""
main.py — AiDarwaza Main Orchestrator
======================================
Full system with persistent real-time overlay.
Overlay stays visible until face leaves frame.
Press Q to quit.
"""

import cv2
import time
import logging
import numpy as np
from datetime import datetime
from pathlib import Path

from config import cfg
from vision.capture import FaceCamera
from vision.antispoof import AntiSpoofDetector
from vision.recognition import extract_and_recognize, add_face_to_db
from vision.audio_capture import record_audio
from vision.speech_to_text import transcribe_audio
from vision.audio_validator import validate_audio
from vision.speaker import play_audio
from vision.db_tools import save_event
from nlp.dialogue_manager import handle_dialogue, init_state
from nlp.identity_extractor import extract_identity
from nlp.intent_router import route_intent
from nlp.response_mapper import get_response_path
from llm.ollama_client import is_ollama_running
from plugins import loader as plugin_loader
from plugins.base import VisitorEvent

logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Overlay colors (BGR)
# ---------------------------------------------------------------------------
COLOR_GREEN  = (0,   220,  0)
COLOR_RED    = (0,   0,    220)
COLOR_ORANGE = (0,   165,  255)
COLOR_YELLOW = (0,   220,  220)
COLOR_WHITE  = (255, 255,  255)
COLOR_GRAY   = (160, 160,  160)
COLOR_CYAN   = (220, 220,  0)
COLOR_BLACK  = (0,   0,    0)
FONT         = cv2.FONT_HERSHEY_SIMPLEX


def draw_text_bg(frame, text, pos, scale=0.55, color=COLOR_WHITE, thickness=1):
    x, y = pos
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, thickness)
    cv2.rectangle(frame, (x-3, y-th-3), (x+tw+3, y+bl+3), COLOR_BLACK, -1)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray,
    last_face,
    last_status: str,
    last_blur: float,
    last_spoof: dict,
    last_recog: dict,
    overlay_state: dict,
    face_visible: bool,
) -> np.ndarray:
    """
    Draw persistent overlay. Values stay visible until face leaves frame.
    face_visible controls whether to show the bounding box.
    """
    h, w = frame.shape[:2]

    # ---- Top status bar ----
    cv2.rectangle(frame, (0, 0), (w, 42), (15, 15, 15), -1)
    status_text = overlay_state.get("status", "Scanning...")
    status_color = {
        "Scanning...":        COLOR_GRAY,
        "Stabilizing...":     COLOR_YELLOW,
        "SPOOF DETECTED":     COLOR_RED,
        "Too blurry":         COLOR_ORANGE,
        "Face too small":     COLOR_ORANGE,
        "Cooldown active":    COLOR_GRAY,
        "Waiting for visitor to leave...": COLOR_YELLOW,
    }.get(status_text, COLOR_GREEN)

    draw_text_bg(frame, f"AiDarwaza  |  {status_text}",
                 (10, 28), 0.7, status_color, 2)

    # ---- Face bounding box (only when face visible) ----
    if face_visible and last_face is not None:
        x1, y1, x2, y2 = [int(c) for c in last_face.bbox]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)

        is_known  = last_recog.get("is_known", False)
        is_real   = last_spoof.get("is_real", None)
        box_color = COLOR_GREEN if (is_known and is_real) else (COLOR_RED if is_real is False else COLOR_CYAN)

        # Corner brackets
        cl = 18; ct = 3
        cv2.line(frame, (x1, y1), (x1+cl, y1), box_color, ct)
        cv2.line(frame, (x1, y1), (x1, y1+cl), box_color, ct)
        cv2.line(frame, (x2, y1), (x2-cl, y1), box_color, ct)
        cv2.line(frame, (x2, y1), (x2, y1+cl), box_color, ct)
        cv2.line(frame, (x1, y2), (x1+cl, y2), box_color, ct)
        cv2.line(frame, (x1, y2), (x1, y2-cl), box_color, ct)
        cv2.line(frame, (x2, y2), (x2-cl, y2), box_color, ct)
        cv2.line(frame, (x2, y2), (x2, y2-cl), box_color, ct)

        # Name label
        name = last_recog.get("name", "Unknown")
        draw_text_bg(frame, name, (x1, max(20, y1-10)), 0.7, box_color, 2)

        # Det score
        det = getattr(last_face, "det_score", 0.0)
        draw_text_bg(frame, f"Det: {det:.2f}", (x1, max(42, y1+16)), 0.5, COLOR_GRAY, 1)

        # Landmarks
        if last_face.kps is not None:
            for kp in last_face.kps:
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, box_color, -1)

    # ---- Right metrics panel (PERSISTENT — stays even when face gone) ----
    px = w - 185
    panel_h = 210
    cv2.rectangle(frame, (px-6, 50), (w, 50+panel_h), (15, 15, 15), -1)

    def metric(label, value, color, y):
        draw_text_bg(frame, label, (px, y), 0.52, COLOR_GRAY, 1)
        draw_text_bg(frame, value, (px+65, y), 0.52, color, 1)

    # Blur
    blur_ok = last_blur >= cfg.vision.blur_threshold if last_blur > 0 else None
    metric("BLUR:", f"{last_blur:.1f} {'OK' if blur_ok else 'LOW' if blur_ok is not None else 'N/A'}",
           COLOR_GREEN if blur_ok else (COLOR_ORANGE if blur_ok is not None else COLOR_GRAY), 72)

    # Det score
    det_score = getattr(last_face, "det_score", 0.0) if last_face else 0.0
    det_ok = det_score >= cfg.vision.min_det_score
    metric("DET:", f"{det_score:.2f} {'OK' if det_ok else 'LOW'}" if last_face else "N/A",
           COLOR_GREEN if det_ok else COLOR_RED, 97)

    # Spoof
    is_real   = last_spoof.get("is_real", None)
    spoof_conf= last_spoof.get("score", 0.0)
    if is_real is None:
        metric("SPOOF:", "N/A", COLOR_GRAY, 122)
    elif is_real:
        metric("SPOOF:", f"REAL {spoof_conf:.2f}", COLOR_GREEN, 122)
    else:
        metric("SPOOF:", f"FAKE {spoof_conf:.2f}", COLOR_RED, 122)

    # Recognition
    is_known = last_recog.get("is_known", None)
    if is_known is None:
        metric("RECOG:", "N/A", COLOR_GRAY, 147)
    elif is_known:
        metric("RECOG:", "KNOWN", COLOR_GREEN, 147)
    else:
        metric("RECOG:", "UNKNOWN", COLOR_CYAN, 147)

    # Status
    stat_color = COLOR_GREEN if last_status == "READY" else COLOR_YELLOW
    metric("STATUS:", last_status, stat_color, 172)

    # Area
    if last_face is not None:
        bx1, by1, bx2, by2 = [int(c) for c in last_face.bbox]
        area = (bx2-bx1)*(by2-by1)
        area_ok = area >= cfg.vision.min_face_area
        metric("AREA:", f"{area} {'OK' if area_ok else 'SMALL'}", COLOR_GREEN if area_ok else COLOR_ORANGE, 197)
    else:
        metric("AREA:", "N/A", COLOR_GRAY, 197)

    # ---- Bottom dialogue bar (PERSISTENT) ----
    stage     = overlay_state.get("stage", "")
    transcript= overlay_state.get("transcript", "")
    if stage or transcript:
        cv2.rectangle(frame, (0, h-52), (w, h), (15, 15, 15), -1)
        if stage:
            retry = overlay_state.get("retry_count", 0)
            draw_text_bg(frame, f"Stage: {stage}  Retry: {retry}",
                         (10, h-33), 0.55, COLOR_YELLOW, 1)
        if transcript:
            short = transcript[:65] + "..." if len(transcript) > 65 else transcript
            draw_text_bg(frame, f"STT: {short}", (10, h-10), 0.5, COLOR_WHITE, 1)

    return frame


def compute_blur(frame, face) -> float:
    if face is None or frame is None:
        return 0.0
    try:
        x1, y1, x2, y2 = [int(c) for c in face.bbox]
        crop = frame[max(0,y1):y2, max(0,x1):x2]
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def run_startup_checks() -> None:
    logger.info("\n" + cfg.summary())
    cfg.validate(strict=False)
    if not is_ollama_running():
        logger.warning("Ollama not running — intent fallback disabled.")
    plugin_loader.load_all(cfg.plugins.enabled_plugins)


# ---------------------------------------------------------------------------
# Dialogue
# ---------------------------------------------------------------------------

def run_dialogue(
    face_status, face_name, face_index,
    embedding, face_crop, frame,
    overlay_state,
) -> dict:
    state       = init_state()
    retry_count = 0
    spoken_name = ""
    intent      = ""
    intent_conf = 0.0
    transcript  = ""
    entities    = {}
    start_time  = time.time()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    crop_name = f"{face_name.replace(' ', '_')}_{timestamp}.jpg"
    crop_path = cfg.paths.faces_dir / crop_name
    try:
        cv2.imwrite(str(crop_path), face_crop)
    except Exception as e:
        logger.warning(f"Could not save face crop: {e}")
        crop_path = Path("")

    plugin_loader.dispatch("on_visitor_identified", VisitorEvent(
        face_index=face_index, name=face_name,
        is_known=(face_status == "Known"),
    ))

    while True:
        overlay_state["status"]     = "Dialogue active"
        overlay_state["stage"]      = state.get("stage", "")
        overlay_state["name"]       = face_name
        overlay_state["transcript"] = transcript

        result       = handle_dialogue(
            dialogue_state=state, face_status=face_status,
            face_name=face_name, intent=intent,
            intent_confidence=intent_conf, entities=entities,
            retry_count=retry_count,
        )
        response_key = result["response_key"]
        next_state   = result["next_state"]
        retry_count  = result["retry_count"]
        done         = result["done"]

        audio_path = get_response_path(response_key)
        if audio_path:
            play_audio(audio_path)
        else:
            logger.warning(f"No audio for key: {response_key}")

        if done:
            state["stage"] = next_state
            break

        state["stage"] = next_state

        if next_state in ("ASK_NAME", "ASK_INTENT"):
            overlay_state["stage"] = next_state

            audio_file = record_audio("visitor")
            if not audio_file:
                retry_count = min(retry_count + 1, cfg.nlp.max_dialogue_retries)
                continue

            if not validate_audio(audio_file):
                retry_count = min(retry_count + 1, cfg.nlp.max_dialogue_retries)
                continue

            transcript = transcribe_audio(audio_file)
            overlay_state["transcript"] = transcript
            logger.info(f"Transcript: '{transcript}'")

            if not transcript:
                retry_count = min(retry_count + 1, cfg.nlp.max_dialogue_retries)
                continue

            if next_state == "ASK_NAME":
                id_result = extract_identity(transcript)
                if id_result["valid"]:
                    spoken_name      = id_result["name"]
                    entities["name"] = spoken_name
                    if face_status == "Unknown" and spoken_name and embedding is not None:
                        face_index  = add_face_to_db(face_crop, embedding, spoken_name)
                        face_name   = spoken_name
                        face_status = "Known"
                        overlay_state["name"] = face_name

            elif next_state == "ASK_INTENT":
                intent_result = route_intent(transcript)
                intent        = intent_result["intent"]
                intent_conf   = intent_result["confidence"]
                overlay_state["retry_count"] = retry_count
                if not spoken_name and intent_result.get("name"):
                    spoken_name      = intent_result["name"]
                    entities["name"] = spoken_name

    duration_ms = int((time.time() - start_time) * 1000)
    conflict    = 0
    if face_name and spoken_name and face_name != "Unknown" and face_name != spoken_name:
        conflict = 1

    event_id = save_event(
        name=face_name, face_index=face_index,
        image_path=str(crop_path), speech_text=transcript,
        intent=intent, spoken_name=spoken_name,
        conflict=conflict, duration_ms=duration_ms,
    )

    summary = {
        "event_id": event_id, "face_status": face_status,
        "face_name": face_name, "spoken_name": spoken_name,
        "intent": intent, "intent_conf": intent_conf,
        "transcript": transcript, "conflict": conflict,
        "duration_ms": duration_ms, "crop_path": str(crop_path),
    }
    logger.info(f"Dialogue complete | {summary}")

    plugin_loader.dispatch("on_dialogue_complete", VisitorEvent(
        face_index=face_index, name=face_name,
        is_known=(face_status == "Known"),
        intent=intent, intent_confidence=intent_conf,
        spoken_name=spoken_name, transcript=transcript,
        face_crop_path=str(crop_path),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        conflict=bool(conflict),
    ))
    return summary


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    run_startup_checks()
    logger.info("Initializing components...")

    camera    = FaceCamera()
    antispoof = AntiSpoofDetector()

    # Persistent overlay state — survives between frames
    overlay_state = {
        "status":      "Scanning...",
        "name":        "",
        "stage":       "",
        "transcript":  "",
        "retry_count": 0,
    }

    # Persistent display values — never reset until face gone for 2+ seconds
    last_face   = None
    last_blur   = 0.0
    last_spoof  = {"is_real": None, "score": 0.0}
    last_recog  = {"is_known": None, "name": ""}
    last_status = "Scanning..."
    face_visible= False

    # Runtime state
    last_interaction_time = 0.0
    waiting_for_disappear = False
    disappear_start       = None
    frame_count           = 0

    plugin_loader.dispatch("on_system_startup")
    logger.info("AiDarwaza running. Press Q to quit.")

    try:
        while True:
            frame_count += 1

            ret, raw_frame = camera.cap.read()
            if not ret or raw_frame is None:
                time.sleep(0.05)
                continue

            # Build display frame with persistent overlay
            display = raw_frame.copy()
            display = draw_overlay(
                display, last_face, last_status,
                last_blur, last_spoof, last_recog,
                overlay_state, face_visible,
            )

            if cfg.system.show_preview_window:
                cv2.imshow(cfg.system.preview_window_title, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Q pressed — shutting down.")
                    break

            # Process every Nth frame
            if frame_count % cfg.system.frame_skip != 0:
                continue

            # Face detection
            frame, face, status = camera.read()
            last_status = status

            # Update blur
            if face is not None:
                last_blur    = compute_blur(raw_frame, face)
                last_face    = face
                face_visible = True
            else:
                # Face gone — keep last values visible but mark as not visible
                face_visible = False

            # ---- Waiting for visitor to leave ----
            if waiting_for_disappear:
                overlay_state["status"] = "Waiting for visitor to leave..."
                if face is None:
                    if disappear_start is None:
                        disappear_start = time.time()
                    elif time.time() - disappear_start >= cfg.vision.disappear_seconds:
                        logger.info("Visitor left — resetting.")
                        waiting_for_disappear = False
                        disappear_start       = None
                        camera.reset_stability()
                        # Clear overlay only when visitor fully gone
                        last_face    = None
                        last_blur    = 0.0
                        last_spoof   = {"is_real": None, "score": 0.0}
                        last_recog   = {"is_known": None, "name": ""}
                        last_status  = "Scanning..."
                        face_visible = False
                        overlay_state.update({
                            "status": "Scanning...", "name": "",
                            "stage": "", "transcript": "", "retry_count": 0,
                        })
                else:
                    disappear_start = None
                continue

            if face is None or status not in ("READY", "STABILIZING"):
                if face is None:
                    overlay_state["status"] = "Scanning..."
                elif status == "BLUR":
                    overlay_state["status"] = "Too blurry"
                elif status == "SMALL":
                    overlay_state["status"] = "Face too small"
                elif status == "LANDMARK":
                    overlay_state["status"] = "Look at camera"
                elif status == "STABILIZING":
                    overlay_state["status"] = "Stabilizing..."
                continue

            if status == "STABILIZING":
                overlay_state["status"] = "Stabilizing..."
                continue

            # Cooldown
            now = time.time()
            if now - last_interaction_time < cfg.system.cooldown_seconds:
                remaining = cfg.system.cooldown_seconds - (now - last_interaction_time)
                overlay_state["status"] = f"Cooldown active"
                continue

            # Anti-spoof
            is_real, spoof_score = antispoof.predict_from_frame(raw_frame, tuple(face.bbox))
            last_spoof = {"is_real": is_real, "score": spoof_score}

            if not is_real:
                overlay_state["status"] = "SPOOF DETECTED"
                logger.warning(f"Spoof! score={spoof_score:.3f}")
                continue

            # Recognition
            face_status, face_name, face_index, embedding, crop = \
                extract_and_recognize(raw_frame, face)

            last_recog = {"is_known": (face_status == "Known"), "name": face_name}
            overlay_state["status"] = f"{'Known' if face_status == 'Known' else 'Unknown'}: {face_name}"
            overlay_state["name"]   = face_name

            plugin_loader.dispatch("on_face_detected", raw_frame, face, is_real)
            logger.info(f"Interaction | status={face_status} name={face_name}")

            summary = run_dialogue(
                face_status, face_name, face_index,
                embedding, crop, raw_frame, overlay_state,
            )

            last_interaction_time = time.time()
            waiting_for_disappear = True
            disappear_start       = None
            plugin_loader.dispatch("on_cooldown_start", face_index)

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        logger.info("Shutting down...")
        camera.release()
        cv2.destroyAllWindows()
        plugin_loader.unload_all()
        logger.info("AiDarwaza stopped cleanly.")


if __name__ == "__main__":
    main()
