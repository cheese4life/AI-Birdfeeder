# Smart Birdhouse Motion Detection

# Frame differencing against a periodically refreshed baseline.
# On trigger: saves high-quality JPEG + JSON metadata sidecar.
# Tuned for zero false negatives. Bedrock handles filtering.


import cv2
import json
import time
import uuid
import sys
import signal
from datetime import datetime, timezone
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.json"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def setup_camera(cfg):
    cam = cv2.VideoCapture(cfg["camera_index"])
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["resolution"][0])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["resolution"][1])
    cam.set(cv2.CAP_PROP_FPS, 30)
    if not cam.isOpened():
        print("ERROR: Cannot open camera", flush=True)
        sys.exit(1)
    # warm up — first few frames are often garbage
    for _ in range(10):
        cam.read()
    return cam


def grab_baseline(cam, det_cfg):
    """Capture and preprocess a baseline frame."""
    ok, frame = cam.read()
    if not ok:
        return None, None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (det_cfg["blur_kernel"], det_cfg["blur_kernel"]), 0)
    return frame, gray


def get_roi_slice(shape, roi):
    """Convert fractional ROI [x1, y1, x2, y2] to pixel slices."""
    h, w = shape[:2]
    x1 = int(roi[0] * w)
    y1 = int(roi[1] * h)
    x2 = int(roi[2] * w)
    y2 = int(roi[3] * h)
    return slice(y1, y2), slice(x1, x2)


def detect_motion(gray_frame, baseline_gray, det_cfg, roi_slices):
    """Returns (triggered: bool, changed_pixels: int)."""
    diff = cv2.absdiff(gray_frame[roi_slices], baseline_gray[roi_slices])
    _, thresh = cv2.threshold(diff, det_cfg["threshold"], 255, cv2.THRESH_BINARY)
    changed = cv2.countNonZero(thresh)
    return changed >= det_cfg["min_changed_pixels"], changed


def save_capture(frame, capture_cfg):
    """Save JPEG + JSON sidecar. Returns (image_path, meta_path, meta_dict)."""
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    capture_id = str(uuid.uuid4())

    out_dir = Path(capture_cfg["output_dir"]) / date_str
    out_dir.mkdir(parents=True, exist_ok=True)

    img_path = out_dir / f"{capture_id}.jpg"
    meta_path = out_dir / f"{capture_id}.json"

    cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, capture_cfg["jpeg_quality"]])

    meta = {
        "uuid": capture_id,
        "timestamp": now.isoformat(),
        "date": date_str,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    return img_path, meta_path, meta


def run():
    cfg = load_config()
    det_cfg = cfg["detection"]
    cap_cfg = cfg["capture"]

    print("Starting detector...", flush=True)
    cam = setup_camera(cfg)

    print("Capturing baseline...", flush=True)
    baseline_raw, baseline_gray = grab_baseline(cam, det_cfg)
    if baseline_gray is None:
        print("ERROR: Failed to capture baseline", flush=True)
        sys.exit(1)

    roi_slices = get_roi_slice(baseline_gray.shape, det_cfg["roi"])
    baseline_time = time.monotonic()
    last_trigger = 0
    trigger_count = 0

    print(f"Detector running — ROI: {det_cfg['roi']}, "
          f"threshold: {det_cfg['threshold']}, "
          f"min_pixels: {det_cfg['min_changed_pixels']}, "
          f"cooldown: {det_cfg['cooldown_seconds']}s", flush=True)

    while True:
        ok, frame = cam.read()
        if not ok:
            print("WARN: Frame read failed, retrying...", flush=True)
            time.sleep(0.5)
            continue

        now = time.monotonic()

        if now - baseline_time > det_cfg["baseline_refresh_seconds"]:
            print("Refreshing baseline...", flush=True)
            baseline_raw, baseline_gray = grab_baseline(cam, det_cfg)
            baseline_time = now
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (det_cfg["blur_kernel"], det_cfg["blur_kernel"]), 0)

        triggered, changed = detect_motion(gray, baseline_gray, det_cfg, roi_slices)

        if triggered and (now - last_trigger) > det_cfg["cooldown_seconds"]:
            trigger_count += 1
            last_trigger = now
            img_path, meta_path, meta = save_capture(frame, cap_cfg)
            print(f"[TRIGGER #{trigger_count}] {changed} px changed — "
                  f"saved {img_path.name} @ {meta['timestamp']}", flush=True)

        # ~10 fps to limit CPU strain
        time.sleep(0.1)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    run()
