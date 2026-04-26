# Smart Birdhouse --> Combined web server, live stream, motion + bird detection.
# Architecture:
#  camera_thread: reads raw MJPEG from USB cam via ffmpeg (zero decode for stream)
#  motion_thread: frame-differencing on decoded frames, saves triggers
#  bird_thread: MobileNet-SSD bird detection, publishes overlay JPEG
#  Flask: /stream (raw), /stream_detect (overlay), /api/triggers, etc.
#  Note: Web design made with Claude Opus 4.6
# Please refer to AI-disclosure.md to learn more about how AI assisted with development


import json
import subprocess
import time
import uuid as uuid_mod
import cv2
import numpy as np
import boto3
from datetime import date, datetime, timezone
from pathlib import Path
from threading import Thread, Condition, Lock
from queue import Queue

from flask import Flask, Response, render_template, jsonify, request
import functools
import traceback
import os

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
RESULTS_DIR = CACHE_DIR / "results"
IMAGES_DIR = CACHE_DIR / "images"
CONFIG_PATH = BASE_DIR / "config.json"
MODEL_DIR = BASE_DIR

# ── Shared state ──
raw_jpeg = None           
raw_cond = Condition()

detect_jpeg = None        
detect_cond = Condition()

raw_frame_bgr = None      
frame_lock = Lock()

trigger_log = []          
upload_queue = Queue()    
ai_debug_log = []        
pipeline_status = {"state": "idle", "last_bird": None, "last_ts": None}  

thread_health = {}  
def resilient_thread(name, restart_delay=3):
    """Decorator: catch all exceptions, log, restart the thread function after a delay."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            thread_health[name] = {"alive": True, "last_heartbeat": time.time(), "restarts": 0, "error": None}
            while True:
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    thread_health[name]["alive"] = False
                    thread_health[name]["error"] = str(e)
                    thread_health[name]["restarts"] += 1
                    print(f"[CRASH] {name} died: {e}", flush=True)
                    traceback.print_exc()
                    time.sleep(restart_delay)
                    print(f"[RESTART] {name} (restart #{thread_health[name]['restarts']})", flush=True)
                    thread_health[name]["alive"] = True
                    thread_health[name]["error"] = None
        return wrapper
    return decorator

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
]
BIRD_CLASS_ID = 3


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def _start_ffmpeg(cfg):
    w, h = cfg["resolution"]
    return subprocess.Popen(
        [
            "ffmpeg", "-f", "v4l2", "-input_format", "mjpeg",
            "-video_size", f"{w}x{h}", "-framerate", "30",
            "-i", f"/dev/video{cfg['camera_index']}",
            "-c:v", "copy", "-f", "mjpeg", "-flush_packets", "1", "-",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0,
    )



@resilient_thread("camera")
def camera_thread():
    global raw_jpeg, raw_frame_bgr

    cfg = load_config()
    cam_dev = f"/dev/video{cfg['camera_index']}"

    backoff = 1
    while not os.path.exists(cam_dev):
        print(f"[CAMERA] Waiting for {cam_dev}… (retry in {backoff}s)", flush=True)
        time.sleep(backoff)
        backoff = min(backoff * 2, 30)

    proc = _start_ffmpeg(cfg)
    buf = b""
    n = 0
    ffmpeg_backoff = 1

    while True:
        chunk = proc.stdout.read(8192)
        if not chunk:
            proc.wait()
            print(f"ffmpeg died, restarting in {ffmpeg_backoff}s…", flush=True)
            time.sleep(ffmpeg_backoff)
            ffmpeg_backoff = min(ffmpeg_backoff * 2, 30)
            # Check device still exists
            if not os.path.exists(cam_dev):
                print(f"[CAMERA] {cam_dev} gone — waiting…", flush=True)
                while not os.path.exists(cam_dev):
                    time.sleep(2)
                print(f"[CAMERA] {cam_dev} reappeared", flush=True)
            proc = _start_ffmpeg(cfg)
            buf = b""
            continue
        ffmpeg_backoff = 1  

        buf += chunk
        while True:
            start = buf.find(b"\xff\xd8")
            if start == -1:
                buf = b""
                break
            end = buf.find(b"\xff\xd9", start + 2)
            if end == -1:
                buf = buf[start:]
                break

            jpeg_bytes = buf[start : end + 2]
            buf = buf[end + 2 :]
            n += 1

            # publish raw JPEG (no decode)
            with raw_cond:
                raw_jpeg = jpeg_bytes
                raw_cond.notify_all()

            if n % 3 == 0:
                arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is not None:
                    with frame_lock:
                        raw_frame_bgr = bgr



@resilient_thread("motion")
def motion_thread():
    cfg = load_config()
    det = cfg["detection"]
    cap_cfg = cfg["capture"]
    blur_k = det["blur_kernel"]

    while raw_frame_bgr is None:
        time.sleep(0.5)

    with frame_lock:
        frame = raw_frame_bgr.copy()
    baseline = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (blur_k, blur_k), 0)
    baseline_time = time.monotonic()
    last_trigger = 0
    trigger_count = 0

    h, w = baseline.shape[:2]
    roi = det["roi"]
    ry = slice(int(roi[1] * h), int(roi[3] * h))
    rx = slice(int(roi[0] * w), int(roi[2] * w))

    print(f"Motion detection active — min_px={det['min_changed_pixels']}", flush=True)

    while True:
        time.sleep(0.1)
        with frame_lock:
            if raw_frame_bgr is None:
                continue
            frame = raw_frame_bgr.copy()

        now = time.monotonic()
        if now - baseline_time > det["baseline_refresh_seconds"]:
            baseline = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (blur_k, blur_k), 0)
            baseline_time = now
            continue

        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (blur_k, blur_k), 0)
        diff = cv2.absdiff(gray[ry, rx], baseline[ry, rx])
        _, thresh = cv2.threshold(diff, det["threshold"], 255, cv2.THRESH_BINARY)
        changed = cv2.countNonZero(thresh)

        if changed >= det["min_changed_pixels"] and (now - last_trigger) > det["cooldown_seconds"]:
            trigger_count += 1
            last_trigger = now
            ts = datetime.now(timezone.utc)
            date_str = ts.strftime("%Y-%m-%d")
            capture_id = str(uuid_mod.uuid4())

            out_dir = Path(cap_cfg["output_dir"]) / date_str
            out_dir.mkdir(parents=True, exist_ok=True)

            img_path = out_dir / f"{capture_id}.jpg"
            meta_path = out_dir / f"{capture_id}.json"
            cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, cap_cfg["jpeg_quality"]])
            meta = {"uuid": capture_id, "timestamp": ts.isoformat(), "date": date_str}
            meta_path.write_text(json.dumps(meta, indent=2))
            trigger_log.append(meta)
            if len(trigger_log) > 50:
                trigger_log.pop(0)
            print(f"[MOTION #{trigger_count}] {changed}px — {capture_id[:8]}…", flush=True)



@resilient_thread("bird_detect")
def bird_thread():
    global detect_jpeg

    cfg = load_config()
    cap_cfg = cfg["capture"]
    cooldown = cfg["detection"]["cooldown_seconds"]

    net = cv2.dnn.readNetFromCaffe(
        str(MODEL_DIR / "mobilenet_ssd.prototxt"),
        str(MODEL_DIR / "mobilenet_ssd.caffemodel"),
    )
    print("Bird detection model loaded (MobileNet-SSD VOC)", flush=True)

    while raw_frame_bgr is None:
        time.sleep(0.5)

    last_bird_trigger = 0
    bird_trigger_count = 0

    while True:
        time.sleep(0.15)  # ~6-7 fps
        with frame_lock:
            if raw_frame_bgr is None:
                continue
            frame = raw_frame_bgr.copy()

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        bird_found = False
        best_bird_conf = 0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])
            if confidence < 0.3:
                continue

            label = VOC_CLASSES[class_id] if class_id < len(VOC_CLASSES) else "?"
            x1 = max(0, int(detections[0, 0, i, 3] * w))
            y1 = max(0, int(detections[0, 0, i, 4] * h))
            x2 = min(w, int(detections[0, 0, i, 5] * w))
            y2 = min(h, int(detections[0, 0, i, 6] * h))

            if class_id == BIRD_CLASS_ID:
                bird_found = True
                best_bird_conf = max(best_bird_conf, confidence)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(frame, f"BIRD {confidence:.0%}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)
                cv2.putText(frame, f"{label} {confidence:.0%}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

        _, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with detect_cond:
            detect_jpeg = enc.tobytes()
            detect_cond.notify_all()

        # Bird detected — capture + upload to S3 for Bedrock analysis
        now = time.monotonic()
        if bird_found and best_bird_conf >= 0.4 and (now - last_bird_trigger) > cooldown:
            bird_trigger_count += 1
            last_bird_trigger = now

            ts = datetime.now(timezone.utc)
            date_str = ts.strftime("%Y-%m-%d")
            capture_id = str(uuid_mod.uuid4())

            out_dir = Path(cap_cfg["output_dir"]) / date_str
            out_dir.mkdir(parents=True, exist_ok=True)

            # Save the raw frame (not the overlay)
            with frame_lock:
                raw = raw_frame_bgr.copy() if raw_frame_bgr is not None else frame
            img_path = out_dir / f"{capture_id}.jpg"
            cv2.imwrite(str(img_path), raw, [cv2.IMWRITE_JPEG_QUALITY, cap_cfg["jpeg_quality"]])

            meta = {"uuid": capture_id, "timestamp": ts.isoformat(), "date": date_str,
                    "local_confidence": float(best_bird_conf)}
            (out_dir / f"{capture_id}.json").write_text(json.dumps(meta, indent=2))
            trigger_log.append(meta)
            if len(trigger_log) > 50:
                trigger_log.pop(0)

            upload_queue.put((str(img_path), date_str, capture_id))
            print(f"[BIRD #{bird_trigger_count}] conf={best_bird_conf:.0%} — {capture_id[:8]}… → S3", flush=True)


# S3 upload 

@resilient_thread("s3_upload")
def s3_upload_thread():
    """Uploads triggered captures to S3 to invoke Lambda → Bedrock."""
    cfg = load_config()
    s3_cfg = cfg["s3"]
    s3 = boto3.client("s3", region_name=s3_cfg["region"])
    bucket = s3_cfg["bucket"]
    prefix = s3_cfg["prefix_images"]

    print(f"S3 uploader ready — bucket={bucket}", flush=True)

    while True:
        img_path, date_str, capture_id = upload_queue.get()
        try:
            pipeline_status["state"] = "uploading"
            key = f"{prefix}/{date_str}/{capture_id}.jpg"
            s3.upload_file(img_path, bucket, key)
            pipeline_status["state"] = "processing"
            print(f"[S3] Uploaded {capture_id[:8]}… → s3://{bucket}/{key}", flush=True)
        except Exception as e:
            pipeline_status["state"] = "idle"
            print(f"[S3] Upload failed: {e}", flush=True)


#Result sync 

@resilient_thread("result_sync")
def result_sync_thread():
    """Polls S3 for new Bedrock results and caches them locally."""
    cfg = load_config()
    s3_cfg = cfg["s3"]
    s3 = boto3.client("s3", region_name=s3_cfg["region"])
    bucket = s3_cfg["bucket"]
    prefix = s3_cfg["prefix_results"]
    cap_dir = Path(cfg["capture"]["output_dir"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print("Result sync thread started", flush=True)

    while True:
        try:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if not key.endswith(".json"):
                        continue
                    # key = results/2026-04-09/uuid.json
                    parts = key.split("/")
                    if len(parts) < 3:
                        continue
                    date_str = parts[1]
                    fname = parts[2]
                    uuid = fname.replace(".json", "")

                    local_result = RESULTS_DIR / date_str / fname
                    if local_result.exists():
                        continue

                    local_result.parent.mkdir(parents=True, exist_ok=True)
                    s3.download_file(bucket, key, str(local_result))

                    try:
                        result_data = json.loads(local_result.read_text())
                        ai_debug_log.append(result_data)
                        if len(ai_debug_log) > 30:
                            ai_debug_log.pop(0)
                        if result_data.get("is_bird"):
                            pipeline_status["state"] = "detected"
                            pipeline_status["last_bird"] = result_data.get("species", {})
                            pipeline_status["last_ts"] = result_data.get("timestamp")
                        else:
                            pipeline_status["state"] = "idle"
                    except Exception:
                        pipeline_status["state"] = "idle"

                    src_img = cap_dir / date_str / f"{uuid}.jpg"
                    dst_img = IMAGES_DIR / date_str / f"{uuid}.jpg"
                    if src_img.exists() and not dst_img.exists():
                        dst_img.parent.mkdir(parents=True, exist_ok=True)
                        import shutil
                        shutil.copy2(str(src_img), str(dst_img))

                    print(f"[SYNC] New result: {date_str}/{uuid[:8]}…", flush=True)
        except Exception as e:
            print(f"[SYNC] Error: {e}", flush=True)

        if pipeline_status["state"] == "detected":
            pipeline_status["state"] = "idle"

        time.sleep(15)


# Stream generators

def gen_raw():
    while True:
        with raw_cond:
            raw_cond.wait(timeout=3)
            frame = raw_jpeg
        if frame is None:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
               + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")


def gen_detect():
    while True:
        with detect_cond:
            detect_cond.wait(timeout=3)
            frame = detect_jpeg
        if frame is None:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
               + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")


# Data helpers

def load_results(date_str):
    results_path = RESULTS_DIR / date_str
    if not results_path.exists():
        return []
    results = []
    for f in sorted(results_path.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            data["_uuid"] = f.stem
            data["_date"] = date_str
            results.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return results


def get_latest_bird():
    dates = sorted(
        [d.name for d in RESULTS_DIR.iterdir() if d.is_dir()], reverse=True
    ) if RESULTS_DIR.exists() else []
    newest = None
    for d in dates:
        results = load_results(d)
        birds = [r for r in results if r.get("is_bird")]
        for b in birds:
            ts = b.get("timestamp", "")
            if newest is None or ts > newest.get("timestamp", ""):
                newest = b
        if newest:
            return newest 
    return newest


def get_species_summary(date_str):
    results = load_results(date_str)
    birds = [r for r in results if r.get("is_bird")]
    species_map = {}
    for b in birds:
        name = (b.get("species", {}).get("common", "Unknown")
                if isinstance(b.get("species"), dict) else b.get("species", "Unknown"))
        if name not in species_map:
            species_map[name] = {
                "common": name,
                "scientific": (b.get("species", {}).get("scientific", "")
                               if isinstance(b.get("species"), dict) else ""),
                "count": 0,
                "fun_facts": b.get("fun_facts", []),
                "description": b.get("description", ""),
                "photo_uuid": b.get("_uuid"),
                "photo_date": b.get("_date"),
                "_latest_ts": b.get("timestamp", ""),
            }
        species_map[name]["count"] += 1
        ts = b.get("timestamp", "")
        if ts >= species_map[name]["_latest_ts"]:
            species_map[name]["_latest_ts"] = ts
            species_map[name]["photo_uuid"] = b.get("_uuid")
            species_map[name]["photo_date"] = b.get("_date")
            species_map[name]["fun_facts"] = b.get("fun_facts", []) or species_map[name]["fun_facts"]
            species_map[name]["description"] = b.get("description", "") or species_map[name]["description"]
    return sorted(species_map.values(), key=lambda s: s["count"], reverse=True)


def get_available_dates():
    if not RESULTS_DIR.exists():
        return []
    return sorted([d.name for d in RESULTS_DIR.iterdir() if d.is_dir()], reverse=True)



@app.route("/")
def index():
    return render_template("live.html")

@app.route("/live")
def live():
    return render_template("live.html")

@app.route("/daily")
def daily():
    dates = get_available_dates()
    selected = request.args.get("date", dates[0] if dates else date.today().isoformat())
    species = get_species_summary(selected)
    return render_template("daily.html", dates=dates, selected=selected, species=species)

@app.route("/stream")
def stream():
    return Response(gen_raw(), mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-cache"})

@app.route("/stream_detect")
def stream_detect():
    return Response(gen_detect(), mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-cache"})

@app.route("/api/latest")
def api_latest():
    bird = get_latest_bird()
    resp = jsonify(bird) if bird else jsonify(None)
    resp.headers["Cache-Control"] = "no-cache, no-store"
    return resp

@app.route("/api/triggers")
def api_triggers():
    resp = jsonify(trigger_log[-10:])
    resp.headers["Cache-Control"] = "no-cache, no-store"
    return resp

@app.route("/api/status")
def api_status():
    resp = jsonify(pipeline_status)
    resp.headers["Cache-Control"] = "no-cache, no-store"
    return resp

@app.route("/api/debug")
def api_debug():
    resp = jsonify(ai_debug_log[-15:])
    resp.headers["Cache-Control"] = "no-cache, no-store"
    return resp

@app.route("/api/health")
def api_health():
    now = time.time()
    threads = {}
    all_ok = True
    for name, info in thread_health.items():
        ok = info["alive"] and info["error"] is None
        threads[name] = {
            "alive": info["alive"],
            "restarts": info["restarts"],
            "error": info["error"],
        }
        if not ok:
            all_ok = False
    status_code = 200 if all_ok else 503
    resp = jsonify({"healthy": all_ok, "threads": threads})
    resp.status_code = status_code
    resp.headers["Cache-Control"] = "no-cache, no-store"
    return resp

@app.route("/cache/images/<date_str>/<filename>")
def serve_image(date_str, filename):
    img_path = IMAGES_DIR / date_str / filename
    if img_path.exists() and img_path.suffix in (".jpg", ".jpeg", ".png"):
        return Response(img_path.read_bytes(), mimetype="image/jpeg")
    return "", 404

@app.route("/captures/<date_str>/<filename>")
def serve_capture(date_str, filename):
    cfg = load_config()
    img_path = Path(cfg["capture"]["output_dir"]) / date_str / filename
    if img_path.exists() and img_path.suffix in (".jpg", ".jpeg", ".png"):
        return Response(img_path.read_bytes(), mimetype="image/jpeg")
    return "", 404


def _start_threads():
    """Start all background threads once."""
    Thread(target=camera_thread, daemon=True).start()
    Thread(target=motion_thread, daemon=True).start()
    Thread(target=bird_thread, daemon=True).start()
    Thread(target=s3_upload_thread, daemon=True).start()
    Thread(target=result_sync_thread, daemon=True).start()
    time.sleep(2)
    print("All threads started — http://0.0.0.0:5000/", flush=True)


_threads_started = False

def ensure_threads():
    global _threads_started
    if not _threads_started:
        _threads_started = True
        _start_threads()


if __name__ == "__main__":
    ensure_threads()
    app.run(host="0.0.0.0", port=5000, threaded=True)
else:
    ensure_threads()
