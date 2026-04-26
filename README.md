# Smart Birdhouse

A Raspberry Pi–based birdhouse camera that uses motion detection and MobileNet-SSD to spot birds, uploads captures to S3, and runs them through AWS Bedrock (Claude) for species identification. Results are displayed in a live web UI.

## Architecture

```
USB Camera
    │
    ▼
camera_thread (ffmpeg MJPEG)
    │
    ├──► raw JPEG stream  ──► /stream         (Flask)
    │
    ├──► motion_thread    ──► saves JPEG + JSON sidecar to captures/
    │
    └──► bird_thread (MobileNet-SSD)
              │
              ├──► /stream_detect             (overlay JPEG stream)
              │
              └──► s3_upload_thread  ──► S3 bucket
                                              │
                                              ▼
                                        Lambda → Bedrock (Claude)
                                              │
                                              ▼
                                        result_sync_thread ──► web UI
```

## Requirements

- Python 3.10+
- USB webcam (V4L2, MJPEG capable)
- `ffmpeg` installed system-wide
- AWS account with S3 bucket, Lambda function, and Bedrock access
- AWS credentials configured (see below)

### Python packages

```bash
pip install flask gunicorn opencv-python-headless boto3 numpy ultralytics
```

## Setup

### 1. Clone and configure

```bash
git clone https://github.com/your-username/birdhouse.git
cd birdhouse
cp config.json.example config.json
```

Edit `config.json`:

```json
{
    "camera_index": 0,
    "resolution": [1920, 1080],
    "detection": {
        "roi": [0.0, 0.0, 1.0, 1.0],
        "threshold": 30,
        "blur_kernel": 21,
        "min_changed_pixels": 5000,
        "baseline_refresh_seconds": 300,
        "cooldown_seconds": 12
    },
    "capture": {
        "output_dir": "/path/to/birdhouse/captures",
        "jpeg_quality": 95
    },
    "s3": {
        "bucket": "your-s3-bucket-name",
        "prefix_images": "images",
        "prefix_results": "results",
        "region": "us-east-1"
    }
}
```

- `roi` — fractional `[x1, y1, x2, y2]` region of interest for motion detection. `[0,0,1,1]` means the full frame.
- `cooldown_seconds` — minimum gap between motion triggers.
- `baseline_refresh_seconds` — how often to refresh the background reference frame.

### 2. Download model files

The model files are not included in the repo due to size. Place them in the project root:

| File | Source |
|---|---|
| `mobilenet_ssd.caffemodel` | [chuanqi305/MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD) |
| `mobilenet_ssd.prototxt` | included in repo |
| `yolov8n.pt` | `pip install ultralytics && yolo export model=yolov8n.pt` or download from [Ultralytics](https://github.com/ultralytics/ultralytics) |

### 3. Configure AWS credentials

The app uses boto3's standard credential chain — no keys are stored in code or config. Use one of:

**Option A — AWS CLI (recommended for local dev):**
```bash
aws configure
```

**Option B — Environment variables:**
```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1
```

**Option C — IAM role** (recommended for EC2/Pi in production). Attach a role with `s3:PutObject`, `s3:GetObject`, and `s3:ListBucket` permissions to your instance.

### 4. AWS infrastructure

You'll need:

- An **S3 bucket** (set in `config.json`). Enable event notifications to trigger a Lambda function on `s3:ObjectCreated:*` under the `images/` prefix.
- A **Lambda function** that reads the uploaded JPEG, calls **Amazon Bedrock** (Claude), and writes a result JSON to `results/<date>/<uuid>.json` in the same bucket.
- The result JSON is expected to have at minimum:
  ```json
  {
    "is_bird": true,
    "species": {"common_name": "Black-capped Chickadee"},
    "timestamp": "2026-04-26T12:00:00+00:00"
  }
  ```

### 5. Run

**Development:**
```bash
python web.py
```

**Production (systemd):**
```bash
# Edit birdhouse.service — update User and WorkingDirectory to match your setup
sudo cp birdhouse.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable birdhouse
sudo systemctl start birdhouse
```

The web UI will be available at `http://<your-pi-ip>:5000`.

## Web UI

| Route | Description |
|---|---|
| `/` | Daily gallery of confirmed bird sightings |
| `/live` | Live camera feed with detection overlay |
| `/stream` | Raw MJPEG stream |
| `/stream_detect` | MJPEG stream with MobileNet-SSD bounding boxes |
| `/api/triggers` | JSON — recent motion triggers |
| `/api/status` | JSON — pipeline state (idle / uploading / processing / detected) |

## Project structure

```
web.py                  # Main app: camera, motion, detection, Flask server
detector.py             # Standalone motion-only detector (no web server)
config.json.example     # Config template — copy to config.json and edit
mobilenet_ssd.prototxt  # MobileNet-SSD network definition
birdhouse.service       # systemd unit file
templates/              # Jinja2 HTML templates
captures/               # Local capture storage (gitignored)
cache/                  # Cached S3 results and images (gitignored)
```
