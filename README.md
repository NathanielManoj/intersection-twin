# Intersection Digital Twin

> A real-time cloud-based pipeline that uses multiple cameras and custom AI to create a live digital twin of an intersection — eliminating blind spots and improving safety.

**Capstone Group 19 — Ontario Tech University**

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Setup Guide](#setup-guide)
  - [1. Google Cloud Setup](#1-google-cloud-setup)
  - [2. Confluent Kafka Setup](#2-confluent-kafka-setup)
  - [3. Neo4j AuraDB Setup](#3-neo4j-auradb-setup)
  - [4. Deploy to Cloud Run](#4-deploy-to-cloud-run)
  - [5. Run the Pipeline](#5-run-the-pipeline)
- [Homography Calibration](#homography-calibration)
- [YOLO Model Training](#yolo-model-training)
- [Dashboard](#dashboard)
- [Neo4j Graph Structure](#neo4j-graph-structure)
- [Troubleshooting](#troubleshooting)
- [Cloud Resources](#cloud-resources)

---

## Overview

The Intersection Digital Twin system connects edge devices (Jetson Xavier) and cameras to a cloud pipeline that:

1. Receives camera frames from edge devices or video files
2. Runs YOLO object detection on each frame
3. Applies homography to create a birds eye view (BEV) of the intersection
4. Stitches both camera views into a unified digital twin
5. Streams metadata through Kafka into Neo4j graph database
6. Displays everything live on a web dashboard

The system runs two parallel pipelines:

**Telemetry Pipeline:**
```
Jetson Xavier (Kuksa + Eclipse Ditto) → Confluent Kafka → Neo4j AuraDB
```

**Image Stitching Pipeline:**
```
Camera / Video → Cloud Run (YOLO + BEV Stitching) → GCS → Kafka → Neo4j → Dashboard
```

---

## Architecture

```
Edge Devices                    Cloud
─────────────────               ──────────────────────────────────────────────
Jetson Xavier                   Confluent Kafka
  ├── Kuksa Databroker    ────►    vehicle.digitaltwin topic
  └── Eclipse Ditto               camera.stitched.output topic
                                        │
Laptop / Jetson                         ▼
  └── send_frames.py     ────►  Google Cloud Storage
        │                         images/ (input frames)
        │                         outputs/ (stitched results)
        │                               │
        │                               ▼
        └──────────────────────► Cloud Run (intersection-twin)
                                   ├── app.py (Flask server)
                                   ├── main.py (YOLO + stitching)
                                   ├── detect.py (YOLO detection)
                                   └── simulate.py (digital twin grid)
                                               │
                                               ▼
                                        Neo4j AuraDB
                                   (Intersection → StitchedFrame)
                                               │
                                               ▼
                                          Dashboard
                          /dashboard — live annotated camera views
```

---

## Repository Structure

```
intersection-twin/
├── app.py                  # Flask web server — /dashboard, /stitch, /upload endpoints
├── main.py                 # Core pipeline — downloads from GCS, runs YOLO, uploads outputs
├── detect.py               # YOLO detection logic
├── simulate.py             # Renders birds eye digital twin grid
├── pick_corners.py         # Homography calibration tool
├── send_frames.py          # Sends video frames to Cloud Run /upload endpoint
├── Dockerfile              # Container definition for Cloud Run
├── requirements.txt        # Python dependencies
├── stitch.sh               # Shell script to trigger a stitch manually
├── final.pt                # Fine-tuned YOLO model (trained on ROSMASTERs)
├── config/
│   ├── H_cam1.npy          # Homography matrix for camera 1
│   ├── H_cam2.npy          # Homography matrix for camera 2
│   ├── cam1_scene.jpg      # Reference scene image for camera 1
│   └── cam2_scene.jpg      # Reference scene image for camera 2
└── videos/                 # Local test videos (not committed to git)
    ├── video1.avi
    └── video2.avi
```

---

## Prerequisites

- Python 3.11+
- Docker
- Google Cloud SDK (`gcloud`)
- ffmpeg (`brew install ffmpeg` on Mac, `sudo apt install ffmpeg` on Jetson/Linux)
- A GCP project with billing enabled
- A Confluent Cloud account
- A Neo4j AuraDB account

---

## Setup Guide

### 1. Google Cloud Setup

**Enable required APIs:**
```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

**Create GCS bucket:**
```bash
gcloud storage buckets create gs://vehicle-camera-frames --location=us-east1

# Disable public access prevention and make bucket public
gcloud storage buckets update gs://vehicle-camera-frames --no-public-access-prevention
gsutil iam ch allUsers:objectViewer gs://vehicle-camera-frames
```

**Upload test images:**
```bash
gcloud storage cp config/cam1_scene.jpg gs://vehicle-camera-frames/images/cam1_scene.jpg
gcloud storage cp config/cam2_scene.jpg gs://vehicle-camera-frames/images/cam2_scene.jpg
```

**Upload YOLO model via GCS (never use git for .pt files — they corrupt):**
```bash
# From your local machine
gcloud storage cp ~/final.pt gs://vehicle-camera-frames/models/final.pt

# In Cloud Shell
gcloud storage cp gs://vehicle-camera-frames/models/final.pt ~/intersection-twin/final.pt
```

---

### 2. Confluent Kafka Setup

1. Create a free cluster at [confluent.cloud](https://confluent.cloud)
2. Create two topics:
   - `vehicle.digitaltwin` — telemetry from Ditto
   - `camera.stitched.output` — stitching metadata
3. Generate an API key for Cloud Run (save the key and secret)
4. Create two Neo4j Sink Connectors (see [Neo4j Graph Structure](#neo4j-graph-structure) for Cypher queries)

**Bootstrap server:** `pkc-619z3.us-east1.gcp.confluent.cloud:9092`

> **Note:** Confluent connectors bill continuously even when paused. Delete them when not in use and recreate before demos using the Cypher queries below.

---

### 3. Neo4j AuraDB Setup

1. Create a free instance at [console.neo4j.io](https://console.neo4j.io)
2. Save your password immediately — it is only shown once
3. Connection details:
   - **URI:** `neo4j+s://YOUR_INSTANCE_ID.databases.neo4j.io`
   - **Username:** your instance ID
   - **Encrypted:** `true` (required for AuraDB)

> **Note:** AuraDB Free pauses after 3 days of inactivity. Resume from the console before running the pipeline.

**Neo4j Sink Connector — Telemetry topic:**
```json
{"vehicle.digitaltwin": "MERGE (d:Device {id: __value.path}) MERGE (r:LidarReading {revision: toInteger(__value.revision)}) SET r.timestamp = __value.timestamp, r.centroidX = toFloat(__value.value.Centroid.x), r.centroidY = toFloat(__value.value.Centroid.y), r.pointCount = toInteger(__value.value.PointCount) MERGE (d)-[:HAS_READING]->(r)"}
```

**Neo4j Sink Connector — Stitching topic:**
```json
{"camera.stitched.output": "MERGE (i:Intersection {id: 'intersection_1'}) MERGE (f:StitchedFrame {timestamp: __value.timestamp}) SET f.gcs_url = __value.gcs_url, f.cam1_objects = toInteger(__value.cam1_objects), f.cam2_objects = toInteger(__value.cam2_objects), f.total_unique_objects = toInteger(__value.total_unique_objects) MERGE (i)-[:HAS_FRAME]->(f)"}
```

---

### 4. Deploy to Cloud Run

**Clone the repo in Google Cloud Shell:**
```bash
git clone https://github.com/NathanielManoj/intersection-twin.git
cd intersection-twin

# Download the YOLO model from GCS
gcloud storage cp gs://vehicle-camera-frames/models/final.pt ~/intersection-twin/final.pt
```

**Build the Docker image:**
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/intersection-twin
```

> This takes ~8-12 minutes on first build.

**Deploy to Cloud Run:**
```bash
gcloud run deploy intersection-twin \
  --image gcr.io/YOUR_PROJECT_ID/intersection-twin \
  --platform managed \
  --region us-east1 \
  --set-env-vars GCS_BUCKET=vehicle-camera-frames,\
KAFKA_BOOTSTRAP=YOUR_CONFLUENT_BOOTSTRAP_SERVER,\
KAFKA_API_KEY=YOUR_CONFLUENT_API_KEY,\
KAFKA_API_SECRET=YOUR_CONFLUENT_API_SECRET,\
NEO4J_URI=neo4j+s://YOUR_INSTANCE_ID.databases.neo4j.io,\
NEO4J_USER=YOUR_NEO4J_USERNAME,\
NEO4J_PASSWORD=YOUR_NEO4J_PASSWORD \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --allow-unauthenticated
```

---

### 5. Run the Pipeline

**Install dependencies on your laptop or Jetson:**
```bash
pip3 install requests
```

**Send video frames continuously:**
```bash
python3 send_frames.py --video1 videos/video1.avi --video2 videos/video2.avi --interval 1
```

**Send a single frame pair for testing:**
```bash
python3 send_frames.py --video1 videos/video1.avi --video2 videos/video2.avi --once
```

**Trigger a stitch manually using static GCS images:**
```bash
./stitch.sh
```

Or via curl:
```bash
curl -X POST https://YOUR_CLOUD_RUN_URL/stitch \
  -H "Content-Type: application/json" \
  -d '{"cam1_gcs": "images/cam1_scene.jpg", "cam2_gcs": "images/cam2_scene.jpg"}'
```

**Upload a live frame directly:**
```bash
curl -X POST https://YOUR_CLOUD_RUN_URL/upload \
  -F "camera_id=cam1" \
  -F "session_id=test123" \
  -F "frame=@/path/to/frame.jpg"
```

---

## Homography Calibration

The homography matrices define how each camera's perspective is warped into a birds eye view. They must be recalibrated whenever cameras are moved.

**Step 1 — Extract a reference frame from each video:**
```bash
ffmpeg -ss 2 -i videos/video1.avi -vframes 1 -update 1 config/cam1_scene.jpg -y
ffmpeg -ss 2 -i videos/video2.avi -vframes 1 -update 1 config/cam2_scene.jpg -y
```

**Step 2 — Run calibration for each camera:**
```bash
python3 pick_corners.py --image config/cam1_scene.jpg --cam cam1
python3 pick_corners.py --image config/cam2_scene.jpg --cam cam2
```

**Instructions:**
- **Stage 1:** Click a wide rectangle around the entire intersection area including all 4 reference markers (white papers). Order: Top-Left → Top-Right → Bottom-Left → Bottom-Right. Press Enter.
- **Stage 2:** Click the 4 reference markers precisely in the same order. Press Enter.
- Click the same physical corners in the same order for both cameras.

New `H_cam1.npy` and `H_cam2.npy` will be saved to `config/`.

---

## YOLO Model Training

The model (`final.pt`) was fine-tuned on ROSMASTER robot images using Google Colab and Roboflow.

**To retrain:**

1. Extract training frames from your videos:
```bash
mkdir -p training_images
ffmpeg -i videos/video1.avi -vf fps=2 training_images/cam1_frame_%03d.jpg
ffmpeg -i videos/video2.avi -vf fps=2 training_images/cam2_frame_%03d.jpg
```

2. Upload and label at [roboflow.com](https://roboflow.com) — export as YOLOv8 format

3. Train in Google Colab:
```python
from ultralytics import YOLO
model = YOLO("yolo11m.pt")
model.train(data="dataset/data.yaml", epochs=50, imgsz=640, batch=8)
```

4. Download `best.pt`, verify it, then upload to GCS:
```bash
# Verify before anything else
python3 -c "import torch; torch.load('best.pt', map_location='cpu', weights_only=False); print('Valid!')"

# Upload to GCS (never commit .pt files to git — they corrupt)
gcloud storage cp best.pt gs://vehicle-camera-frames/models/final.pt
```

> **Important:** Always store `.pt` files in GCS, not in git or OneDrive. Large binary files corrupt during git push/pull and OneDrive sync.

---

## Dashboard

The live dashboard is served at `/dashboard` on your Cloud Run URL.

**It shows:**
- Camera 1 annotated view (real camera frame with YOLO bounding boxes)
- Camera 2 annotated view
- Birds eye digital twin grid (objects plotted by position)
- Last stitched timestamp
- Object detection counts per camera
- Total frames processed

**Output files per stitch (stored in GCS `outputs/TIMESTAMP/`):**

| File | Description |
|------|-------------|
| `sim_global.jpg` | Digital twin birds eye grid |
| `annotated_orig_cam1.jpg` | Camera 1 with YOLO boxes |
| `annotated_orig_cam2.jpg` | Camera 2 with YOLO boxes |
| `bev_cam1.jpg` | Camera 1 birds eye view |
| `bev_cam2.jpg` | Camera 2 birds eye view |
| `bev_annotated_cam1.jpg` | Camera 1 BEV with detections |
| `bev_annotated_cam2.jpg` | Camera 2 BEV with detections |
| `sim_cam1.jpg` | Camera 1 simulation view |
| `sim_cam2.jpg` | Camera 2 simulation view |

---

## Neo4j Graph Structure

```
(Intersection {id: 'intersection_1'})
        │
        └──[:HAS_FRAME]──► (StitchedFrame {
                                timestamp,
                                gcs_url,
                                cam1_objects,
                                cam2_objects,
                                total_unique_objects
                            })

(Device {id: path})
        │
        └──[:HAS_READING]──► (LidarReading {
                                  revision,
                                  timestamp,
                                  centroidX,
                                  centroidY,
                                  pointCount
                              })
```

**Useful queries:**
```cypher
// See all nodes
MATCH (n) RETURN n LIMIT 25

// Latest stitched frames
MATCH (i:Intersection)-[:HAS_FRAME]->(f:StitchedFrame)
RETURN f.timestamp, f.total_unique_objects
ORDER BY f.timestamp DESC LIMIT 10

// Total frames processed
MATCH (f:StitchedFrame) RETURN count(f) AS total

// LiDAR readings per device
MATCH (d:Device)-[:HAS_READING]->(r:LidarReading)
RETURN d.id, count(r) AS readings ORDER BY readings DESC
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Dashboard shows no image | GCS public access prevention enabled | Run `gcloud storage buckets update gs://vehicle-camera-frames --no-public-access-prevention` then `gsutil iam ch allUsers:objectViewer gs://vehicle-camera-frames` |
| 500 error — corrupted .pt file | Model file corrupted via git or OneDrive | Re-download from Colab, verify with `torch.load()`, upload via GCS |
| Neo4j shows no new data | Connector paused or AuraDB paused | Resume AuraDB at console.neo4j.io, check connector status in Confluent |
| BEV image is mostly black | Homography matrices calibrated for wrong camera position | Recalibrate with `pick_corners.py` using current camera position |
| Ditto services restarting on Jetson | Kernel blocking JVM thread creation | Add `privileged: true` to policies, things, things-search, connectivity, gateway in docker-compose.yml |
| 404 when running forward script | Ditto Thing or LiDAR feature doesn't exist | Create via curl: `curl -u ditto:ditto -X PUT http://localhost:8080/api/2/things/org.vehicle:lidar-2d/features/LiDAR -H "Content-Type: application/json" -d '{"properties": {"Centroid": {"x": 0.0, "y": 0.0, "z": "0.0"}, "Intensity": {"Avg": "0.0"}, "PointCount": 0}}'` |
| Kafka connection URI invalid | Special characters in secret not URL-encoded | Run `encodeURIComponent("your_secret")` in browser console and use encoded value |
| Neo4j connector not writing data | Wrong Cypher format | Delete connector and recreate — edits to running connectors often don't take effect |
| Cloud Shell files missing after session | Cloud Shell home directory reset | All files are in GitHub — just clone again and download model from GCS |

---

## Cloud Resources

| Resource | Value |
|----------|-------|
| GCP Project | `atomic-horizon-488020-d5` |
| GCS Bucket | `vehicle-camera-frames` |
| Cloud Run Service | `intersection-twin` (us-east1) |
| Dashboard URL | `https://intersection-twin-809600582102.us-east1.run.app/dashboard` |
| Kafka Bootstrap | `pkc-619z3.us-east1.gcp.confluent.cloud:9092` |
| Kafka Telemetry Topic | `vehicle.digitaltwin` |
| Kafka Stitching Topic | `camera.stitched.output` |
| Neo4j Console | `console.neo4j.io` |
| Neo4j URI | `neo4j+s://0565a42a.databases.neo4j.io` |

---

## Contact

Capstone Group 19 — Ontario Tech University
