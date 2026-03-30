import os
import subprocess
import json
import datetime
from flask import Flask, request, jsonify, render_template_string
from confluent_kafka import Producer
from google.cloud import storage
from neo4j import GraphDatabase

app = Flask(__name__)

KAFKA_BOOTSTRAP  = os.environ.get("KAFKA_BOOTSTRAP", "")
KAFKA_API_KEY    = os.environ.get("KAFKA_API_KEY", "")
KAFKA_API_SECRET = os.environ.get("KAFKA_API_SECRET", "")
KAFKA_TOPIC      = "camera.stitched.output"
GCS_BUCKET       = os.environ.get("GCS_BUCKET", "vehicle-camera-frames")
NEO4J_URI        = os.environ.get("NEO4J_URI", "")
NEO4J_USER       = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD   = os.environ.get("NEO4J_PASSWORD", "")

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Intersection Digital Twin</title>
    <meta http-equiv="refresh" content="30">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #0f1117; color: #ffffff; font-family: Arial, sans-serif; padding: 30px; }
        h1 { color: #4A90D9; font-size: 28px; margin-bottom: 6px; }
        .subtitle { color: #888; font-size: 14px; margin-bottom: 30px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 30px; }
        .card { background: #1a1d2e; border-radius: 12px; padding: 20px; border: 1px solid #2a2d3e; }
        .card-title { color: #888; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
        .card-value { font-size: 36px; font-weight: bold; color: #4A90D9; }
        .card-sub { font-size: 12px; color: #666; margin-top: 4px; }
        .image-section { background: #1a1d2e; border-radius: 12px; padding: 20px; border: 1px solid #2a2d3e; }
        .image-section h2 { color: #4A90D9; font-size: 18px; margin-bottom: 6px; }
        .image-section p { color: #888; font-size: 12px; margin-bottom: 16px; }
        .image-section img { width: 100%; border-radius: 8px; border: 1px solid #2a2d3e; }
        .no-image { color: #666; font-size: 14px; padding: 40px; text-align: center; border: 1px dashed #2a2d3e; border-radius: 8px; }
        .refresh { color: #666; font-size: 12px; margin-top: 20px; text-align: right; }
        .status { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #2ecc71; margin-right: 6px; }
    </style>
</head>
<body>
    <h1><span class="status"></span>Intersection Digital Twin Dashboard</h1>
    <p class="subtitle">GCP Project: atomic-horizon-488020-d5 &nbsp;|&nbsp; Real-time intersection monitoring</p>
    <div class="grid">
        <div class="card">
            <div class="card-title">Last Stitched</div>
            <div class="card-value" style="font-size: 20px;">{{ timestamp }}</div>
            <div class="card-sub">Most recent frame processed</div>
        </div>
        <div class="card">
            <div class="card-title">Total Objects Detected</div>
            <div class="card-value">{{ total_objects }}</div>
            <div class="card-sub">Cam1: {{ cam1_objects }} &nbsp;|&nbsp; Cam2: {{ cam2_objects }}</div>
        </div>
        <div class="card">
            <div class="card-title">Total Frames Processed</div>
            <div class="card-value">{{ total_frames }}</div>
            <div class="card-sub">Stored in Neo4j</div>
        </div>
    </div>
    <div class="image-section">
        <h2>Latest Intersection View — YOLO Detection</h2>
        <p>Real camera frames with object detection overlaid</p>
        {% if image_url %}
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px;">
            <div>
                <p style="color: #4A90D9; font-size: 13px; margin-bottom: 8px;">Camera 1</p>
                <img src="{{ cam1_url }}" alt="Camera 1 annotated" style="width:100%; border-radius:8px;">
            </div>
            <div>
                <p style="color: #4A90D9; font-size: 13px; margin-bottom: 8px;">Camera 2</p>
                <img src="{{ cam2_url }}" alt="Camera 2 annotated" style="width:100%; border-radius:8px;">
            </div>
        </div>
        <p style="color: #888; font-size: 12px; margin-bottom: 8px;">Birds Eye View (Digital Twin)</p>
        <img src="{{ image_url }}" alt="Digital twin view" style="width:100%; border-radius:8px;">
        {% else %}
            <div class="no-image">No stitched frames available yet. Trigger a stitch to see the output here.</div>
        {% endif %}
    </div>
    <p class="refresh">Auto-refreshes every 30 seconds &nbsp;|&nbsp; Last loaded: {{ now }}</p>
</body>
</html>
"""

def get_producer():
    return Producer({
        'bootstrap.servers': KAFKA_BOOTSTRAP,
        'security.protocol': 'SASL_SSL',
        'sasl.mechanism': 'PLAIN',
        'sasl.username': KAFKA_API_KEY,
        'sasl.password': KAFKA_API_SECRET,
    })

def get_latest_frame_from_neo4j():
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run("""
                MATCH (i:Intersection)-[:HAS_FRAME]->(f:StitchedFrame)
                RETURN f.timestamp AS timestamp, f.gcs_url AS gcs_url,
                       f.cam1_objects AS cam1_objects, f.cam2_objects AS cam2_objects,
                       f.total_unique_objects AS total_objects
                ORDER BY f.timestamp DESC LIMIT 1
            """)
            record = result.single()
            total = session.run("MATCH (f:StitchedFrame) RETURN count(f) AS total").single()
            driver.close()
            if record:
                return {
                    "timestamp": record["timestamp"],
                    "gcs_url": record["gcs_url"],
                    "cam1_objects": record["cam1_objects"] or 0,
                    "cam2_objects": record["cam2_objects"] or 0,
                    "total_objects": record["total_objects"] or 0,
                    "total_frames": total["total"] or 0
                }
    except Exception as e:
        print(f"Neo4j error: {e}")
    return None

def get_image_url(gcs_path):
    try:
        blob_path = gcs_path.replace(f"gs://{GCS_BUCKET}/", "").lstrip("/")
        return f"https://storage.googleapis.com/{GCS_BUCKET}/{blob_path}"
    except Exception as e:
        print(f"GCS error: {e}")
    return None

def run_stitch(cam1_gcs, cam2_gcs, timestamp):
    result = subprocess.run([
        "python3", "main.py",
        "--cam1_gcs", cam1_gcs,
        "--cam2_gcs", cam2_gcs,
        "--timestamp", timestamp
    ], capture_output=True, text=True)
    return result

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/dashboard", methods=["GET"])
def dashboard():
    frame = get_latest_frame_from_neo4j()
    image_url = None
    if frame and frame.get("gcs_url"):
        image_url = get_image_url(frame["gcs_url"])
    cam1_url = None
    cam2_url = None
    if frame and frame.get("timestamp"):
        ts = frame["timestamp"]
        cam1_url = f"https://storage.googleapis.com/{GCS_BUCKET}/outputs/{ts}/annotated_orig_cam1.jpg"
        cam2_url = f"https://storage.googleapis.com/{GCS_BUCKET}/outputs/{ts}/annotated_orig_cam2.jpg"

    return render_template_string(DASHBOARD_HTML,
        timestamp=frame["timestamp"] if frame else "No data yet",
        total_objects=frame["total_objects"] if frame else 0,
        cam1_objects=frame["cam1_objects"] if frame else 0,
        cam2_objects=frame["cam2_objects"] if frame else 0,
        total_frames=frame["total_frames"] if frame else 0,
        image_url=image_url,
        cam1_url=cam1_url,
        cam2_url=cam2_url,
        now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

@app.route("/upload", methods=["POST"])
def upload():
    """
    Accept a raw JPEG frame from a Jetson or laptop.
    Once both cam1 and cam2 are uploaded with the same session_id,
    automatically trigger the stitch.
    
    Usage:
      curl -X POST .../upload -F "camera_id=cam1" -F "session_id=abc123" -F "frame=@frame.jpg"
      curl -X POST .../upload -F "camera_id=cam2" -F "session_id=abc123" -F "frame=@frame.jpg"
    """
    camera_id  = request.form.get("camera_id")
    session_id = request.form.get("session_id", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    frame_file = request.files.get("frame")

    if not camera_id or camera_id not in ["cam1", "cam2"]:
        return jsonify({"error": "camera_id must be cam1 or cam2"}), 400
    if not frame_file:
        return jsonify({"error": "frame file is required"}), 400

    # Upload frame to GCS
    gcs_path = f"images/live/{session_id}/{camera_id}.jpg"
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(gcs_path)
        blob.upload_from_file(frame_file, content_type="image/jpeg")
        print(f"Uploaded {camera_id} frame to gs://{GCS_BUCKET}/{gcs_path}")
    except Exception as e:
        return jsonify({"error": f"GCS upload failed: {e}"}), 500

    # Check if both frames exist for this session
    cam1_path = f"images/live/{session_id}/cam1.jpg"
    cam2_path = f"images/live/{session_id}/cam2.jpg"

    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        cam1_exists = bucket.blob(cam1_path).exists()
        cam2_exists = bucket.blob(cam2_path).exists()
    except Exception as e:
        return jsonify({"error": f"GCS check failed: {e}"}), 500

    # If both frames are ready, trigger stitch
    if cam1_exists and cam2_exists:
        print(f"Both frames ready for session {session_id} — triggering stitch!")
        result = run_stitch(cam1_path, cam2_path, session_id)

        if result.returncode != 0:
            return jsonify({"error": result.stderr}), 500

        output = result.stdout
        cam1_objects = 0
        cam2_objects = 0
        total_objects = 0

        for line in output.split('\n'):
            if line.startswith("cam1:") and "objects detected" in line:
                cam1_objects = int(line.split()[1])
            if line.startswith("cam2:") and "objects detected" in line:
                cam2_objects = int(line.split()[1])
            if line.startswith("Global:"):
                total_objects = int(line.split()[1])

        metadata = {
            "timestamp": session_id,
            "gcs_url": f"gs://{GCS_BUCKET}/outputs/{session_id}/sim_global.jpg",
            "cam1_gcs": cam1_path,
            "cam2_gcs": cam2_path,
            "cam1_objects": cam1_objects,
            "cam2_objects": cam2_objects,
            "total_unique_objects": total_objects
        }

        try:
            producer = get_producer()
            producer.produce(KAFKA_TOPIC, json.dumps(metadata).encode('utf-8'))
            producer.flush()
        except Exception as e:
            print(f"Kafka publish failed: {e}")

        return jsonify({"status": "stitched", "metadata": metadata}), 200

    return jsonify({
        "status": "uploaded",
        "camera_id": camera_id,
        "session_id": session_id,
        "gcs_path": gcs_path,
        "waiting_for": "cam2" if camera_id == "cam1" else "cam1"
    }), 200

@app.route("/stitch", methods=["POST"])
def stitch():
    data = request.get_json()
    cam1_gcs = data.get("cam1_gcs")
    cam2_gcs = data.get("cam2_gcs")

    if not cam1_gcs or not cam2_gcs:
        return jsonify({"error": "cam1_gcs and cam2_gcs are required"}), 400

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result = run_stitch(cam1_gcs, cam2_gcs, timestamp)

    if result.returncode != 0:
        return jsonify({"error": result.stderr}), 500

    output = result.stdout
    cam1_objects = 0
    cam2_objects = 0
    total_objects = 0

    for line in output.split('\n'):
        if line.startswith("cam1:") and "objects detected" in line:
            cam1_objects = int(line.split()[1])
        if line.startswith("cam2:") and "objects detected" in line:
            cam2_objects = int(line.split()[1])
        if line.startswith("Global:"):
            total_objects = int(line.split()[1])

    metadata = {
        "timestamp": timestamp,
        "gcs_url": f"gs://{GCS_BUCKET}/outputs/{timestamp}/sim_global.jpg",
        "cam1_gcs": cam1_gcs,
        "cam2_gcs": cam2_gcs,
        "cam1_objects": cam1_objects,
        "cam2_objects": cam2_objects,
        "total_unique_objects": total_objects
    }

    try:
        producer = get_producer()
        producer.produce(KAFKA_TOPIC, json.dumps(metadata).encode('utf-8'))
        producer.flush()
    except Exception as e:
        print(f"Kafka publish failed: {e}")

    return jsonify({"status": "success", "output": output, "metadata": metadata}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
