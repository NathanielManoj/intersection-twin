#!/usr/bin/env python3
"""
send_frames.py — Video Frame Sender
-------------------------------------
Extracts one frame every INTERVAL seconds from two video files
and sends each frame to the Cloud Run /upload endpoint.
Once both cam1 and cam2 frames are received by the server,
it automatically triggers the stitching pipeline.

Works on MacBook and Jetson Xavier.
Supports .avi, .mp4, .mov and any other ffmpeg-supported format.

Folder structure:
  root/
  ├── videos/
  │   ├── video1.avi   (or .mp4, .mov)
  │   └── video2.avi
  └── send_frames.py

Install dependencies:
  pip install requests

Also requires ffmpeg:
  macOS : brew install ffmpeg
  Jetson: sudo apt install ffmpeg

Run:
  python3 send_frames.py
  python3 send_frames.py --video1 myvideo1.mp4 --video2 myvideo2.mp4
  python3 send_frames.py --interval 5 --once   (send just one frame pair and exit)
"""

import os
import sys
import subprocess
import time
import argparse
import tempfile
from datetime import datetime, timezone
import requests

# ──────────────────────────────────────────────────────
# CONFIG — edit these values for your local test environment.
# UPLOAD_URL should point to the service /upload endpoint.
# VIDEO_1 and VIDEO_2 should be the two source camera files.
# INTERVAL controls how often a fresh frame pair is sent.
# JPEG_Q controls ffmpeg JPEG compression quality.
# ──────────────────────────────────────────────────────
UPLOAD_URL = "https://intersection-twin-809600582102.us-east1.run.app/upload"
VIDEO_1    = "video1.avi"
VIDEO_2    = "video2.avi"
INTERVAL   = 3       # seconds between each frame pair
JPEG_Q     = "2"     # JPEG quality: 1 (best) to 31 (worst)
# ──────────────────────────────────────────────────────


def check_ffmpeg():
    """Ensure ffmpeg is installed before trying to extract video frames."""
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True)
    if result.returncode != 0:
        print("ERROR: ffmpeg not found.")
        print("  macOS : brew install ffmpeg")
        print("  Jetson: sudo apt install ffmpeg")
        sys.exit(1)


def get_video_duration(video_path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def extract_frame(video_path: str, timestamp: float) -> str:
    """
    Extract one JPEG frame at `timestamp` seconds from the video.
    Saves to a temp file and returns the path.
    Returns None on failure.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp_path = tmp.name
    tmp.close()

    result = subprocess.run(
        [
            "ffmpeg",
            "-ss", f"{timestamp:.3f}",
            "-i", video_path,
            "-vframes", "1",
            "-q:v", JPEG_Q,
            "-y",
            tmp_path,
        ],
        capture_output=True,
    )

    if result.returncode != 0 or not os.path.exists(tmp_path):
        return None

    # Verify file has content
    if os.path.getsize(tmp_path) < 100:
        os.unlink(tmp_path)
        return None

    return tmp_path


def send_frame(frame_path: str, camera_id: str, session_id: str) -> dict:
    """
    Send a JPEG frame to the Cloud Run /upload endpoint.
    Returns the response JSON on success, None on failure.
    """
    try:
        with open(frame_path, "rb") as f:
            response = requests.post(
                UPLOAD_URL,
                data={
                    "camera_id": camera_id,
                    "session_id": session_id,
                },
                files={
                    "frame": (f"{camera_id}.jpg", f, "image/jpeg"),
                },
                timeout=300
            )

        if response.status_code == 200:
            return response.json()
        else:
            print(f"  [FAIL] Server returned {response.status_code}: {response.text}")
            return None

    except requests.exceptions.ConnectionError:
        print(f"  [FAIL] Could not connect to {UPLOAD_URL}")
        print("  Make sure you have internet access and the service is running.")
        return None
    except requests.exceptions.Timeout:
        print(f"  [FAIL] Request timed out — stitching may still be running on server")
        return None
    except Exception as e:
        print(f"  [FAIL] Unexpected error: {e}")
        return None


def process_result(result: dict, camera_id: str, session_id: str):
    """Print the result of an upload or stitch."""
    if result is None:
        return

    status = result.get("status")

    if status == "uploaded":
        waiting_for = result.get("waiting_for", "other camera")
        print(f"  Uploaded {camera_id} — waiting for {waiting_for}...")

    elif status == "stitched":
        meta = result.get("metadata", {})
        print(f"  Stitch complete for session {session_id}!")
        print(f"  cam1 objects : {meta.get('cam1_objects', '?')}")
        print(f"  cam2 objects : {meta.get('cam2_objects', '?')}")
        print(f"  total unique : {meta.get('total_unique_objects', '?')}")
        print(f"  Output URL   : https://storage.googleapis.com/vehicle-camera-frames/outputs/{session_id}/sim_global.jpg")
        print(f"  Dashboard    : https://intersection-twin-809600582102.us-east1.run.app/dashboard")


def main():
    parser = argparse.ArgumentParser(description="Send video frames to Cloud Run stitching service")
    parser.add_argument("--video1",   default=VIDEO_1, help="Path to cam1 video file")
    parser.add_argument("--video2",   default=VIDEO_2, help="Path to cam2 video file")
    parser.add_argument("--interval", type=float, default=INTERVAL, help="Seconds between frame captures")
    parser.add_argument("--once",     action="store_true", help="Send just one frame pair and exit")
    parser.add_argument("--start",    type=float, default=0.0, help="Start position in video (seconds)")
    args = parser.parse_args()

    check_ffmpeg()

    # Check video files exist
    for path in [args.video1, args.video2]:
        if not os.path.exists(path):
            print(f"ERROR: Video not found: {path}")
            print("  Place your videos in the same folder as this script")
            print("  or pass --video1 and --video2 with the full path.")
            sys.exit(1)

    # Get durations for looping
    duration1 = get_video_duration(args.video1)
    duration2 = get_video_duration(args.video2)

    if duration1 == 0 or duration2 == 0:
        print("ERROR: Could not read video duration. Is ffprobe installed?")
        sys.exit(1)

    print("=" * 60)
    print(f"Video 1  : {args.video1} ({duration1:.1f}s)")
    print(f"Video 2  : {args.video2} ({duration2:.1f}s)")
    print(f"Interval : every {args.interval}s")
    print(f"Endpoint : {UPLOAD_URL}")
    print("=" * 60)
    print("Sending frames. Press Ctrl+C to stop.\n")

    position = args.start

    try:
        while True:
            t1 = position % duration1
            t2 = position % duration2

            # Use shared session ID for this frame pair
            session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

            print(f"[t={position:>7.1f}s]  Session: {session_id}")

            for video_path, cam_id, frame_ts in [
                (args.video1, "cam1", t1),
                (args.video2, "cam2", t2),
            ]:
                print(f"  Extracting {cam_id} frame @ {frame_ts:.1f}s...")

                # Extract frame
                frame_path = extract_frame(video_path, frame_ts)
                if frame_path is None:
                    print(f"  [SKIP] Failed to extract frame from {video_path}")
                    continue

                frame_size = os.path.getsize(frame_path) / 1024
                print(f"  Frame size: {frame_size:.1f} KB — sending to Cloud Run...")

                # Send to Cloud Run
                result = send_frame(frame_path, cam_id, session_id)
                process_result(result, cam_id, session_id)

                # Clean up temp file
                try:
                    os.unlink(frame_path)
                except OSError:
                    pass

            if args.once:
                print("\nDone! (--once flag set)")
                break

            position += args.interval
            print(f"\nWaiting {args.interval}s...\n" + "-" * 60)
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
