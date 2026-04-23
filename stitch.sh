#!/bin/bash
# stitch.sh — Manual test script for the /stitch endpoint.
# This sends a POST request to trigger stitching of two camera images.
# Useful for testing the API without uploading frames via /upload.
curl -X POST https://intersection-twin-809600582102.us-east1.run.app/stitch \
  -H "Content-Type: application/json" \
  -d '{
    "cam1_gcs": "images/cam1_scene.jpg",
    "cam2_gcs": "images/cam2_scene.jpg"
  }'
