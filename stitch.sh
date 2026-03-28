#!/bin/bash
curl -X POST https://intersection-twin-809600582102.us-east1.run.app/stitch \
  -H "Content-Type: application/json" \
  -d '{
    "cam1_gcs": "images/cam1_scene.jpg",
    "cam2_gcs": "images/cam2_scene.jpg"
  }'
