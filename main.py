

import os
import sys
import argparse
import cv2
import numpy as np


BEV_SIZE_PX = 800


def load_yolo(model_name="yolo11m.pt"):
    from ultralytics import YOLO
    model = YOLO(model_name)
    model.fuse()
    return model


def check_setup(cam1_image, cam2_image):
    required = {
        "cam1 image":       cam1_image,
        "cam2 image":       cam2_image,
        "cam1 homography":  "config/H_cam1.npy",
        "cam2 homography":  "config/H_cam2.npy",
    }
    missing = {k: v for k, v in required.items() if not os.path.exists(v)}
    if missing:
        print("[ERROR] Missing files:")
        for name, path in missing.items():
            print(f"  {name}: {path}")
        print("\nFor missing homography files run:")
        print("  python pick_corners.py --image <cam1_image> --cam cam1")
        print("  python pick_corners.py --image <cam2_image> --cam cam2")
        sys.exit(1)


def warp_bev(image_path, H_path, output_path):
  
    img = cv2.imread(image_path)
    H   = np.load(H_path)
    bev = cv2.warpPerspective(img, H, (BEV_SIZE_PX, BEV_SIZE_PX))
    cv2.imwrite(output_path, bev)
    return bev


def main():
    ap = argparse.ArgumentParser(description="Intersection Digital Twin")
    ap.add_argument("--cam1",   default="config/cam1_scene.jpg",
                    help="Path to cam1 image")
    ap.add_argument("--cam2",   default="config/cam2_scene.jpg",
                    help="Path to cam2 image")
    ap.add_argument("--conf",   type=float, default=0.4,
                    help="YOLO confidence threshold (default: 0.4)")
    ap.add_argument("--output", default="outputs",
                    help="Output folder (default: outputs/)")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    check_setup(args.cam1, args.cam2)

  
    bev_cam1_path = os.path.join(args.output, "bev_cam1.jpg")
    bev_cam2_path = os.path.join(args.output, "bev_cam2.jpg")

    warp_bev(args.cam1, "config/H_cam1.npy", bev_cam1_path)
    warp_bev(args.cam2, "config/H_cam2.npy", bev_cam2_path)

    yolo = load_yolo()

    from detect import run_detection

    cam1_dets = run_detection(
        original_path = args.cam1,
        bev_path      = bev_cam1_path,
        H_path        = "config/H_cam1.npy",
        cam_id        = "cam1",
        yolo_model    = yolo,
        yolo_conf     = args.conf,
        output_dir    = args.output,
    )

    cam2_dets = run_detection(
        original_path = args.cam2,
        bev_path      = bev_cam2_path,
        H_path        = "config/H_cam2.npy",
        cam_id        = "cam2",
        yolo_model    = yolo,
        yolo_conf     = args.conf,
        output_dir    = args.output,
    )

    print(f"cam1: {len(cam1_dets)} objects detected")
    print(f"cam2: {len(cam2_dets)} objects detected")

    from simulate import render_all
    render_all(cam1_dets, cam2_dets, output_dir=args.output)


if __name__ == "__main__":
    main()