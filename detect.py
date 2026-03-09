
import cv2
import numpy as np
import os

INTERSECTION_M = 5.0
BEV_SIZE_PX    = 800
M_PER_PX       = INTERSECTION_M / BEV_SIZE_PX   


def _warp_points(pts, H):
    
    src = pts.reshape(-1, 1, 2).astype(np.float32)
    return cv2.perspectiveTransform(src, H).reshape(-1, 2)


def _px_to_world(px, py):
    return px * M_PER_PX, py * M_PER_PX


def run_detection(original_path: str,
                  bev_path: str,
                  H_path: str,
                  cam_id: str,
                  yolo_model,
                  yolo_conf: float = 0.4,
                  output_dir: str = "outputs") -> list:
 
    orig = cv2.imread(original_path)
    if orig is None:
        raise FileNotFoundError(f"Cannot open image: {original_path}")

    bev = cv2.imread(bev_path)
    if bev is None:
        raise FileNotFoundError(f"Cannot open BEV: {bev_path}")
    if bev.shape[:2] != (BEV_SIZE_PX, BEV_SIZE_PX):
        bev = cv2.resize(bev, (BEV_SIZE_PX, BEV_SIZE_PX))

    if not os.path.exists(H_path):
        raise FileNotFoundError(
            f"Homography not found: {H_path}\n"
            f"Run pick_corners.py --image <scene> --cam {cam_id} first."
        )
    H = np.load(H_path)

    results        = yolo_model(orig, conf=yolo_conf, verbose=False)[0]
    detections     = []
    annotated_orig = orig.copy()
    annotated_bev  = bev.copy()

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf  = float(box.conf[0])
        label = yolo_model.names[int(box.cls[0])]

        # Warp all 4 corners of the bounding box into BEV space
        corners        = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32)
        warped_corners = _warp_points(corners, H)

        # Skip if fewer than 2 corners land inside the BEV canvas
        inside = [(0 <= wx <= BEV_SIZE_PX and 0 <= wy <= BEV_SIZE_PX)
                  for wx, wy in warped_corners]
        if sum(inside) < 2:
            continue

        # Ground contact point = warped bottom-centre
        ground        = np.array([[(x1+x2)/2, float(y2)]], dtype=np.float32)
        warped_ground = _warp_points(ground, H)[0]
        wx, wy        = _px_to_world(warped_ground[0], warped_ground[1])

        # World size from warped box extents
        world_w = max((warped_corners[:,0].max() - warped_corners[:,0].min()) * M_PER_PX, 0.3)
        world_h = max((warped_corners[:,1].max() - warped_corners[:,1].min()) * M_PER_PX, 0.3)

        detections.append({
            "label":      label,
            "confidence": conf,
            "world_x":    wx,
            "world_y":    wy,
            "world_w":    world_w,
            "world_h":    world_h,
            "bev_box":    warped_corners,   
            "cam_id":     cam_id,
        })

        # Draw on original image
        cv2.rectangle(annotated_orig, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(annotated_orig, f"{label} {conf:.2f}",
                    (x1, max(y1-6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Draw warped box on BEV
        pts = warped_corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(annotated_bev, [pts], isClosed=True, color=(0,255,0), thickness=2)
        bx, by = int(warped_ground[0]), int(warped_ground[1])
        cv2.putText(annotated_bev, f"{label} ({wx:.1f},{wy:.1f})m",
                    (bx+6, by), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f"annotated_orig_{cam_id}.jpg"), annotated_orig)
    cv2.imwrite(os.path.join(output_dir, f"bev_annotated_{cam_id}.jpg"),  annotated_bev)

    return detections