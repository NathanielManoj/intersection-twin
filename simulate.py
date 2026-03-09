"""
simulate.py
-----------
Renders three simulated top-down intersection maps.

Camera positions (bird's eye view of the 5x5m intersection):

    cam2 (0,0) ─────────────── (5,0)
         |                        |
         |      intersection      |
         |                        |
      (0,5) ─────────────── cam1 (5,5)

Coordinate transform logic:
    Each camera's BEV has its own local coordinate system with (0,0)
    at the top-left of its image. Since each camera is at a different
    corner looking inward, we must rotate each camera's local coords
    into the shared global world frame.

    cam1 is at world (5,5) looking toward (0,0):
        Its BEV top-left is near world (5,5) and bottom-right near (0,0).
        So:  global_x = 5.0 - local_x
             global_y = 5.0 - local_y

    cam2 is at world (0,0) looking toward (5,5):
        Its BEV top-left is near world (0,0) — already aligned.
        So:  global_x = local_x
             global_y = local_y

Outputs:
    outputs/sim_cam1.jpg    -- cam1 objects in global frame (green)
    outputs/sim_cam2.jpg    -- cam2 objects in global frame (blue)
    outputs/sim_global.jpg  -- merged, deduplicated global digital twin
"""

import cv2
import numpy as np
import os

INTERSECTION_M    = 5.0
CANVAS_PX         = 800
DEDUP_THRESHOLD_M = 0.6
M_TO_PX           = CANVAS_PX / INTERSECTION_M   # 160 px/m

# BGR colours
C_BG     = ( 20,  20,  20)
C_GRID   = ( 55,  55,  55)
C_BORDER = (200, 200, 200)
C_CAM1   = (100, 230, 100)   # green
C_CAM2   = (100, 100, 230)   # blue
C_BOTH   = (100, 230, 230)   # cyan
C_TEXT   = (255, 255, 255)

# Fixed fallback sizes if world_w/world_h are missing
OBJECT_SIZES = {
    "car":        (0.3, 0.5),
    "truck":      (0.4, 0.7),
    "bus":        (0.4, 0.9),
    "motorcycle": (0.15, 0.3),
    "bicycle":    (0.12, 0.25),
    "person":     (0.1, 0.1),
    "chair":      (0.1, 0.1),
    "laptop":     (0.08, 0.08),
    "default":    (0.12, 0.12),
}

# Camera positions in world frame (metres)
CAM_POSITIONS = {
    "cam1": (5.0, 5.0),   # bottom-right corner
    "cam2": (0.0, 0.0),   # top-left corner
}


# ── Coordinate transforms ─────────────────────────────────────────────────────

def _to_global(det):
    """
    Converts a detection's local BEV coordinates to the shared global
    world frame based on which camera it came from.

    cam1 at (5,5): its local (0,0) is near world (5,5), so we flip both axes.
    cam2 at (0,0): its local (0,0) is already world (0,0), no change needed.
    """
    cam_id = det.get("cam_id", "cam1")
    x, y   = det["world_x"], det["world_y"]

    if cam_id == "cam1":
        # cam1 looks from (5,5) toward (0,0) — flip both axes
        gx = INTERSECTION_M - x
        gy = INTERSECTION_M - y
    else:
        # cam2 looks from (0,0) toward (5,5) — already in global frame
        gx = x
        gy = y

    return {**det, "world_x": gx, "world_y": gy}


def _to_px(wx, wy):
    return int(wx * M_TO_PX), int(wy * M_TO_PX)


# ── Canvas ────────────────────────────────────────────────────────────────────

def _blank_canvas():
    canvas = np.full((CANVAS_PX, CANVAS_PX, 3), C_BG, dtype=np.uint8)

    # Grid every 0.5 m
    step = int(0.5 * M_TO_PX)
    for i in range(0, CANVAS_PX, step):
        cv2.line(canvas, (i, 0), (i, CANVAS_PX), C_GRID, 1)
        cv2.line(canvas, (0, i), (CANVAS_PX, i), C_GRID, 1)

    # Intersection boundary
    cv2.rectangle(canvas, _to_px(0, 0),
                  _to_px(INTERSECTION_M, INTERSECTION_M), C_BORDER, 2)

    # Axis labels
    cv2.putText(canvas, "0,0", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_TEXT, 1)
    cv2.putText(canvas, "5,5", (CANVAS_PX - 30, CANVAS_PX - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_TEXT, 1)

    # 1 m scale bar
    bx, by = 20, CANVAS_PX - 20
    cv2.line(canvas, (bx, by), (bx + int(M_TO_PX), by), C_TEXT, 2)
    cv2.putText(canvas, "1m", (bx, by - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_TEXT, 1)

    return canvas


def _draw_camera(canvas, cam_id, color):
    """Draws a camera marker (triangle + label) at its world position."""
    pos = CAM_POSITIONS[cam_id]
    px  = _to_px(*pos)

    # Triangle pointing inward toward intersection centre
    cx, cy = px
    tri = np.array([[cx, cy-12], [cx-8, cy+8], [cx+8, cy+8]])
    cv2.drawContours(canvas, [tri], 0, color, -1)
    cv2.putText(canvas, cam_id.upper(), (cx - 12, cy - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def _draw_object(canvas, det, color):
    w_m, h_m = OBJECT_SIZES.get(det["label"], OBJECT_SIZES["default"])

    hw   = max(int(w_m * M_TO_PX / 2), 6)
    hh   = max(int(h_m * M_TO_PX / 2), 6)
    cx, cy = _to_px(det["world_x"], det["world_y"])

    x1 = max(cx - hw, 0);  y1 = max(cy - hh, 0)
    x2 = min(cx + hw, CANVAS_PX - 1);  y2 = min(cy + hh, CANVAS_PX - 1)

    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), C_TEXT,  1)
    cv2.putText(canvas, det["label"], (x1, max(y1 - 4, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_TEXT, 1)


def _legend(canvas, entries):
    for i, (text, color) in enumerate(entries):
        y = 20 + i * 22
        cv2.rectangle(canvas, (8, y - 10), (18, y), color, -1)
        cv2.putText(canvas, text, (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_TEXT, 1)


# ── Deduplication ─────────────────────────────────────────────────────────────

def _deduplicate(cam1_global, cam2_global):
    """
    Merges detections that are already in the global frame.
    Same label + distance < DEDUP_THRESHOLD_M => averaged into one object.
    """
    merged = [{**d, "source": "cam1"} for d in cam1_global]

    for d2 in cam2_global:
        matched_idx = None
        for i, d1 in enumerate(merged):
            if d1["label"] != d2["label"]:
                continue
            dist = np.hypot(d1["world_x"] - d2["world_x"],
                            d1["world_y"] - d2["world_y"])
            if dist < DEDUP_THRESHOLD_M:
                matched_idx = i
                break

        if matched_idx is not None:
            d1 = merged[matched_idx]
            merged[matched_idx] = {
                **d1,
                "world_x":    (d1["world_x"] + d2["world_x"]) / 2,
                "world_y":    (d1["world_y"] + d2["world_y"]) / 2,
                "confidence": max(d1["confidence"], d2["confidence"]),
                "source":     "both",
            }
        else:
            merged.append({**d2, "source": "cam2"})

    return merged


# ── Public API ────────────────────────────────────────────────────────────────

def render_all(cam1_dets: list, cam2_dets: list, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # Convert all detections to global world frame
    cam1_global = [_to_global(d) for d in cam1_dets]
    cam2_global = [_to_global(d) for d in cam2_dets]

    color_map = {"cam1": C_CAM1, "cam2": C_CAM2, "both": C_BOTH}

    # sim_cam1
    c1 = _blank_canvas()
    _draw_camera(c1, "cam1", C_CAM1)
    _draw_camera(c1, "cam2", C_CAM2)
    for det in cam1_global:
        _draw_object(c1, det, C_CAM1)
    cv2.putText(c1, "Camera 1 View", (10, CANVAS_PX - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_CAM1, 2)
    _legend(c1, [("Cam1 detections", C_CAM1)])
    cv2.imwrite(os.path.join(output_dir, "sim_cam1.jpg"), c1)

    # sim_cam2
    c2 = _blank_canvas()
    _draw_camera(c2, "cam1", C_CAM1)
    _draw_camera(c2, "cam2", C_CAM2)
    for det in cam2_global:
        _draw_object(c2, det, C_CAM2)
    cv2.putText(c2, "Camera 2 View", (10, CANVAS_PX - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_CAM2, 2)
    _legend(c2, [("Cam2 detections", C_CAM2)])
    cv2.imwrite(os.path.join(output_dir, "sim_cam2.jpg"), c2)

    # sim_global — merged and deduplicated
    merged = _deduplicate(cam1_global, cam2_global)
    cg = _blank_canvas()
    _draw_camera(cg, "cam1", C_CAM1)
    _draw_camera(cg, "cam2", C_CAM2)
    for det in merged:
        _draw_object(cg, det, color_map[det["source"]])
    cv2.putText(cg, "Global Digital Twin", (10, CANVAS_PX - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_TEXT, 2)
    _legend(cg, [("Cam1 only", C_CAM1), ("Cam2 only", C_CAM2), ("Both", C_BOTH)])
    cv2.imwrite(os.path.join(output_dir, "sim_global.jpg"), cg)

    n_both = sum(1 for d in merged if d["source"] == "both")
    print(f"cam1: {len(cam1_dets)} objects | cam2: {len(cam2_dets)} objects")
    print(f"Global: {len(merged)} unique objects ({n_both} seen by both)")
    print(f"Saved sim_cam1.jpg, sim_cam2.jpg, sim_global.jpg → {output_dir}/")