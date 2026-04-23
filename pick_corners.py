
"""
pick_corners.py
---------------
Interactive helper to compute homography matrices for each camera.
It walks the user through a two-step corner picking process so that
camera video can be warped into a common bird's-eye view coordinate frame.
"""

import cv2
import numpy as np
import os
import argparse

OUTPUT_SIZE = 800
WIN         = "picker"

LABELS_STAGE1 = ["TL", "TR", "BL", "BR"]
LABELS_STAGE2 = ["TL (0,0)", "TR (5,0)", "BL (0,5)", "BR (5,5)"]
COLORS        = [(0,0,255), (0,165,255), (255,0,0), (0,255,0)]

pts = []   # shared global for mouse callback


def on_click(event, x, y, flags, param):
    """Mouse callback that records clicked corner points."""
    if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
        pts.append([x, y])


def draw(img, labels, instruction):
    out = img.copy()
    for i, pt in enumerate(pts):
        cv2.circle(out, tuple(pt), 8, COLORS[i], -1)
        cv2.putText(out, labels[i], (pt[0]+10, pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[i], 2)
    if len(pts) == 4:
        order = [0, 1, 3, 2, 0]
        for i in range(4):
            cv2.line(out, tuple(pts[order[i]]), tuple(pts[order[i+1]]), (255,255,255), 1)
        cv2.putText(out, "ENTER=confirm  R=reset", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    else:
        cv2.putText(out, instruction, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    return out


def pick_points(img, labels, instruction):
    """Display an image and let the user click exactly 4 points."""
    global pts
    pts = []

    cv2.namedWindow(WIN)
    cv2.imshow(WIN, draw(img, labels, instruction))
    cv2.waitKey(100)
    cv2.setMouseCallback(WIN, on_click)

    while True:
        cv2.imshow(WIN, draw(img, labels, instruction))
        key = cv2.waitKey(20) & 0xFF
        if key == ord('r'):
            pts = []
        elif key == 13 and len(pts) == 4:   
            break
        elif key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("Cancelled.")

    cv2.destroyAllWindows()
    return [p[:] for p in pts]   


def warp(img, src_pts, size=OUTPUT_SIZE):
    """Compute a perspective transform and warp the image to a square BEV."""
    src = np.float32(src_pts)
    dst = np.float32([[0,0],[size,0],[0,size],[size,size]])
    H   = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, H, (size, size)), H


def save_corners(src_pts, cam_name):
    os.makedirs("config", exist_ok=True)
    path = f"config/corners_{cam_name}.txt"
    entries = zip(
        ["corner_0  world (0,0)", "corner_1  world (5,0)",
         "corner_2  world (0,5)", "corner_3  world (5,5)"],
        src_pts
    )
    with open(path, "w") as f:
        for label, (u, v) in entries:
            f.write(f"# {label}\n{int(u)} {int(v)}\n")
    print(f"Corners saved → {path}")


def main(image_path, cam_name):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open: {image_path}")

    # Resize for display if very large
    h, w   = img.shape[:2]
    scale1 = min(1.0, 1200 / max(h, w))
    disp   = cv2.resize(img, (int(w*scale1), int(h*scale1))) if scale1 < 1 else img.copy()

    # Stage 1: rough warp 
    print("\n── Stage 1 ──────────────────────────────────────────")
    print("Click 4 points around the intersection area (any rough rectangle).")
    print("Order: Top-Left → Top-Right → Bottom-Left → Bottom-Right")
    print("R = reset  |  ENTER = confirm\n")

    rough_pts  = pick_points(disp, LABELS_STAGE1, f"Stage 1 — click: {LABELS_STAGE1[0]}")

    # Scale back to original image coords, then warp
    rough_orig = [[int(x/scale1), int(y/scale1)] for x, y in rough_pts]
    rough_bev, H1 = warp(img, rough_orig)   # H1: original image -> rough BEV

    # ── Stage 2: pick 5x5 m label corners on rough BEV 
    print("── Stage 2 ──────────────────────────────────────────")
    print("The rough BEV is now shown.")
    print("Click the 4 corners of the 5x5 m ground label.")
    print("Order: Top-Left → Top-Right → Bottom-Left → Bottom-Right")
    print("R = reset  |  ENTER = confirm\n")

    label_pts = pick_points(rough_bev, LABELS_STAGE2,
                            f"Stage 2 — click label corner: {LABELS_STAGE2[0]}")

    # ── Final warp: warp rough BEV so label corners fill the canvas 
    final_bev, H2 = warp(rough_bev, label_pts)

    # Show final result
    preview = final_bev.copy()
    cv2.putText(preview, "Final BEV — press any key to save", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
    cv2.namedWindow(WIN)
    cv2.imshow(WIN, preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save 
    os.makedirs("config", exist_ok=True)

    bev_path = f"config/bev_{cam_name}.jpg"
    cv2.imwrite(bev_path, final_bev)
    print(f"BEV saved  → {bev_path}")

    # Save the combined homography (rough → final) for reference
    # H_combined maps directly from original image -> final BEV
    H_combined = H2 @ H1
    np.save(f"config/H_{cam_name}.npy", H_combined)

    # The corners in the final BEV are simply the 4 canvas corners
    # (the label corners now map to 0,0 / 800,0 / 0,800 / 800,800)
    final_corners = [[0, 0], [OUTPUT_SIZE, 0], [0, OUTPUT_SIZE], [OUTPUT_SIZE, OUTPUT_SIZE]]
    save_corners(final_corners, cam_name)

    print(f"\nDone. Run  python main.py  to detect and simulate.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--cam",   required=True, choices=["cam1", "cam2"])
    args = ap.parse_args()
    main(args.image, args.cam)