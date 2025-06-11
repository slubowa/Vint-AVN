import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import torchvision.transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONTEXT_SIZE = 5
GOAL_OFFSET = 10
IMAGE_SIZE = (224, 224)

TRIALS_PATH = "./vint_train/data/waymo_vint/segment-10226164909075980558_180_000_200_000_with_camera_labels"
TRAJ_DATA_PATH = os.path.join(TRIALS_PATH, "traj_data.pkl")
CHECKPOINT_PATH = "./logs/vint-release/vint-5c_2025_03_23_04_37_02/latest.pth"

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def preprocess_image(img_path):
    transform = T.Compose([
        T.Resize(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)

def load_trajectory_data(pkl_path):
    import pickle
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data

def transform_local_waypoints_to_global(local_wps, anchor_pos, anchor_yaw, scale=1.0):
    out = []
    for xloc, yloc, c_loc, s_loc in local_wps:
        xloc *= scale
        yloc *= scale
        th_loc = np.arctan2(s_loc, c_loc)
        th_global = anchor_yaw + th_loc
        c_glob = np.cos(th_global)
        s_glob = np.sin(th_global)
        X_glob = anchor_pos[0] + (xloc * np.cos(anchor_yaw) - yloc * np.sin(anchor_yaw))
        Y_glob = anchor_pos[1] + (xloc * np.sin(anchor_yaw) + yloc * np.cos(anchor_yaw))
        out.append([X_glob, Y_glob, c_glob, s_glob])
    return np.array(out, dtype=np.float32)

def main():
    log("[START] Script to stitch predicted waypoints into a full route.")

    if not os.path.exists(CHECKPOINT_PATH):
        log(f"[ERROR] No checkpoint found: {CHECKPOINT_PATH}")
        sys.exit(1)

    log(f"[INFO] Loading checkpoint from: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if "model" not in ckpt:
        log("[ERROR] 'model' key not in checkpoint. Expected full model object.")
        sys.exit(1)

    model = ckpt["model"]
    model.to(DEVICE)
    model.eval()
    log("[INFO] Loaded ViNT model successfully.")

    data = load_trajectory_data(TRAJ_DATA_PATH)
    gt_positions = data["position"]
    gt_yaws = data["yaw"]
    T_frames = len(gt_positions)
    log(f"[INFO] Loaded {T_frames} frames from trajectory data.")

    all_jpgs = sorted([f for f in os.listdir(TRIALS_PATH) if f.endswith(".jpg")])
    max_frame_idx = len(all_jpgs)
    log(f"[INFO] Found {max_frame_idx} image frames in directory.")

    predicted_global_all = []

    num_steps = max_frame_idx - (CONTEXT_SIZE + 1) - GOAL_OFFSET + 1
    if num_steps <= 0:
        log("[ERROR] Not enough frames to run inference.")
        return

    for i in range(num_steps):
        log(f"[STEP {i}] Loading context frames...")
        context_list = []
        valid = True
        for c in range(CONTEXT_SIZE + 1):
            frame_idx = i + c
            img_path = os.path.join(TRIALS_PATH, f"{frame_idx}.jpg")
            if not os.path.exists(img_path):
                log(f"[WARN] Missing context frame: {img_path}")
                valid = False
                break
            img_tensor = preprocess_image(img_path).to(DEVICE)
            context_list.append(img_tensor)

        if not valid:
            log(f"[SKIP] Incomplete context at step {i}, skipping.")
            continue

        obs_img = torch.cat(context_list, dim=1)

        goal_idx = i + GOAL_OFFSET
        goal_path = os.path.join(TRIALS_PATH, f"{goal_idx}.jpg")
        if not os.path.exists(goal_path):
            log(f"[SKIP] Missing goal image at step {i}, skipping.")
            continue
        goal_img = preprocess_image(goal_path).to(DEVICE)

        log(f"[STEP {i}] Running model inference...")
        with torch.no_grad():
            dist_pred, action_pred = model(obs_img, goal_img)

        waypoints_local = action_pred[0].cpu().numpy()

        if i >= T_frames:
            log(f"[SKIP] Ground-truth trajectory exhausted at step {i}.")
            continue

        anchor_pos = gt_positions[i]
        anchor_yaw = gt_yaws[i]

        log(f"[STEP {i}] Transforming predicted waypoints to global coordinates...")
        wps_global = transform_local_waypoints_to_global(waypoints_local, anchor_pos, anchor_yaw, scale=1.0)
        predicted_global_all.append(wps_global[:, :2])

        log(f"[STEP {i}] Stored {len(wps_global)} predicted global waypoints.")

    if not predicted_global_all:
        log("[WARN] No predictions made.")
        return

    predicted_global_all = np.concatenate(predicted_global_all, axis=0)
    log("[DONE] Inference completed. Plotting results...")

    plt.figure(figsize=(8, 6))
    plt.plot(gt_positions[:, 0], gt_positions[:, 1], "g--", label="GT entire route")
    plt.plot(predicted_global_all[:, 0], predicted_global_all[:, 1], "r.", label="Predicted")
    plt.legend()
    plt.axis("equal")
    plt.title("Full Route: Predicted vs. Ground Truth")
    plt.show()

    log("[DONE] Route plotted successfully.")

if __name__ == "__main__":
    main()