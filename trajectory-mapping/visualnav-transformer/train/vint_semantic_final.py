
import os
import glob
import random
import pickle
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision.transforms as T

from Semantic.semantic_model import run_deeplab_segmentation

# USER CONFIGURATION
CSV_PATH      = "waymo_front_camera_intrinsics.csv"
TRIALS_PATH   = "./vint_train/data/waymo_vint"
CHECKPOINT    = "./logs/vint-release/vint-5c_2025_03_23_04_37_02/latest.pth"
CONTEXT_SIZE  = 5
GOAL_OFFSET   = 10
NUM_OVERLAYS  = 5
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_SEGMENTS = [
    "segment-10017090168044687777_6380_000_6400_000_with_camera_labels",
    "segment-10793018113277660068_2714_540_2734_540_with_camera_labels",
    "segment-10226164909075980558_180_000_200_000_with_camera_labels",
]

CAMERA_HEIGHT_M = 1.5
Z_OFFSET_M      = 8.0

def load_intrinsics(csv_file):
    """
    Read CSV of intrinsics and return dict segment intrinsics.
    """
    df = pd.read_csv(csv_file)
    intr_map = {}
    for _, row in df.iterrows():
        seg = str(row["segment_id"]).replace(".tfrecord_", "").replace(".tfrecord", "")
        intr_map[seg] = {k: float(row[k]) for k in ("fx","fy","cx","cy","k1","k2","p1","p2","k3")}
    return intr_map

def build_camera_matrices(intrinsics):
    """
    Build OpenCV camera matrix K and distortion coefficients from intrinsics.
    """
    K = np.array([
        [intrinsics["fx"], 0.0,            intrinsics["cx"]],
        [0.0,             intrinsics["fy"], intrinsics["cy"]],
        [0.0,             0.0,             1.0]
    ], dtype=np.float32)
    dist = np.array([
        intrinsics["k1"],
        intrinsics["k2"],
        intrinsics["p1"],
        intrinsics["p2"],
        intrinsics["k3"]
    ], dtype=np.float32)
    return K, dist

def preprocess_image(path):
    """
    Load and preprocess an image for ViNT (224x224).
    """
    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],
                    [0.229,0.224,0.225])
    ])
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(DEVICE)

def run_inference_for_step(seg_dir, step, model):
    """
    Run ViNT inference at a given step index.
    Returns local waypoints  and last context image path.
    """
    jpgs = sorted(
        glob.glob(os.path.join(seg_dir, "*.jpg")),
        key=lambda x: int(os.path.basename(x).split(".")[0])
    )
    # build observation tensor
    obs_tensors = [preprocess_image(jpgs[step + i]) for i in range(CONTEXT_SIZE + 1)]
    obs = torch.cat(obs_tensors, dim=1)
    # build goal tensor
    goal = preprocess_image(jpgs[step + GOAL_OFFSET])
    # inference
    with torch.no_grad():
        _, action = model(obs, goal)
    local = action[0].cpu().numpy()[:, :2]
    last_ctx = jpgs[step + CONTEXT_SIZE]
    return local, last_ctx

def local_to_global(local, anchor_pos, anchor_yaw):
    """
    Convert local waypoints (x forward, y left) to global XY.
    """
    c = np.cos(anchor_yaw)
    s = np.sin(anchor_yaw)
    X = anchor_pos[0] + local[:,0]*c - local[:,1]*s
    Y = anchor_pos[1] + local[:,0]*s + local[:,1]*c
    return np.stack([X, Y], axis=1)

def project_to_image(local_xy, K, dist):
    """
    Project local ground-plane points to pixel coordinates.
    """
    X = -local_xy[:,1]
    Y =  CAMERA_HEIGHT_M * np.ones_like(X)
    Z =  local_xy[:,0] + Z_OFFSET_M
    xyz = np.stack([X, Y, Z], axis=1).astype(np.float32)
    uv, _ = cv2.projectPoints(xyz,
                              rvec=np.zeros(3, dtype=np.float32),
                              tvec=np.zeros(3, dtype=np.float32),
                              cameraMatrix=K,
                              distCoeffs=dist)
    return uv.squeeze(1)

def plot_topdown(gt, pred):
    """
    Plot top-down ground truth vs. predicted route.
    """
    plt.figure(figsize=(8,6))
    plt.plot(gt[:,0], gt[:,1], 'g--', label='Ground truth')
    plt.plot(pred[:,0], pred[:,1], 'r.', label='Predicted')
    plt.legend()
    plt.axis('equal')
    plt.show()

def get_drivable_mask(image_path):
    """
    Run DeepLab segmentation at full resolution and return a boolean
    drivable-area mask (True=drivable).
    """
    semantic_map = run_deeplab_segmentation(image_path)
    return (semantic_map == 'road')

def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    # load ViNT model
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model = ckpt['model'].to(DEVICE).eval()
    logging.info('Loaded ViNT model')

    # load camera intrinsics
    intr_map = load_intrinsics(CSV_PATH)
    logging.info(f'Loaded intrinsics for {len(intr_map)} segments')

    for seg in TEST_SEGMENTS:
        seg_dir = os.path.join(TRIALS_PATH, seg)
        traj_pkl = os.path.join(seg_dir, 'traj_data.pkl')
        if not os.path.exists(traj_pkl):
            logging.warning(f'Missing traj_data.pkl for {seg}')
            continue

        # load ground truth trajectory
        with open(traj_pkl, 'rb') as f:
            data = pickle.load(f)
        gt_pos = data['position']
        num_steps = len(os.listdir(seg_dir)) - (CONTEXT_SIZE + 1) - GOAL_OFFSET + 1
        if num_steps <= 0:
            logging.warning(f'Not enough frames for {seg}')
            continue

        # inference over all steps
        local_preds, img_paths, global_preds = [], [], []
        for i in range(num_steps):
            local, imgp = run_inference_for_step(seg_dir, i, model)
            logging.info(f'{seg} step={i} local waypoints:\n{local}')
            local_preds.append(local)
            img_paths.append(imgp)

            # only first predicted waypoint per step
            first_wp = local[0:1, :]
            glob_pt  = local_to_global(first_wp, data['position'][i], data['yaw'][i])
            global_preds.append(glob_pt)

        # BEV route plot
        pred_all = np.concatenate(global_preds, axis=0)
        plot_topdown(gt_pos, pred_all)

        # pick steps to visualize
        steps = sorted(random.sample(range(num_steps), min(NUM_OVERLAYS, num_steps)))
        K, dist = build_camera_matrices(intr_map[seg])

        # overlay on images with semantic correction
        for step in steps:
            local_all = local_preds[step]
            imgp      = img_paths[step]
            mask      = get_drivable_mask(imgp)  # HxW bool

            uv_all = project_to_image(local_all, K, dist)

            # correct any off-road points
            for j, (xg, yg) in enumerate(local_all):
                u, v = int(round(uv_all[j,0])), int(round(uv_all[j,1]))
                if v < 0 or v >= mask.shape[0] or u < 0 or u >= mask.shape[1] or not mask[v,u]:
                    # try small lateral shifts
                    for delta in [0.2, -0.2, 0.4, -0.4]:
                        cand = np.array([[xg, yg + delta]], dtype=np.float32)
                        u2, v2 = project_to_image(cand, K, dist)[0]
                        ui, vi = int(round(u2)), int(round(v2))
                        if 0 <= vi < mask.shape[0] and 0 <= ui < mask.shape[1] and mask[vi, ui]:
                            uv_all[j] = [u2, v2]
                            break

            # plot horizon overlay
            img = plt.imread(imgp)
            plt.figure(figsize=(6,4))
            plt.imshow(img)
            plt.plot(uv_all[:,0], uv_all[:,1],
                     linestyle='-', marker='o', color='r', markersize=5)
            for j in range(len(uv_all)-1):
                dx = uv_all[j+1,0] - uv_all[j,0]
                dy = uv_all[j+1,1] - uv_all[j,1]
                plt.arrow(uv_all[j,0], uv_all[j,1], dx, dy,
                          shape='full', lw=0, length_includes_head=True,
                          head_width=5, head_length=7, color='r')
            plt.title(f'{seg} step={step} horizon w/ semantic adjust')
            plt.axis('off')
            plt.show()

if __name__ == '__main__':
    main()