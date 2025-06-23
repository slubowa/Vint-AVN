import os
import pickle
import numpy as np
import shutil
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Quaternion

# Set paths
NUSCENES_ROOT = "/home/paperspace/Documents/vint_project/nuscenes_data"  
OUTPUT_FOLDER = "/home/paperspace/Documents/vint_project/nu_output"  
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load NuScenes Full dataset
nusc = NuScenes(version='v1.0-trainval', dataroot=NUSCENES_ROOT, verbose=True)

# Process each scene
for scene in nusc.scene:
    scene_name = scene['name']
    scene_folder = os.path.join(OUTPUT_FOLDER, scene_name)
    os.makedirs(scene_folder, exist_ok=True)

    sample_token = scene['first_sample_token']
    traj_data = {"position": [], "yaw": []}
    count = 0

    while sample_token:
        sample = nusc.get('sample', sample_token)
        cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])  # Use only front camera
        img_path = os.path.join(NUSCENES_ROOT, cam_data['filename'])  # Image file
        output_img_path = os.path.join(scene_folder, f"{count}.jpg")

        # Copy image to output folder
        shutil.copy(img_path, output_img_path)

        # Get ego-pose (global world coordinates)
        ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
        global_xyz = np.array(ego_pose['translation'])  # (x, y, z)
        yaw = Quaternion(ego_pose['rotation']).yaw_pitch_roll[0]  # Extract yaw angle

        # Convert to (x, y) format (ignore z-axis)
        xy_position = global_xyz[:2]

        # Store trajectory data
        traj_data["position"].append(xy_position)
        traj_data["yaw"].append(yaw)

        sample_token = sample['next']
        count += 1

        print(f"Processed: {scene_name} - Frame {count}, Position: {xy_position}, Yaw: {yaw}")

    # Convert lists to numpy arrays
    traj_data["position"] = np.array(traj_data["position"])  # Shape [T, 2]
    traj_data["yaw"] = np.array(traj_data["yaw"])  # Shape [T,]

    # Save trajectory data in ViNT format
    with open(os.path.join(scene_folder, "traj_data.pkl"), "wb") as f:
        pickle.dump(traj_data, f)

print("NuScenes Full dataset successfully converted to ViNT format.")
