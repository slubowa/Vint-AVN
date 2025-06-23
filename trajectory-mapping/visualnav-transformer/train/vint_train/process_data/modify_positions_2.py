import pickle
import numpy as np
import os

# Define dataset directory
DATASET_DIR = "nu_output"  # Update path if necessary

# Process all scenes
for scene_name in os.listdir(DATASET_DIR):
    scene_path = os.path.join(DATASET_DIR, scene_name)
    traj_file = os.path.join(scene_path, "traj_data.pkl")

    if not os.path.exists(traj_file):
        continue

    with open(traj_file, "rb") as f:
        traj_data = pickle.load(f)

    # Convert absolute positions to relative
    positions = np.array(traj_data["position"])
    origin = positions[0]  # Set first position as reference
    relative_positions = positions - origin

    # Update the file
    traj_data["position"] = relative_positions.tolist()

    with open(traj_file, "wb") as f:
        pickle.dump(traj_data, f)

    print(f" Converted positions to relative format for {scene_name}")

print("All scenes updated successfully!")