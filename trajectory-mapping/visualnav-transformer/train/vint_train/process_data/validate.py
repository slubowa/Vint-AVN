import os
import pickle
import numpy as np
from PIL import Image

# Define the main dataset directory
data_folder = "nu_output"  

# Check if the data folder exists
if not os.path.exists(data_folder):
    raise FileNotFoundError(f"Dataset folder '{data_folder}' not found!")

# Get a list of all scene folders
scenes = sorted([s for s in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, s))])

# Function to validate a scene's structure
def check_scene_structure(scene):
    scene_path = os.path.join(data_folder, scene)
    images = sorted([f for f in os.listdir(scene_path) if f.endswith(".jpg")])
    traj_file = os.path.join(scene_path, "traj_data.pkl")

    if not images:
        print(f"Scene {scene} has no images!")
    else:
        print(f"Scene {scene} - {len(images)} images found, first: {images[0]}, last: {images[-1]}")

    if not os.path.exists(traj_file):
        print(f"Missing traj_data.pkl in {scene}!")
    else:
        print(f"Found traj_data.pkl in {scene}")

# Run checks on all scenes
for scene in scenes:
    check_scene_structure(scene)

# get_data_path function for ViNT Dataset Loader
def get_data_path(data_folder: str, scene: str, frame_num: int) -> str:
    """ Generate correct path to an image file for ViNT data loading. """
    return os.path.join(data_folder, scene, f"{frame_num}.jpg")

# Test with a random scene
test_scene = scenes[0] if scenes else None
if test_scene:
    test_img_path = get_data_path(data_folder, test_scene, 0)  # First frame
    print(f"\n🔍 Testing get_data_path: {test_img_path}")
    if os.path.exists(test_img_path):
        print(f"Path is correct and image exists: {test_img_path}")
    else:
        print(f"Path does not exist: {test_img_path}. Check folder structure!")
