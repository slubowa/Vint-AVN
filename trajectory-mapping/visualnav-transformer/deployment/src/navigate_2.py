import os
import sys
import torch
import numpy as np
from PIL import Image as PILImage
import argparse
import carla
import atexit
import time
import random
from collections import deque

# Import ViNT
sys.path.append("/home/paperspace/Documents/vint_project/trajectory-mapping/visualnav-transformer/train")
from vint_train.models.vint.vint import ViNT

# Initialize CARLA Client
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.load_world("Town01")

# Enable synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
settings.no_rendering_mode = False
world.apply_settings(settings)

blueprint_library = world.get_blueprint_library()

# Load the ViNT Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ViNT(late_fusion=False).to(device)

# Load model weights
MODEL_WEIGHTS_PATH = "/home/paperspace/Documents/vint_project/trajectory-mapping/visualnav-transformer/deployment/model_weights/latest.pth"
try:
    checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit(1)

model.eval()

# Rolling buffer for past images
context_size = 5
context_queue = deque(maxlen=context_size + 1)

# Function to check if a spawn point is free
def is_spawn_point_free(world, spawn_point, radius=2.0):
    actors = world.get_actors()
    for actor in actors:
        if actor.get_transform().location.distance(spawn_point.location) < radius:
            return False
    return True

# Spawn the vehicle
vehicle_bp = blueprint_library.find("vehicle.tesla.model3")
spawn_points = world.get_map().get_spawn_points()
random.shuffle(spawn_points)

vehicle = None
for spawn_point in spawn_points:
    if is_spawn_point_free(world, spawn_point):
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle:
            print(f"Vehicle spawned successfully at: {spawn_point}")
            break

if vehicle is None:
    print("ERROR: No free spawn points available!")
    time.sleep(2)

# Move the spectator camera to focus on the vehicle
spectator = world.get_spectator()
vehicle_transform = vehicle.get_transform()
spectator.set_transform(carla.Transform(
    vehicle_transform.location + carla.Location(z=10),  # Elevate camera above vehicle
    vehicle_transform.rotation
))


# Attach a camera sensor
camera_bp = blueprint_library.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "224")
camera_bp.set_attribute("image_size_y", "224")
camera_bp.set_attribute("fov", "90")

camera_transform = carla.Transform(carla.Location(x=1.5, z=2.0))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

print("CARLA setup complete! Running in headless mode.")

# Preprocess CARLA images for ViNT
def preprocess_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
    image_pil = PILImage.fromarray(array).resize((224, 224))
    
    image_tensor = torch.from_numpy(np.array(image_pil)).permute(2, 0, 1).float() / 255.0
    return image_tensor.unsqueeze(0).to(device)

# Update the rolling image queue
def update_context(image):
    image_tensor = preprocess_image(image)
    context_queue.append(image_tensor)

# Define goal dynamically
def get_goal_image():
    if len(context_queue) < context_size:
        return None
    return context_queue[0]  # Goal = first frame in queue

# Process the image with ViNT
def process_image(image):
    update_context(image)
    
    if len(context_queue) < context_size:
        print(f"[INFO] Skipping frame: Not enough past frames yet ({len(context_queue)}/{context_size}).")
        return None

    obs_img = torch.cat(list(context_queue), dim=1)
    goal_img = get_goal_image()

    if goal_img is None:
        print("[INFO] Skipping frame: No goal image available.")
        return None

    obsgoal_img = torch.cat([obs_img, goal_img], dim=1)

    try:
        with torch.no_grad():
            _, waypoints_pred = model(obsgoal_img, goal_img)
    except Exception as e:
        print(f"[ERROR] Model inference failed: {e}")
        return None

    return waypoints_pred[0].cpu().numpy()

# Move vehicle based on waypoints
def move_vehicle(vehicle, waypoints):
    if waypoints is None:
        return

    control = carla.VehicleControl()
    first_waypoint = waypoints[0]

    # Extract position and yaw
    x_offset, y_offset, _, yaw = first_waypoint

    # Convert to world coordinates
    vehicle_transform = vehicle.get_transform()
    forward_vector = vehicle_transform.get_forward_vector()
    
    new_x = vehicle_transform.location.x + x_offset * forward_vector.x - y_offset * forward_vector.y
    new_y = vehicle_transform.location.y + x_offset * forward_vector.y + y_offset * forward_vector.x
    new_yaw = vehicle_transform.rotation.yaw + np.degrees(yaw)

    # Compute steering angle
    angle_diff = (new_yaw - vehicle_transform.rotation.yaw + 180) % 360 - 180
    control.steer = np.clip(angle_diff / 45.0, -1.0, 1.0)

    # Adjust throttle and brake
    control.throttle = 0.5 if abs(angle_diff) < 10 else 0.3
    control.brake = 0.0

    # Apply control
    vehicle.apply_control(control)

    print(f"[INFO] Moving to ({new_x:.2f}, {new_y:.2f}) with yaw {new_yaw:.2f}")

# Camera callback function
def camera_callback(image):
    waypoints = process_image(image)
    if waypoints is not None:
        print(f"Predicted Waypoints: {waypoints}")
        move_vehicle(vehicle, waypoints)

# Attach callback to camera
camera.listen(lambda image: camera_callback(image))

# Main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy ViNT in CARLA (Headless Mode)")
    parser.add_argument("--model", "-m", default="vint", type=str, help="Model name")
    args = parser.parse_args()

    print("ViNT is running inside CARLA in headless mode.")

    try:
        while settings.synchronous_mode:
            world.tick()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Shutting down...")
