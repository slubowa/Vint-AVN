import carla
import time
import random

# Connect to CARLA
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

# Load the world
world = client.get_world()

# Get blueprint library
blueprint_library = world.get_blueprint_library()

# Select a Tesla Model 3
vehicle_bp = blueprint_library.find("vehicle.tesla.model3")

# Get a random spawn point
spawn_points = world.get_map().get_spawn_points()
spawn_point = random.choice(spawn_points)

# Spawn the vehicle
vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

if vehicle is None:
    print("❌ ERROR: Vehicle not spawned! Try again.")
    exit(1)

print(f"✅ Vehicle spawned at: {spawn_point.location}")

# Give CARLA time to update
time.sleep(2)

# Attach spectator camera above the vehicle and looking forward
spectator = world.get_spectator()

def update_spectator():
    """Moves the spectator camera just above the car and makes it look forward."""
    transform = vehicle.get_transform()
    location = transform.location + carla.Location(z=2.5)  # Just above vehicle
    rotation = carla.Rotation(pitch=0, yaw=transform.rotation.yaw)  # Looking forward
    spectator.set_transform(carla.Transform(location, rotation))

update_spectator()
print("📷 Spectator camera attached to vehicle.")

# Apply throttle to move forward
print("🚗 Moving vehicle forward...")
vehicle.apply_control(carla.VehicleControl(throttle=0.5))

# Move for 60 seconds
start_time = time.time()
while time.time() - start_time < 60:
    transform = vehicle.get_transform()
    print(f"📍 Vehicle location: x={transform.location.x:.2f}, y={transform.location.y:.2f}, yaw={transform.rotation.yaw:.2f}")

    # Continuously update the camera to follow the car
    update_spectator()

    time.sleep(0.1)  # Faster updates for smooth movement

# Stop vehicle
print("🛑 Stopping vehicle...")
vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

# Wait a moment before destroying
time.sleep(2)

# Destroy vehicle
vehicle.destroy()
print("✅ Test complete, vehicle destroyed.")
