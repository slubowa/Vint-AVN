from nuscenes.nuscenes import NuScenes

# Initialize NuScenes
nusc = NuScenes(version='v1.0-mini', dataroot='/Users/simo/Documents/Thesis/Nuscene', verbose=True)

# Define output file
output_file = "nuscenes_output.txt"

# Open file for writing
with open(output_file, "w") as file:
    # Iterate through samples (keyframes)
    for sample in nusc.sample:
        cam_front_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])

        # Write data to file
        file.write(f"Sample Token: {sample['token']}\n")
        file.write(f"Sample Data Token: {cam_front_data['token']}\n")
        file.write(f"Filename: {cam_front_data['filename']}\n\n")

print(f"Output written to {output_file}")