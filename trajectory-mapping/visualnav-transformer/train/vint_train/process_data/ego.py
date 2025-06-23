# Get the ego pose for this sample (Global to Ego transformation)
ego_pose_data = nusc.get('ego_pose', nusc.get('sample_data', sample['data']['CAM_FRONT'])['ego_pose_token'])

# Extract ego translation and rotation
T_ego = ego_pose_data['translation']  # [x, y, z]
R_ego = ego_pose_data['rotation']  # Quaternion

print(f"Ego Vehicle Translation (Global Frame): {T_ego}")
print(f"Ego Vehicle Rotation (Quaternion): {R_ego}")