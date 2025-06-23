# waymo_to_vint_pkl.py
# Converts Waymo TFRecord data into ViNT-compatible image + .pkl trajectory format

import os
import math
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2 as open_dataset

# --- Utility: Get yaw angle (rotation around Z) from rotation matrix ---
def get_yaw_from_transform(transform):
    matrix = np.array(transform).reshape(4, 4)
    rot = matrix[:3, :3]
    yaw = math.atan2(rot[1, 0], rot[0, 0])
    return yaw

# --- Utility: Convert global x, y coords to local (ego-frame) coordinates ---
def to_local_coords(xy_global, ref_xy, ref_yaw):
    dx, dy = xy_global[0] - ref_xy[0], xy_global[1] - ref_xy[1]
    cos_yaw = math.cos(-ref_yaw)
    sin_yaw = math.sin(-ref_yaw)
    x_local = dx * cos_yaw - dy * sin_yaw
    y_local = dx * sin_yaw + dy * cos_yaw
    return [x_local, y_local]

# --- Main extraction function ---
def process_waymo_tfrecord(tfrecord_path, output_dir, context_size=5, spacing=1, camera_name='FRONT'):
    os.makedirs(output_dir, exist_ok=True)
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')

    all_entries = []
    frame_buffer = []

    print("Reading frames from:", tfrecord_path)

    # --- Step 1: Load all frames and store pose/image info ---
    for data in tqdm(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        # Get pose info
        transform = frame.pose.transform
        x = transform[3]
        y = transform[7]
        yaw = get_yaw_from_transform(transform)

        # Get FRONT camera image
        for img in frame.images:
            if open_dataset.CameraName.Name.Name(img.name) == camera_name:
                image_data = tf.image.decode_jpeg(img.image).numpy()
                frame_buffer.append({
                    'image': image_data,
                    'x': x,
                    'y': y,
                    'yaw': yaw
                })
                break

    # --- Step 2: Build samples with context (ViNT needs sequences) ---
    sample_id = 0
    for i in range(len(frame_buffer) - context_size * spacing):
        ref_frame = frame_buffer[i]
        ref_pos = [ref_frame['x'], ref_frame['y']]
        ref_yaw = ref_frame['yaw']

        pos_seq = []
        yaw_seq = []

        for j in range(context_size):
            idx = i + j * spacing
            cur = frame_buffer[idx]
            cur_pos = [cur['x'], cur['y']]
            local_pos = to_local_coords(cur_pos, ref_pos, ref_yaw)

            pos_seq.append(local_pos)
            yaw_seq.append(cur['yaw'])  # Global yaw (ViNT expects absolute yaw)

        # Save image
        image_filename = f"frame_{sample_id:05d}.jpg"
        image_path = os.path.join(output_dir, image_filename)
        tf.keras.utils.save_img(image_path, ref_frame['image'])

        # Save .pkl entry
        pkl_data = {
            'image_path': image_filename,
            'pos': pos_seq,  # Relative positions
            'yaw': yaw_seq   # Global yaws
        }

        pkl_filename = f"sample_{sample_id:05d}.pkl"
        pkl_path = os.path.join(output_dir, pkl_filename)

        with open(pkl_path, 'wb') as f:
            pickle.dump(pkl_data, f)

        sample_id += 1

    print(f"Saved {sample_id} samples with context size {context_size}.")
    print("Output directory:", output_dir)

# Example usage (you can comment this out if running as a module)
process_waymo_tfrecord(
tfrecord_path='your_file.tfrecord',
output_dir='waymo_dataset/',
context_size=5,
spacing=1)
