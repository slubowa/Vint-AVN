import os
import cv2
import torch
import pandas as pd
import numpy as np
from torchvision import transforms

# Define paths
dataset_name = "IDD_Multimodal_Processed"
image_folders = ["d0", "d1", "d2"]
output_root = os.path.join("processed_vint_input", dataset_name)

# Earth's radius in meters
R = 6371000  # Approximate mean radius of Earth

# Function to normalize GPS coordinates
def normalize_gps(df):
    df['lat_norm'] = (df['latitude'] - df['latitude'].min()) / (df['latitude'].max() - df['latitude'].min())
    df['lon_norm'] = (df['longitude'] - df['longitude'].min()) / (df['longitude'].max() - df['longitude'].min())
    return df

# Function to compute yaw (heading direction change) BEFORE transformation
def compute_yaw(df):
    dx = np.diff(df["latitude"], prepend=df["latitude"].iloc[0])
    dy = np.diff(df["longitude"], prepend=df["longitude"].iloc[0])
    yaw = np.arctan2(dy, dx)  # Compute angle of movement in radians
    return yaw

# Function to convert latitude/longitude to ego-relative coordinates
def convert_to_relative(df):
    lat0, lon0 = df.iloc[0]["latitude"], df.iloc[0]["longitude"]
    df["dx"] = R * (df["longitude"] - lon0) * np.cos(np.radians(lat0))
    df["dy"] = R * (df["latitude"] - lat0)
    return df

# Image transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize for ViNT
    transforms.ToTensor()
])

for folder in image_folders:
    traj_path = os.path.join(output_root, folder)
    os.makedirs(traj_path, exist_ok=True)
    
    # Load all CSVs and merge them
    csv_paths = [f"{folder}/train.csv", f"{folder}/test.csv", f"{folder}/val.csv"]
    gps_data = pd.concat([pd.read_csv(csv) for csv in csv_paths], ignore_index=True)
    gps_data = gps_data.sort_values(by=["image_idx"]).reset_index(drop=True)
    
    # Compute yaw BEFORE transformation
    gps_data["yaw"] = compute_yaw(gps_data)
    
    # Normalize GPS and convert to ego-relative coordinates
    gps_data = normalize_gps(gps_data)
    gps_data = convert_to_relative(gps_data)
    
    data_list = []
    traj_data = []
    
    for _, row in gps_data.iterrows():
        image_idx = f"{int(row['image_idx']):06d}.jpg"
        image_path = os.path.join(folder, "leftCamImgs", image_idx)
        
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = transform(img)
            
            # Prepare input: (image, GPS-relative dx/dy, yaw)
            input_tensor = torch.cat([img_tensor, 
                                      torch.tensor([row['dx'], row['dy'], row['yaw']], dtype=torch.float32).unsqueeze(1).unsqueeze(2).expand(3, 224, 224)], dim=0)
            data_list.append(input_tensor)
            
            # Store trajectory data
            traj_data.append({"image_idx": row["image_idx"], "dx": row["dx"], "dy": row["dy"], "yaw": row["yaw"]})
    
    # Stack tensors and save
    if data_list:
        final_tensor = torch.stack(data_list)
        torch.save(final_tensor, os.path.join(traj_path, "traj_data.pkl"))
    
    # Save trajectory metadata
    pd.DataFrame(traj_data).to_pickle(os.path.join(traj_path, "traj_data.pkl"))
    print(f"Processed trajectory saved at {traj_path}")
