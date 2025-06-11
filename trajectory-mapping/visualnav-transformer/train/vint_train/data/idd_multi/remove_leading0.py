import os

# Define the directory containing the images
base_path = os.getcwd()

# Loop through each route folder
for route in ["route1", "route2", "route3"]:
    folder_path = os.path.join(base_path, route)

    print(f"🔍 Checking folder: {folder_path}")  # Debug print

    if os.path.exists(folder_path):
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(".jpg"):
                old_path = os.path.join(folder_path, filename)

                # Extract the numeric part, remove leading zeros
                try:
                    num_part = filename.split(".")[0]
                    new_filename = f"{int(num_part)}.jpg"  # Convert to int to remove leading zeros
                    new_path = os.path.join(folder_path, new_filename)

                    # Print debugging info
                    print(f"📂 Found file: {filename} -> New name: {new_filename}")

                    # Rename file if necessary
                    if old_path != new_path:
                        os.rename(old_path, new_path)
                        print(f"✅ Renamed: {old_path} -> {new_path}")
                    else:
                        print(f"⚠️ Skipping rename: {filename} already correct.")
                except ValueError as e:
                    print(f"❌ Error processing file {filename}: {e}")

    else:
        print(f"🚨 Folder does not exist: {folder_path}")

print("🎉 Filename update process completed!")
