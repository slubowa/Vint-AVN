import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet101

# Load pretrained DeepLabV3 model (trained on COCO stuff/Cityscapes style classes)
model = deeplabv3_resnet101(pretrained=True).eval().cuda() if torch.cuda.is_available() else deeplabv3_resnet101(pretrained=True).eval()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms for input images
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Cityscapes/COCO-stuff label mapping (simplified)
ID2LABEL = {
    0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat',
    9: 'chair', 10: 'cow', 11: 'dining table', 12: 'dog', 13: 'horse', 14: 'motorbike',
    15: 'person', 16: 'potted plant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tv/monitor'
    # NOTE: Actual class labels may vary if using Cityscapes-trained models
}

# For simplicity, assign label ID 7 (car) or road-like regions as 'road'
VALID_LABEL_IDS = {7, 6, 14, 15}  # car, bus, motorbike, person (assumed on road)



def run_deeplab_segmentation(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    input_tensor = input_tensor.cuda() if torch.cuda.is_available() else input_tensor

    with torch.no_grad():
        output = model(input_tensor)['out'][0]  # Shape: (num_classes, H, W)
        label_map = output.argmax(0).cpu().numpy()  # Shape: (H, W)

    semantic_map = np.full(label_map.shape, 'other', dtype=object)
    for id_val in np.unique(label_map):
        label_name = ID2LABEL.get(id_val, 'other')
        if id_val in VALID_LABEL_IDS:
            label_name = 'road'
        semantic_map[label_map == id_val] = label_name

    return semantic_map  # shape (H, W), values like 'road', 'other'


# Optional: quick test
#if __name__ == "__main__":
#    test_map = run_deeplab_segmentation("/home/paperspace/Documents/vint_project/trajectory-mapping/visualnav-transformer/data_test/bdd_frames/1/frame_0006.jpg")
 #   print("Unique labels:", np.unique(test_map))
