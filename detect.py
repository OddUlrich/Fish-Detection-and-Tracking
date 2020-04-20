# -*- coding: utf-8 -*-

# Import models and utils.
from models import Darknet
from tools.utils import load_classes, non_max_suppression

# Import necessary libraries.
import time, datetime, random
import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np

# Import visualization library.
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


# Load the pre-trained configuration and weights.
config_path = 'config/yolov3.cfg'
weights_path = 'config/yolov3.weights'
class_path = 'config/coco.names'
img_size = 416      # image size.
cof_threshold = 0.8     # confidence threshold.
nms_threshold = 0.4     # non-maximum suppression threshold.

model = Darknet(config_path, img_size)
model.load_weights(weights_path)
model.cuda()
model.eval()
classes = load_classes(class_path)
Tensor = torch.cuda.FloatTensor

def detection(img):
    """
    Runs the model and gets detections result of a image.
    
    Input:
        img - Image object from Image.open().
    Return:
        detections[0] - the detection result with shape (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    
    # Building a transformer for scaling and padding image.
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    img_w = round(img.size[0] * ratio)
    img_h = round(img.size[1] * ratio)
    
    img_transforms = transforms.Compose(
        [transforms.Resize((img_h, img_w)),
         transforms.Pad((max(int((img_h-img_w)/2), 0), max(int((img_w-img_h)/2), 0),
                         max(int((img_h-img_w)/2), 0), max(int((img_w-img_h)/2), 0)),
                        (128, 128, 128)),
         transforms.ToTensor(),
        ])
    
    # Convert image to Tensor.
    img_tensor = img_transforms(img).float()
    img_tensor = img_tensor.unsqueeze_(0)
    input_img = Variable(img_tensor.type(Tensor))
    
    # Run the model and get detections.
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, len(classes), cof_threshold, nms_threshold)
    
    return detections[0]
    
    
# Load image and run detection.
img_path = "data/street.jpg"
start_time = time.time()

img = Image.open(img_path)
detections = detection(img)

end_time = time.time()
inference_time = datetime.timedelta(seconds=(end_time-start_time))
print("Inference Time: %s" % (inference_time))

# Plot image.
img = np.array(img)
plt.figure()
fig, ax = plt.subplots(1, figsize=(12, 9))
ax.imshow(img)

# Unpad the image
pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
unpad_h = img_size - pad_y
unpad_w = img_size - pad_x

if detections is not None:
    unique_labels = detections[:, -1].cpu().unique()
    num_pred_class = len(unique_labels)
    
    # Set bounding box color map.
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    box_colors = random.sample(colors, num_pred_class)
    
    # Draw the bounding boxes.
    for x1, y1, x2, y2, object_conf, class_score, class_pred in detections:
        # Position adjustment with scaler.
        box_h = ((y2 - y1) / unpad_h) * img.shape[0]
        box_w = ((x2 - x1) / unpad_w) * img.shape[1]
        y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
        x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
    
        # Draw boxes with label texts.
        color = box_colors[int(np.where(unique_labels == int(class_pred))[0])]
        box = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(box)
        plt.text(x1, y1, s=classes[int(class_pred)], 
                 color='white', 
                 verticalalignment='top',
                 bbox={'color': color, 'pad': 0})
plt.axis('off')

# Save image with bounding boxes.
plt.savefig(img_path.replace(".jpg", "-detected.jpg"), bbox_inches='tight', pad_inches=0.0)
plt.show()


