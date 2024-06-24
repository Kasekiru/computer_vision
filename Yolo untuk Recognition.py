#yolo
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
import urllib.request

# Download YOLOv3 configuration and weights files
yolo_cfg_url = 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg'
yolo_weights_url = 'https://pjreddie.com/media/files/yolov3.weights'
coco_names_url = 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'

# Fungsi untuk mendownload file
def download_file(url, file_name):
    urllib.request.urlretrieve(url, file_name)

# Download files
download_file(yolo_cfg_url, 'yolov3.cfg')
download_file(yolo_weights_url, 'yolov3.weights')
download_file(coco_names_url, 'coco.names')

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Function to perform object detection
def detect_objects(image_path):
    # Load image
    img = cv2.imread(image_path)
    height, width, channels = img.shape

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists for detected bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Process YOLO outputs
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Flag to check if cat is detected
    cat_detected = False

    # Draw bounding boxes on the image
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == "cat":  # Only display bounding boxes for cats
                cat_detected = True
                color = (0, 255, 0)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

    # Print detection result
    if cat_detected:
        print(f"{image_path}: Cat detected")
    else:
        print(f"{image_path}: No cat detected")

# Upload and detect objects in images
uploaded = files.upload()

for fn in uploaded.keys():
    detect_objects(fn)
