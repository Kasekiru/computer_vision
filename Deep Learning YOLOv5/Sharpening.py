import cv2
import numpy as np
import os
import glob

# Path to your images folder
input_folder = "C:/Users/david/OneDrive/Desktop/david kuliah/Semester 6/vision comp/coding/dataset_cat_51"
output_folder = "C:/Users/david/OneDrive/Desktop/david kuliah/Semester 6/vision comp/coding/dataset_cat_51_Sharpening"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get a list of all image files in the input folder
image_paths = glob.glob(os.path.join(input_folder, "*.jpg"))

# Sharpening kernel
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

# Process each image
for image_path in image_paths:
    # Read the image
    img = cv2.imread(image_path)
    
    # Apply the sharpening filter
    sharpened = cv2.filter2D(img, -1, kernel)
    
    # Save the sharpened image to the output folder
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, sharpened)

print(f"Processed {len(image_paths)} images and saved to {output_folder}")
