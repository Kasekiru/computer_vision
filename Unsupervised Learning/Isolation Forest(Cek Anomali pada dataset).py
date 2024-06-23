#IsolationForest
# Step 2: Import necessary libraries
import os
import numpy as np
from PIL import Image
from sklearn.ensemble import IsolationForest

# Define the folder path
folder_path = '/content/drive/My Drive/Dataset'

# Step 3: Load and preprocess images
def load_images(folder_path):
    images = []
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        with Image.open(img_path) as img:
            img = img.resize((64, 64))  # Resize image to 64x64
            img_array = np.array(img).flatten()  # Flatten the image
            images.append((img_file, img_array))
    return images

images = load_images(folder_path)
image_files, image_data = zip(*images)

# Step 4: Apply Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
image_data_array = np.array(image_data)
iso_forest.fit(image_data_array)

# Predict the clusters
predictions = iso_forest.fit_predict(image_data_array)

# Step 5: Compare specific image (cat.jpg)
# Find a file that starts with "cat"
cat_image_file = next((f for f in image_files if f.startswith('cat')), None)

if cat_image_file:
    cat_image_path = os.path.join(folder_path, cat_image_file)
    with Image.open(cat_image_path) as img:
        img = img.resize((64, 64))
        cat_img_array = np.array(img).flatten()

    cat_cluster = iso_forest.predict([cat_img_array])

    # Print out the clusters
    for img_file, cluster in zip(image_files, predictions):
        print(f'Image: {img_file}, Cluster: {cluster}')

    print(f'{cat_image_file} Cluster: {cat_cluster[0]}')
else:
    print('No file starting with "cat" found.')
