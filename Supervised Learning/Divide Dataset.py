# Kode membagi dataset dari kelas-kelas 'black','grey', 'white', 'orange' menjadi 3 folder dataset 'train', 'validation', dan 'test':

import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
base_dir = '/content/drive/MyDrive/Dataset'
classes = ['black', 'grey', 'white', 'orange']

# Create directories
for split in ['train', 'validation', 'test']:
    for class_name in classes:
        os.makedirs(os.path.join(base_dir, split, class_name), exist_ok=True)

# Split data
for class_name in classes:
    class_dir = os.path.join(base_dir, class_name)
    images = os.listdir(class_dir)

    train_images, temp_images = train_test_split(images, test_size=0.4, random_state=42)  # 60% train, 40% temp
    val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)  # 20% val, 20% test

    for image in train_images:
        shutil.copy(os.path.join(class_dir, image), os.path.join(base_dir, 'train', class_name))
    for image in val_images:
        shutil.copy(os.path.join(class_dir, image), os.path.join(base_dir, 'validation', class_name))
    for image in test_images:
        shutil.copy(os.path.join(class_dir, image), os.path.join(base_dir, 'test', class_name))