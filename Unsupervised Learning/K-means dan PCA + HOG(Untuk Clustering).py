#K-means, PCA, HOG
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from skimage.feature import hog
from skimage import exposure
from google.colab.patches import cv2_imshow

# Path ke folder dataset di Google Drive
dataset_folder = '/content/drive/My Drive/Dataset'

# Fungsi untuk memuat dan memproses gambar
def load_and_preprocess_image(image_path, size=(128, 128)):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)
    return image

# Fungsi untuk mengekstrak fitur HOG dari gambar
def extract_features_hog(image):
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return fd

# Memuat dan memproses semua gambar dalam dataset
images = []
features = []
for filename in os.listdir(dataset_folder):
    image_path = os.path.join(dataset_folder, filename)
    image = load_and_preprocess_image(image_path)
    if image is not None:
        images.append(image)
        feature = extract_features_hog(image)
        features.append(feature)

features = np.array(features)

# Menggunakan PCA untuk mengurangi dimensi fitur sebelum K-Means
pca = PCA(n_components=50)  # Ubah sesuai kebutuhan
features_pca = pca.fit_transform(features)

# Menerapkan K-Means clustering pada dataset gambar
kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
clusters = kmeans.fit_predict(features_pca)

# Menyiapkan array untuk menyimpan indeks gambar untuk setiap cluster
cluster_indices = [[] for _ in range(kmeans.n_clusters)]
for i, cluster_label in enumerate(clusters):
    cluster_indices[cluster_label].append(i)

# Menampilkan beberapa contoh gambar dari setiap cluster
num_examples = 15  # Jumlah contoh gambar yang ingin ditampilkan dari setiap cluster
fig, axs = plt.subplots(kmeans.n_clusters, num_examples, figsize=(15, 10))

for cluster_label in range(kmeans.n_clusters):
    for i in range(num_examples):
        if i < len(cluster_indices[cluster_label]):
            idx = cluster_indices[cluster_label][i]
            ax = axs[cluster_label, i]
            ax.imshow(images[idx])
            ax.axis('off')
            ax.set_title(f'Cluster {cluster_label + 1}')
        else:
            ax = axs[cluster_label, i]
            ax.axis('off')  # Menyembunyikan subplot jika tidak ada gambar yang tersedia

plt.tight_layout()
plt.show()

# Fungsi untuk mengelompokkan gambar baru
def classify_new_image(image_path):
    image = load_and_preprocess_image(image_path)
    if image is None:
        print("Gambar tidak valid")
        return -1  # Mengembalikan nilai yang menunjukkan gambar tidak valid
    feature = extract_features_hog(image).reshape(1, -1)
    feature_pca = pca.transform(feature)
    cluster = kmeans.predict(feature_pca)[0]
    return cluster

# Meminta pengguna untuk mengunggah gambar baru
uploaded_image_path = '/content/cat.jpg'  # Ganti dengan path gambar yang diunggah
cluster = classify_new_image(uploaded_image_path)

if cluster != -1:
    image = cv2.imread(uploaded_image_path)
    cv2_imshow(image)
    print(f'Gambar tersebut termasuk dalam cluster: {cluster+1}')