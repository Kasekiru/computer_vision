import cv2
import os

def apply_gaussian_blur(image, kernel_size=(5, 5), sigma_x=0):
    return cv2.GaussianBlur(image, kernel_size, sigma_x)

# Tentukan folder input dan output
input_folder = 'D:/DLL/visi/dataset_cat_51'  # Ganti dengan jalur folder input
output_folder = 'D:/DLL/visi/dataset_cat_51_gaussian_blur'  # Ganti dengan jalur folder output

# Pastikan folder output ada
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterasi melalui file di folder input
for filename in os.listdir(input_folder):
    # Periksa apakah file adalah gambar
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Baca gambar
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Cek apakah gambar berhasil dimuat
        if image is None:
            print(f"Gagal membaca gambar {filename}!")
            continue

        # Terapkan Gaussian blur
        blurred_image = apply_gaussian_blur(image)

        # Simpan gambar hasil blur ke folder output
        output_path = os.path.join(output_folder, 'blurred_' + filename)
        cv2.imwrite(output_path, blurred_image)

        print(f"Gambar {filename} telah diproses dan disimpan sebagai {output_path}.")
