# import os
# import zipfile
# import shutil

# # URL file zip yang akan diunduh
# url = 'https://github.com/Kasekiru/computer_vision/raw/master/Dataset/Cat.zip'

# # Nama file yang akan disimpan
# zip_file_path = '/content/Cat.zip'
# extract_folder = '/content/Cat'  # Folder sementara untuk ekstraksi

# # Google Drive folder tujuan
# google_drive_folder = '/content/drive/My Drive/Dataset'  # Sesuaikan path ini sesuai dengan lokasi di Google Drive Anda

# # Membuat folder ekstraksi jika belum ada
# os.makedirs(extract_folder, exist_ok=True)
# os.makedirs(google_drive_folder, exist_ok=True)

# # Mengunduh file zip menggunakan wget
# !wget -O $zip_file_path $url

# print(f'{zip_file_path} berhasil diunduh')

# # Mengekstrak file zip
# with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#     zip_ref.extractall(extract_folder)

# print(f'File berhasil diekstrak ke {extract_folder}')

# # Memindahkan file yang diekstrak ke Google Drive
# for filename in os.listdir(extract_folder):
#     shutil.move(os.path.join(extract_folder, filename), google_drive_folder)

# print(f'File berhasil dipindahkan ke {google_drive_folder}')

import os
import zipfile
import shutil

# URL file zip yang akan diunduh
url = 'https://github.com/Kasekiru/computer_vision/raw/master/Dataset/Cat.zip'

# Nama file yang akan disimpan
zip_file_path = '/content/Cat.zip'
extract_folder = '/content/Cat'  # Folder sementara untuk ekstraksi

# Google Drive folder tujuan
google_drive_folder = '/content/drive/My Drive/Dataset'  # Sesuaikan path ini sesuai dengan lokasi di Google Drive Anda

# Membuat folder ekstraksi jika belum ada
os.makedirs(extract_folder, exist_ok=True)
os.makedirs(google_drive_folder, exist_ok=True)

# Mengunduh file zip menggunakan wget
!wget -O $zip_file_path $url

print(f'{zip_file_path} berhasil diunduh')

# Mengekstrak file zip
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

print(f'File berhasil diekstrak ke {extract_folder}')

# Memindahkan file yang diekstrak ke Google Drive dan menghapus folder sementara
for root, dirs, files in os.walk(extract_folder):
    for filename in files:
        shutil.move(os.path.join(root, filename), google_drive_folder)

print(f'File berhasil dipindahkan ke {google_drive_folder}')

# Menghapus file zip dan folder sementara
os.remove(zip_file_path)
shutil.rmtree(extract_folder)

print(f'File zip dan folder sementara berhasil dihapus')
