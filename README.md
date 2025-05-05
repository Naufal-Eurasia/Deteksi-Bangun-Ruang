# Deteksi-Bangun-Ruang
Deteksi Bangun Ruang Lingkaran, Segitiga,Persegi, Trapezium dan Persegi panjang


```python
# --- 1. Instalasi ---
!pip install opencv-python-headless matplotlib

# --- 2. Import Library ---
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# --- 3. Upload Gambar (Hanya 5 gambar yang diizinkan) ---
print("Upload hingga 5 gambar bentuk geometris:")
uploaded = files.upload()

# Batasi hanya 5 gambar
if len(uploaded) > 5:
    print("Hanya 5 gambar yang dapat diunggah.")
    uploaded = dict(list(uploaded.items())[:5])

filenames = list(uploaded.keys())

# --- 4. Fungsi Deteksi Bentuk ---
def detect_shape(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    vertices = len(approx)
    shape = "Tidak dikenali"

    if vertices == 3:
        shape = "Segitiga"
    elif vertices == 4:
        # Cek apakah ini persegi atau persegi panjang
        if is_square(approx):
            shape = "Persegi"
        else:
            shape = "Persegi Panjang"  # Jika bukan persegi, maka persegi panjang
            if is_trapezium(approx):
                shape = "Trapesium"  # Trapesium terdeteksi jika ada sepasang sisi paralel
    elif vertices > 4:
        area = cv2.contourArea(contour)
        circularity = 4 * np.pi * (area / (peri * peri))
        if circularity > 0.75:
            shape = "Lingkaran"
        else:
            shape = "Bentuk Lain"  # Jika lebih dari 4 sisi dan tidak bisa dikategorikan

    return shape

# --- 5. Fungsi untuk Mengecek Apakah Sebuah Bentuk adalah Persegi ---
def is_square(approx):
    # Hitung panjang sisi untuk setiap sisi
    side_lengths = [np.linalg.norm(approx[i][0] - approx[(i + 1) % 4][0]) for i in range(4)]
    # Jika semua sisi hampir sama panjang, maka itu persegi
    if abs(side_lengths[0] - side_lengths[1]) < 10 and abs(side_lengths[1] - side_lengths[2]) < 10 and abs(side_lengths[2] - side_lengths[3]) < 10:
        return True
    return False

# --- 6. Fungsi untuk Mengecek Apakah Sebuah Bentuk adalah Trapesium ---
def is_trapezium(approx):
    # Asumsi: Trapesium memiliki 2 sisi yang sejajar dan 2 sisi yang tidak sejajar
    # Menghitung panjang sisi
    side_lengths = [np.linalg.norm(approx[i][0] - approx[(i + 1) % 4][0]) for i in range(4)]
    
    # Menghitung rasio panjang sisi sejajar
    top_bottom_ratio = abs(side_lengths[0] - side_lengths[2])
    
    # Jika perbedaan panjang sisi atas dan bawah cukup besar, kemungkinan besar itu adalah trapesium
    if top_bottom_ratio > 10:  # Saring berdasarkan perbedaan panjang sisi atas dan bawah
        return True
    return False

# --- 7. Proses Deteksi untuk Setiap Gambar ---
for filename in filenames:
    print(f"\n\n===== Memproses Gambar: {filename} =====")

    # --- Preprocessing ---
    img = cv2.imread(filename)
    if img is None:
        print("Gagal membaca gambar.")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # --- Temukan Kontur ---
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = img.copy()
    detected_shapes = []

    for cnt in contours:
        if cv2.contourArea(cnt) > 300:
            shape = detect_shape(cnt)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
                cv2.putText(output, shape, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                detected_shapes.append(shape)

    # --- Tampilkan Gambar Output ---
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title(f"Hasil Deteksi Bentuk: {filename}")
    plt.axis('off')
    plt.show()

    # --- Penjelasan Hasil Deteksi ---
    print("=== Penjelasan Bentuk yang Terdeteksi ===")
    if not detected_shapes:
        print("Tidak ada bentuk yang dikenali.")
    else:
        for shape in set(detected_shapes):
            count = detected_shapes.count(shape)
            print(f"- {shape} terdeteksi sebanyak {count} buah.")
            if shape == "Persegi":
                print("  > Semua sisi sama panjang dan sudutnya 90 derajat.")
            elif shape == "Persegi Panjang":
                print("  > Panjang ≠ lebar tetapi sudutnya siku-siku.")
            elif shape == "Lingkaran":
                print("  > Tidak memiliki sudut dan berbentuk melingkar.")
            elif shape == "Trapesium":
                print("  > Memiliki satu pasang sisi sejajar.")
            elif shape == "Segitiga":
                print("  > Memiliki 3 sisi dan 3 sudut.")

```
Hasil output :
Persegi Panjang
![image](https://github.com/user-attachments/assets/ba04df2a-4f2f-4e70-a92a-912ee900a1fd)
Lingkaran
![image](https://github.com/user-attachments/assets/0902c2f0-0bda-4dc5-9268-9a47dc2736d7)
Persegi
![image](https://github.com/user-attachments/assets/adf22aee-93ba-4bc1-ab5c-735aff0f56b2)
Trapesium
![image](https://github.com/user-attachments/assets/40ae3f94-2fcb-4664-a351-1ddb00285924)
Segitiga
![image](https://github.com/user-attachments/assets/b6b266d6-69b5-4d35-b918-24628f3c91ac)







