import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# 1. Konfigurasi halaman dan Judul Aplikasi Capstone
st.set_page_config(page_title="Deteksi Kematangan Pisang", layout="centered")
st.title("🍌 Sistem Deteksi Kematangan Pisang - YOLOv8")
st.write("Aplikasi Capstone Project untuk sortasi tingkat kematangan buah pisang secara real-time.")

# 2. Memuat model pintar resmi YOLOv8 bawaan pabrik yang stabil
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

def proses_dan_gambar_kematangan(frame):
    # Menambahkan filter presisi conf dan iou agar objek pancingan/tangan/kotak ganda tidak ikut terdeteksi
    hasil = model(frame, verbose=False, conf=0.50, iou=0.3)
    boxes = hasil[0].boxes
    
    # Ambil data matriks gambar asli untuk digambar manual bawaan OpenCV
    img_out = frame.copy()
    
    for box in boxes:
        # Cek apakah objek yang dideteksi oleh YOLOv8 adalah pisang (indeks kelas 46)
        cls_id = int(box.cls[0])
        if cls_id == 46:
            # Ambil koordinat kotak pembatas (bounding box)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # --- 💡 FITUR SMART SIMULASI KEMATANGAN BERDASARKAN PROPORSI WARNA (RGB & HSV) ---
            # Crop area pisang untuk dianalisis warnanya
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                # Mengubah format ke HSV agar lebih kebal terhadap perubahan intensitas cahaya/bayangan
                               # --- KALIBRASI SUPER AKURAT PENGAMATAN WARNA KULIT PISANG ---
                # Mengubah format ke HSV untuk mengunci warna hijau murni
                hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                mean_hsv = cv2.mean(hsv_crop)
                h_val = mean_hsv[0] # Hue (Jenis Warna)
                s_val = mean_hsv[1] # Saturation (Kepekatan Warna)
                
                # Hitung rata-rata BGR asli
                mean_bgr = cv2.mean(crop)
                g_val, r_val = mean_bgr[1], mean_bgr[2]
                
                # Logika filter baru: Mengunci rentang warna hijau (mentah) secara ketat
                if (h_val >= 33 and h_val <= 85) or (g_val > r_val + 1):
                    label_kematangan = "Unripe (Mentah)"
                    warna_kotak = (0, 255, 0) # Hijau murni
                elif r_val > 140 and g_val > 120 and h_val < 33:
                    label_kematangan = "Ripe (Matang)"
                    warna_kotak = (0, 255, 255) # Kuning cerah
                else:
                    label_kematangan = "Overripe (Terlalu Matang)"
                    warna_kotak = (0, 0, 255) # Merah
            else:
                # Kondisi fallback jika ukuran gambar tidak valid
                label_kematangan = "Unripe (Mentah)"
                warna_kotak = (0, 255, 0)

            
            # 3. Gambar Kotak Pembatas & Teks Kelas Kematangan Baru ke Layar
            cv2.rectangle(img_out, (x1, y1), (x2, y2), warna_kotak, 3)
            teks = f"{label_kematangan} {conf:.2f}"
            cv2.putText(img_out, teks, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, warna_kotak, 2)
            
    return img_out

# 4. Fitur Navigasi Input Aplikasi
pilihan_input = st.radio("Pilih Metode Deteksi Kamera:", ("Unggah Foto Gambar", "Gunakan Kamera Live Webcam"))

# --- OPSI 1: DETEKSI VIA UPLOAD FOTO ---
if pilihan_input == "Unggah Foto Gambar":
    berkas_gambar = st.file_uploader("Pilih foto buah pisang...", type=["jpg", "jpeg", "png"])
    
    if berkas_gambar is not None:
        gambar_input = Image.open(berkas_gambar)
        st.image(gambar_input, caption="Foto Pisang yang Diunggah", use_container_width=True)
        
        if st.button("Mulai Analisis Deteksi"):
            with st.spinner("AI sedang memproses fitur fisik visual..."):
                array_gambar = np.array(gambar_input)
                # Konversi RGB ke BGR karena OpenCV membaca format BGR
                array_gambar_bgr = cv2.cvtColor(array_gambar, cv2.COLOR_RGB2BGR)
                
                # Proses deteksi kematangan pisang (Menggunakan filter presisi terpusat)
                hasil_visual = proses_dan_gambar_kematangan(array_gambar_bgr)
                
                # Kembalikan ke RGB untuk ditampilkan dengan benar di Streamlit
                hasil_visual_rgb = cv2.cvtColor(hasil_visual, cv2.COLOR_BGR2RGB)
                st.image(hasil_visual_rgb, caption="Hasil Klasifikasi Tingkat Kematangan Capstone", use_container_width=True)

# --- OPSI 2: DETEKSI VIA REAL-TIME WEBCAM ---
elif pilihan_input == "Gunakan Kamera Live Webcam":
    st.info("Klik tombol di bawah ini untuk membuka umpan kamera real-time.")
    kamera_aktif = st.checkbox("Aktifkan Webcam Laptop")
    bingkai_kamera = st.empty()
    
    if kamera_aktif:
        umpan_video = cv2.VideoCapture(0)
        
        while kamera_aktif:
            sukses, frame = umpan_video.read()
            if not sukses:
                st.warning("Gagal mengakses komponen kamera laptop.")
                break
            
            # Proses deteksi kematangan pisang per frame
            frame_hasil = proses_dan_gambar_kematangan(frame)
            
            frame_rgb = cv2.cvtColor(frame_hasil, cv2.COLOR_BGR2RGB)
            bingkai_kamera.image(frame_rgb, channels="RGB", use_container_width=True)
            
        umpan_video.release()
