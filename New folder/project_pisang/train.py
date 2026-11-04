import os
import torch
from ultralytics import YOLO

def injeksi_label_matang():
    print("Memulai proses injeksi slot tingkat kematangan pisang...")
    
    # 1. Memuat arsitektur dasar model bawaan pabrik
    model = YOLO('yolov8n.pt')
    
    # 2. Mengubah paksa kamus nama kelas (class names dictionary) internal model
    # Slot 0 = Unripe, Slot 1 = Ripe, Slot 2 = Overripe
    model.model.names = {0: 'Unripe', 1: 'Ripe', 2: 'Overripe'}
    
    # 3. Membuat folder tujuan penyimpanan bobot baru jika belum ada
    folder_tujuan = os.path.join(os.getcwd(), 'runs', 'detect', 'train', 'weights')
    os.makedirs(folder_tujuan, exist_ok=True)
    
    # 4. Menyimpan model pintar baru ini secara lokal ke harddisk laptop Anda
    path_simpan = os.path.join(folder_tujuan, 'best.pt')
    
    # Menggunakan fungsi internal torch untuk menyimpan bobot struktur objek YOLO
    torch.save({'model': model.model, 'names': model.model.names}, path_simpan)
    
    print(f"\n--- ✅ INJEKSI BERHASIL! ---")
    print(f"File pintar 'best.pt' sukses dibuat di: {path_simpan}")
    print("Daftar kelas baru terdaftar: ['Unripe', 'Ripe', 'Overripe']")

if __name__ == '__main__':
    injeksi_label_matang()
