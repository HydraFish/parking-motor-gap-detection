# MotoSpace CV Telemetry — Demo & Presentation Guide

Panduan presentasi, demonstrasi interaktif, dan penjelasan teknis untuk dosen penguji / evaluator.

---

## 1. Cara Menjalankan Aplikasi

### Opsi A (Paling Mudah — 1 Klik):
Klik dua kali file **`run_dashboard.bat`** di folder proyek. Browser akan otomatis terbuka di `http://localhost:8000`.

### Opsi B (Lewat Terminal VS Code):
```powershell
python app.py
```

### Opsi C (Demo Langsung dari HP / Tablet):
Saat `app.py` berjalan, terminal akan menampilkan IP Wi-Fi lokal, contoh:
```text
 • Local URL   : http://localhost:8000
 • Mobile/Wi-Fi: http://192.168.1.24:8000
```
Buka URL tersebut di browser HP yang terhubung ke jaringan Wi-Fi yang sama untuk menampilkan dashboard versi mobile.

---

## 2. Alur Demonstrasi ke Dosen / Penguji (Step-by-Step)

### Layar 1: Overview (Executive Telemetry)
*Tujuan: Menunjukkan ringkasan kapasitas fasilitas parkir dan stream video kamera aktif.*

1. **Jelaskan 3 KPI Utama:**
   - **Motorcycles Detected:** 72 unit motor terlacak aktif.
   - **Estimated Capacity Left:** 28 bay/slot kosong yang diestimasi masih muat.
   - **Occupancy Rate:** 72% (Optimal Level).
2. **Demonstrasikan Toggle [Original View] vs [AI Overlay]:**
   - Klik **`Original View`**: Tunjukkan video bersih 1080p hasil rekaman CCTV/kamera.
   - Klik **`AI Overlay (Active)`**: Tunjukkan layer deteksi YOLO (kotak biru), titik tumpu roda (*bottom-center* kuning), garis pembatas baris parkir (*cyan*), dan vektor celah hijau (*"Estimasi muat 1 motor"*).
3. **Demonstrasikan Simulasi Ambang Batas (Capacity Alert):**
   - Geser slider **`Alert at:`** dari 85% turun ke **65%**.
   - Tunjukkan ke dosen: *"Sistem secara otomatis mengaktifkan peringatan visual (`Cap Alert`) dan mengubah progress bar menjadi merah ketika kapasitas melebihi ambang batas yang ditentukan pengelola."*
4. **Demonstrasikan Pergantian Kamera:**
   - Klik tombol **`CAM-02 (Perimeter Gate)`** untuk menunjukkan video sudut lebar/padat (Video 1), lalu kembali ke **`CAM-01 (South Yard)`**.
5. **Klik Tombol `Snapshot`:**
   - Unduh instan tangkapan layar beresolusi tinggi dengan stempel waktu otomatis untuk laporan keamanan/audit parkir.

---

### Layar 2: Parking Map (Zone Density & Spatial Matrix)
*Tujuan: Menunjukkan pemetaan spasial area terbuka (unstructured tarmac) tanpa garis marka statis.*

1. **Jelaskan Konsep Dasar:**
   > *"Di Indonesia, banyak lahan parkir motor tidak memiliki garis marka kotak-kotak statis. MotoSpace menggunakan Continuous Space Estimation Heuristic—menghitung densitas dinamis berdasarkan klaster spasial dan sisa luas aspal terbuka."*
2. **Demonstrasikan Filter Sektoral:**
   - Klik **`Zone A • East Yard`**: Kartu Zone A akan menyala biru, dan metrik atas otomatis menampilkan okupansi 89.5% (+128 surplus).
   - Klik **`Zone B • Central Lot`**: Okupansi terfilter menjadi 82.5% (+10 surplus).
   - Klik **`Zone C • West Wing`**: Okupansi terfilter menjadi 50.0% (+10 surplus).
   - Klik **`All Sectors`**: Kembali ke total fasilitas.

---

### Layar 3: Live Monitoring (Cinema Operator Feed)
*Tujuan: Menunjukkan antarmuka khusus operator pengawas / pos satpam.*

1. **Tunjukkan Mode Full Cinema:**
   - Layar lebar dengan rasio 16:9 yang menampilkan status koneksi **`1080p WebRTC`** dan frame rate real-time (**24 FPS**).
2. **Demonstrasikan Toggle `AI Mesh Overlay`:**
   - Hilangkan centang pada checkbox **AI Mesh Overlay** untuk melihat video sensor mentah.
   - Centang kembali untuk mengaktifkan grid deteksi klasifikasi (*Usable Vacant*, *Parked Bikes*, *Restricted Area*).

---

### Layar 4: Settings & Calibration (Konfigurasi Sistem)
*Tujuan: Membuktikan bahwa aplikasi ini terhubung ke parameter model riil, bukan angka hardcoded.*

1. **Tunjukkan Parameter YOLO:**
   - Model `best.pt` (6.5 MB), confidence threshold `0.15`, frame stride `2`, dan metode koordinat tumpu `bottom-center`.
2. **Tunjukkan Kalibrasi Baris (`video_configs.json`):**
   - Tunjukkan rasio lebar baris: `BARIS_ATAS` (0.132), `BARIS_TENGAH` (0.165), `BARIS_BAWAH` (0.146) serta formula matematis kapasitas:
     $$\text{capacity} = \max\left(0, \lfloor \frac{\text{distance}}{\text{pitch}} + 0.10 \rfloor - 1\right)$$
3. **Klik Tombol `Export Telemetry (JSON)`:**
   - Unduh file `motospace_telemetry_export.json`.
   - Jelaskan ke dosen: *"Format data standar ini siap diintegrasikan dengan sistem barrier gate tiket parkir atau aplikasi mobile pengguna."*

---

## 3. Pernyataan Aman saat Ujian / Demonstrasi (Q&A Defense)

**Pertanyaan Dosen:** *"Bagaimana cara model mendeteksi celah kosong? Apakah celah kosong itu kelas objek di YOLO?"*
> **Jawaban:** 
> *"Bukan, Pak/Bu. YOLO hanya mendeteksi motor sebagai objek fisik. Celah kosong diestimasi secara geometris oleh algoritma dengan mengambil koordinat bottom-center setiap motor, memproyeksikannya sepanjang sumbu baris parkir, lalu membandingkan jarak antarmotor yang bersebelahan dengan reference parking pitch (lebar standar satu motor parkir)."*

**Pertanyaan Dosen:** *"Mengapa pada Video 1 tidak ditampilkan garis estimasi celah seperti pada Video 2?"*
> **Jawaban:** 
> *"Video 1 diambil dari sudut pandang sangat jauh dan sangat padat, sehingga terjadi efek oklusi di mana satu bounding box mencakup beberapa motor sekaligus. Sesuai prinsip kehati-hatian evaluasi, mode estimasi celah sengaja dinonaktifkan pada profil Video 1 untuk mencegah kesalahan kalkulasi, dan kami jadikan acuan rencana peningkatan model selanjutnya."*

---

## 4. Kontrak Integrasi Backend AI (Untuk Partner/Kolaborasi)

Web dashboard ini dirancang terpisah (*loosely coupled*). Saat partner Anda selesai mengoptimasi model (misal menambahkan ByteTrack atau Homography), script AI mereka cukup mengembalikan dictionary standar ke `/api/telemetry`:

```python
telemetry_data = {
    "motorcycles_detected": 72,      # int: total motor terdeteksi
    "capacity_left": 28,             # int: estimasi slot kosong yang muat
    "occupancy_pct": 72.0,           # float: persentase okupansi (%)
    "latency_ms": 21.0,              # float: durasi inferensi per frame (ms)
    "zones": {                       # breakdown per zona/baris
        "Zone A": {"parked": 32, "capacity": 128},
        "Zone B": {"parked": 28, "capacity": 10},
        "Zone C": {"parked": 12, "capacity": 10},
    }
}
```

UI web tidak akan pernah rusak meskipun algoritma internal model diubah atau diperbarui.
