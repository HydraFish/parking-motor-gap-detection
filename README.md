# Prototype Deteksi Motor dan Estimasi Gap

Proyek ini memakai `best.pt` untuk mendeteksi motor pada video. Bounding box dibuat oleh YOLO, sedangkan **gap tidak dideteksi oleh YOLO**. Gap dihitung dari jarak `bottom-center` motor yang bersebelahan dalam baris parkir yang sama.

## Isi proyek

- `best.pt` — model hasil training.
- `process_parking_video.py` — program utama untuk satu video.
- `run_demo.py` — memproses dua video contoh sekaligus.
- `video_configs.json` — ROI/baris parkir, pitch, confidence, dan mode tiap video.
- `test_gap_calculation.py` — pengujian geometri gap tanpa memproses video.
- `test_temporal_smoothing.py` — pengujian stabilitas gap sepanjang waktu.
- `requirements.txt` — library Python.

## Cara paling mudah di VS Code (Windows)

1. Ekstrak ZIP proyek.
2. Salin dua video ke folder yang sama dengan `run_demo.py`:
   - `VID20261110141233.mp4`
   - `VID20261110141303.mp4`
3. Buka folder proyek di VS Code.
4. Buka menu **Terminal → New Terminal**.
5. Jalankan:

   ```powershell
   python -m pip install -r requirements.txt
   python run_demo.py
   ```

   Alternatif: klik dua kali `run_demo.bat`.

6. Hasil muncul di folder `outputs`:
   - `hasil_video1_deteksi.mp4`
   - `hasil_video2_gap.mp4`

Jika perintah `python` tidak dikenali, instal Python 3.10 atau 3.11 dan centang **Add Python to PATH**.

## Memproses video lain

```powershell
python process_parking_video.py `
  --model best.pt `
  --input video_baru.mp4 `
  --output hasil_video_baru.mp4 `
  --config video_configs.json `
  --profile generic_detection
```

`generic_detection` hanya menggambar bounding box. Untuk estimasi gap pada kamera baru, buat profil baru di `video_configs.json` dan kalibrasi:

- polygon baris/ROI;
- arah baris (`axis`);
- `boundary_start` dan `boundary_end` dalam koordinat normalisasi 0–1;
- `boundary_visibility.start` dan `boundary_visibility.end` untuk menandai apakah
  ujung fisik baris benar-benar terlihat;
- `reference_pitch` sebagai rasio terhadap lebar frame;
- `edge_margin_pitch` (default `0.5`);
- confidence dan ukuran inferensi.

Koordinat polygon dan boundary memakai rasio 0–1 sehingga tidak bergantung pada
resolusi. Contoh titik `[0.5, 0.25]` berarti x = 50% lebar dan y = 25% tinggi
gambar.

## Cara kerja output

- Kotak biru: prediksi motor dari YOLO.
- Titik kuning: `bottom-center` motor, sebagai pendekatan posisi di lantai.
- Garis/poligon biru muda: baris parkir/ROI.
- Garis hijau: gap internal di antara dua motor.
- Garis oranye: gap tepi dari boundary ke motor pertama/terakhir.
- Garis ungu: baris kosong dari `boundary_start` ke `boundary_end`.
- Label gap menampilkan tipe (`internal`, `edge_start`, `edge_end`, atau
  `empty_row`) dan estimasi kapasitas.
- Panel atas: jumlah deteksi, kandidat gap, dan estimasi kapasitas tambahan.

## P3: Boundary / Edge Gaps dan Empty Rows

Setiap motor direpresentasikan oleh titik `bottom-center` bounding box. Titik
tersebut dan kedua boundary diproyeksikan sepanjang `axis` baris. Pitch selalu
diambil dari `reference_pitch` milik row yang sedang dihitung.

Rumus gap internal tetap:

```text
capacity = max(0, floor(distance / reference_pitch + tolerance) - 1)
```

Karena titik motor merepresentasikan pusat kendaraan, edge gap mengurangi ruang
setara `edge_margin_pitch` sebelum menghitung slot:

```text
capacity = max(0, floor(distance / reference_pitch
                        + tolerance - edge_margin_pitch))
```

Dengan nilai default `edge_margin_pitch = 0.5`, boundary dianggap sebagai tepi
fisik baris, bukan pusat slot. Jika tidak ada motor dalam row, kapasitas dihitung
dari seluruh panjang proyeksi `boundary_start` sampai `boundary_end`:

```text
capacity = max(0, floor(row_length / reference_pitch + tolerance))
```

Tepi frame kamera tidak otomatis merupakan batas fisik parkiran. `edge_start`
hanya dihitung jika `boundary_visibility.start` bernilai `true`, dan `edge_end`
hanya dihitung jika `boundary_visibility.end` bernilai `true`. Empty row hanya
dilaporkan ketika kedua batas fisik terlihat dan valid. Gap internal tetap dapat
dihitung karena kedua ujungnya berasal dari motor yang terlihat.

Pada profil `video2_gap_demo`, kedua visibility setiap row bernilai `false`
karena baris parkir masih berlanjut atau kendaraan terpotong pada tepi frame.
Koordinat boundary tetap membatasi anggota row, tetapi tidak diperlakukan sebagai
ujung fisik untuk edge gap atau empty row.

Hasil masih estimasi relatif dalam piksel, bukan ukuran meter.

Logika perhitungan dipisahkan dari fungsi visualisasi sehingga dapat diuji
tanpa membuka video. Jalankan pengujian sederhana dengan:

```powershell
python -m unittest -v test_gap_calculation.py
```

## P4: Temporal Hysteresis / Stability Smoothing

P3 menghitung lokasi dan kapasitas gap secara geometri pada satu frame. P4
menstabilkan hasil P3 sepanjang waktu sebelum gap dihitung dan ditampilkan.
P4 tidak melakukan tracking motor, tidak mengubah deteksi YOLO, dan bukan
training model baru.

Pada `video2_gap_demo`, `activation_seconds = 1.5` berarti gap harus terus
ditemukan selama 1,5 detik sebelum dikonfirmasi. `release_seconds = 0.75`
mempertahankan gap terkonfirmasi selama kehilangan sementara hingga 0,75 detik.
Pencocokan ant_frame memakai row, jenis gap, dan overlap interval sepanjang axis
row. Kapasitas stabil menggunakan median lima pengamatan terakhir.

Jika `temporal_smoothing.enabled` tidak ada atau bernilai `false`, hasil P3
langsung digunakan tanpa penundaan. Video 1 tetap memakai mode
`detection_only`, sehingga P4 tidak dijalankan.

```powershell
python test_temporal_smoothing.py
```

## Catatan dua video contoh

- **Video 2** cocok untuk demo penuh karena POV lebih dekat, baris mendatar, dan motor terdeteksi satu per satu.
- **Video 1** sangat jauh dan padat. Model sering membuat satu box untuk beberapa motor. Program memakai potongan gambar agar objek lebih besar, tetapi mode gap sengaja dinonaktifkan karena hasil geometri akan menyesatkan.

Untuk laporan/demonstrasi dosen, gunakan Video 2 sebagai hasil utama dan Video 1 sebagai contoh keterbatasan serta rencana peningkatan model/data.

## Pernyataan yang aman saat demo

> YOLO mendeteksi motor. Sistem kemudian mengambil bottom-center setiap bounding box, memasukkannya ke baris parkir, mengurutkannya sepanjang baris, lalu membandingkan jarak antarmotor dengan reference parking pitch. Karena belum ada ground truth kapasitas gap dan belum dikalibrasi dalam meter, hasil ini adalah qualitative proof of concept.
