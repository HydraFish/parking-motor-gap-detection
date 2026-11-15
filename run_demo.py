"""Jalankan kedua video contoh dengan satu perintah.

Taruh kedua video di folder yang sama dengan file ini, lalu jalankan:
    python run_demo.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "best.pt"
CONFIG = ROOT / "video_configs.json"
OUTPUT_DIR = ROOT / "outputs"

VIDEOS = [
    ("VID20261110141233.mp4", "hasil_video1_deteksi.mp4"),
    ("VID20261110141303.mp4", "hasil_video2_gap.mp4"),
]


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    if not MODEL.exists():
        raise FileNotFoundError("best.pt tidak ditemukan di folder proyek.")

    found = 0
    for input_name, output_name in VIDEOS:
        input_path = ROOT / input_name
        if not input_path.exists():
            print(f"Lewati {input_name}: file belum diletakkan di folder proyek.")
            continue
        found += 1
        command = [
            sys.executable,
            str(ROOT / "process_parking_video.py"),
            "--model",
            str(MODEL),
            "--input",
            str(input_path),
            "--output",
            str(OUTPUT_DIR / output_name),
            "--config",
            str(CONFIG),
            "--profile",
            "auto",
        ]
        subprocess.run(command, check=True)

    if found == 0:
        print("Tidak ada video contoh. Salin video ke folder proyek, lalu jalankan lagi.")
    else:
        print(f"Semua hasil tersedia di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
