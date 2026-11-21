"""Jalankan video kampus yang sudah memiliki kalibrasi ROI.

Taruh video di folder data, lalu jalankan:
    python run_demo.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "best.pt"
CONFIG = ROOT / "video_configs.json"
INPUT_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs" / "same_camera_gap_test"

VIDEOS = [
    ("VID20261116151825.mp4", "VID20261116151825_gap.mp4"),
    ("VID20261116151910.mp4", "VID20261116151910_gap.mp4"),
    ("VID20261116151945.mp4", "VID20261116151945_gap.mp4"),
    ("VID_20261116_152058.mp4", "VID_20261116_152058_gap.mp4"),
    ("VID_20261116_152136.mp4", "VID_20261116_152136_gap.mp4"),
    ("VID20261116152012.mp4", "VID20261116152012_gap.mp4"),
    ("VID20261116152047.mp4", "VID20261116152047_gap.mp4"),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL.exists():
        raise FileNotFoundError("best.pt tidak ditemukan di folder proyek.")

    found = 0
    for input_name, output_name in VIDEOS:
        input_path = INPUT_DIR / input_name
        if not input_path.exists():
            print(f"Lewati {input_name}: file belum ditemukan di {INPUT_DIR}.")
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
        print(f"Tidak ada video contoh. Salin video ke {INPUT_DIR}, lalu jalankan lagi.")
    else:
        print(f"Semua hasil tersedia di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
