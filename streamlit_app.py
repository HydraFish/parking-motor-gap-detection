"""Dashboard lokal untuk mengunggah dan memproses video parkir motor."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

import cv2
import imageio_ffmpeg
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best.pt"
CONFIG_PATH = ROOT / "video_configs.json"
PROCESSOR_PATH = ROOT / "process_parking_video.py"
RUNS_DIR = ROOT / "outputs" / "streamlit_runs"

PROFILE_OPTIONS = {
    "Otomatis": "auto",
    "Deteksi gap parkir": "campus_overhead_same_camera",
    "Deteksi motor saja": "generic_detection",
}


def safe_stem(filename: str) -> str:
    """Buat nama folder aman dari nama file unggahan."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(filename).stem).strip("_")
    return cleaned[:80] or "video"


def auto_device() -> str:
    """Pilih CUDA pertama bila tersedia, selain itu CPU."""
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def read_video_metadata(path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("Video tidak dapat dibuka oleh OpenCV.")
    metadata = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    capture.release()
    fps = float(metadata["fps"])
    metadata["duration"] = float(metadata["frames"]) / fps if fps > 0 else 0.0
    return metadata


def load_profiles() -> set[str]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return set(config.get("profiles", {}))


def run_inference(
    input_path: Path,
    output_path: Path,
    profile: str,
    conf: float | None,
    imgsz: int | None,
    stride: int | None,
    device: str,
) -> tuple[list[str], float]:
    """Jalankan processor yang sama dengan demo dan perbarui progres UI."""
    command = [
        sys.executable,
        "-u",
        str(PROCESSOR_PATH),
        "--model",
        str(MODEL_PATH),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--config",
        str(CONFIG_PATH),
        "--profile",
        profile,
        "--device",
        device,
        "--no-audio",
    ]
    if conf is not None:
        command.extend(["--conf", str(conf)])
    if imgsz is not None:
        command.extend(["--imgsz", str(imgsz)])
    if stride is not None:
        command.extend(["--inference-stride", str(stride)])

    progress = st.progress(0, text="Menyiapkan model dan video…")
    log_box = st.empty()
    lines: list[str] = []
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    assert process.stdout is not None
    frame_pattern = re.compile(r"Memproses\s+(\d+)/(\d+)\s+frame")
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        if not line:
            continue
        lines.append(line)
        match = frame_pattern.search(line)
        if match:
            current, total = map(int, match.groups())
            ratio = min(1.0, current / max(total, 1))
            progress.progress(ratio, text=f"Memproses frame {current}/{total}")
        log_box.code("\n".join(lines[-10:]), language="text")

    return_code = process.wait()
    elapsed = time.perf_counter() - started
    if return_code != 0:
        progress.empty()
        raise RuntimeError("\n".join(lines[-20:]) or "Proses inferensi gagal.")
    if not output_path.exists() or output_path.stat().st_size == 0:
        progress.empty()
        raise RuntimeError("Processor selesai tanpa menghasilkan video output.")

    progress.progress(1.0, text="Pemrosesan selesai")
    return lines, elapsed


def make_browser_compatible(source: Path, destination: Path) -> bool:
    """Konversi video OpenCV ke H.264 agar dapat diputar oleh browser."""
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i",
        str(source),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0 or not destination.exists():
        source.replace(destination)
        return False
    source.unlink(missing_ok=True)
    return True


st.set_page_config(
    page_title="MotoSpace Video Processor",
    page_icon="🏍️",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1180px; padding-top: 2rem;}
      [data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #263244;
        border-radius: 14px;
        padding: 14px 18px;
      }
      [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {color: #f8fafc;}
      .hero {
        padding: 22px 24px;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a, #172554);
        border: 1px solid #263b70;
        margin-bottom: 18px;
      }
      .hero h1 {margin: 0; color: white; font-size: 2rem;}
      .hero p {margin: 8px 0 0; color: #bfdbfe;}
    </style>
    <div class="hero">
      <h1>MotoSpace Video Processor</h1>
      <p>Unggah video, jalankan deteksi motor dan estimasi gap, lalu tinjau hasilnya langsung.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

missing = [
    path.name
    for path in (MODEL_PATH, CONFIG_PATH, PROCESSOR_PATH)
    if not path.exists()
]
if missing:
    st.error(f"File proyek belum lengkap: {', '.join(missing)}")
    st.stop()

try:
    available_profiles = load_profiles()
except (OSError, json.JSONDecodeError) as error:
    st.error(f"Konfigurasi tidak dapat dibaca: {error}")
    st.stop()

with st.sidebar:
    st.header("Konfigurasi")
    profile_label = st.selectbox("Mode pemrosesan", list(PROFILE_OPTIONS))
    profile = PROFILE_OPTIONS[profile_label]
    if profile != "auto" and profile not in available_profiles:
        st.error(f"Profil `{profile}` tidak ditemukan.")
        st.stop()

    detected_device = auto_device()
    device_label = st.selectbox(
        "Perangkat",
        ["Otomatis", "CPU"] + (["CUDA 0"] if detected_device == "0" else []),
    )
    device = detected_device if device_label == "Otomatis" else (
        "0" if device_label == "CUDA 0" else "cpu"
    )

    with st.expander("Pengaturan lanjutan"):
        use_profile_defaults = st.checkbox("Gunakan parameter dari profil", value=True)
        default_conf = 0.15 if profile == "generic_detection" else 0.20
        conf_value = st.slider(
            "Confidence", 0.05, 0.90, default_conf, 0.05,
            disabled=use_profile_defaults,
        )
        imgsz_value = st.select_slider(
            "Ukuran inferensi", [640, 768, 960, 1280], value=960,
            disabled=use_profile_defaults,
        )
        stride_value = st.slider(
            "Frame stride", 1, 12, 5,
            disabled=use_profile_defaults,
        )
        conf = None if use_profile_defaults else conf_value
        imgsz = None if use_profile_defaults else imgsz_value
        stride = None if use_profile_defaults else stride_value

    st.caption(f"Model: `{MODEL_PATH.name}`")
    st.caption(f"Device aktif: `{device}`")
    st.caption("Dashboard ini hanya menjalankan inference, tidak melakukan training.")

uploaded = st.file_uploader(
    "Pilih video",
    type=["mp4", "mov", "avi"],
    help="Video asli tidak diubah. Dashboard menyimpan salinan kerja di folder outputs.",
)

if uploaded is None:
    st.subheader("Cara memakai")
    st.write("1. Pilih mode di sidebar.  2. Unggah video.  3. Klik **Proses video**.  4. Putar atau unduh hasil.")
    st.stop()

left, right = st.columns([1.6, 1])
with left:
    st.subheader("Video input")
    st.video(uploaded)
with right:
    st.subheader("Ringkasan")
    st.write(f"**Nama:** {uploaded.name}")
    st.write(f"**Ukuran:** {uploaded.size / (1024 * 1024):.1f} MB")
    st.write(f"**Mode:** {profile_label}")
    st.write(f"**Output:** tanpa audio")

process_clicked = st.button("▶ Proses video", type="primary", use_container_width=True)

if process_clicked:
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    run_dir = RUNS_DIR / f"{safe_stem(uploaded.name)}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    input_suffix = Path(uploaded.name).suffix.lower() or ".mp4"
    input_path = run_dir / f"{safe_stem(uploaded.name)}{input_suffix}"
    raw_output_path = run_dir / f"{safe_stem(uploaded.name)}_processing.mp4"
    output_path = run_dir / f"{safe_stem(uploaded.name)}_detected.mp4"
    input_path.write_bytes(uploaded.getbuffer())

    try:
        metadata = read_video_metadata(input_path)
        logs, elapsed = run_inference(
            input_path,
            raw_output_path,
            profile,
            conf,
            imgsz,
            stride,
            device,
        )
        with st.spinner("Menyiapkan MP4 agar kompatibel dengan browser…"):
            browser_compatible = make_browser_compatible(raw_output_path, output_path)
    except Exception as error:
        st.error(f"Pemrosesan gagal: {error}")
    else:
        st.session_state["last_result"] = {
            "path": str(output_path),
            "input_name": uploaded.name,
            "profile": profile,
            "device": device,
            "elapsed": elapsed,
            "metadata": metadata,
            "logs": logs,
            "browser_compatible": browser_compatible,
        }
        st.success("Video berhasil diproses.")

result = st.session_state.get("last_result")
if result:
    output_path = Path(result["path"])
    if output_path.exists():
        st.divider()
        st.subheader("Hasil pemrosesan")
        metadata = result["metadata"]
        metric_cols = st.columns(4)
        metric_cols[0].metric("Resolusi", f"{metadata['width']} × {metadata['height']}")
        metric_cols[1].metric("FPS input", f"{metadata['fps']:.2f}")
        metric_cols[2].metric("Durasi video", f"{metadata['duration']:.1f} dtk")
        metric_cols[3].metric("Waktu proses", f"{result['elapsed']:.1f} dtk")

        st.video(str(output_path), format="video/mp4")
        if not result.get("browser_compatible", False):
            st.warning("Konversi H.264 tidak tersedia; gunakan tombol unduh jika preview tidak dapat diputar.")
        st.caption(f"Tersimpan di: `{output_path.relative_to(ROOT)}`")
        with output_path.open("rb") as video_file:
            st.download_button(
                "⬇ Unduh video hasil",
                data=video_file,
                file_name=output_path.name,
                mime="video/mp4",
                use_container_width=True,
            )
