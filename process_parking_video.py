"""Proses video parkir motor dengan model Ultralytics YOLO.

Fitur utama:
- bounding box motor pada video;
- ROI/baris parkir yang dapat dikonfigurasi;
- estimasi celah dari jarak bottom-center antarmotor (bukan kelas YOLO);
- mode sliced inference untuk objek yang sangat kecil;
- hasil MP4 siap diputar, dengan audio asli jika ffmpeg tersedia.

Contoh:
    python process_parking_video.py --model best.pt --input video.mp4 \
        --output hasil.mp4 --config video_configs.json --profile auto
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0

    @property
    def bottom_center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, self.y2)


@dataclass(frozen=True)
class GapResult:
    row_id: str
    gap_type: str
    start: tuple[float, float]
    end: tuple[float, float]
    distance: float
    capacity: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deteksi motor dan estimasi celah parkir dari video."
    )
    parser.add_argument("--model", default="best.pt", help="Path model YOLO .pt")
    parser.add_argument("--input", required=True, help="Path video input")
    parser.add_argument("--output", required=True, help="Path video output .mp4")
    parser.add_argument(
        "--config", default="video_configs.json", help="Path file konfigurasi JSON"
    )
    parser.add_argument(
        "--profile",
        default="auto",
        help="Nama profil dalam config, atau 'auto' berdasarkan nama video",
    )
    parser.add_argument("--conf", type=float, default=None, help="Override confidence")
    parser.add_argument("--imgsz", type=int, default=None, help="Override YOLO image size")
    parser.add_argument(
        "--inference-stride",
        type=int,
        default=None,
        help="Jalankan YOLO setiap N frame (frame di antaranya memakai hasil terakhir)",
    )
    parser.add_argument(
        "--device", default="cpu", help="Contoh: cpu, 0 (GPU pertama), atau mps"
    )
    parser.add_argument(
        "--no-audio", action="store_true", help="Jangan salin audio video asli"
    )
    parser.add_argument(
        "--preview", action="store_true", help="Tampilkan jendela preview saat proses"
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_profile(config: dict[str, Any], requested: str, input_path: Path) -> tuple[str, dict[str, Any]]:
    profiles = config["profiles"]
    if requested != "auto":
        if requested not in profiles:
            raise KeyError(f"Profil '{requested}' tidak ada. Pilihan: {', '.join(profiles)}")
        return requested, profiles[requested]

    filename = input_path.name.lower()
    for name, profile in profiles.items():
        if any(token.lower() in filename for token in profile.get("match", [])):
            return name, profile

    default_name = config.get("default_profile", next(iter(profiles)))
    return default_name, profiles[default_name]


def yolo_to_detections(result: Any, offset_y: int = 0) -> list[Detection]:
    detections: list[Detection] = []
    if result.boxes is None:
        return detections
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy()
    for box, conf, class_id in zip(xyxy, confs, classes):
        x1, y1, x2, y2 = map(float, box)
        detections.append(
            Detection(x1, y1 + offset_y, x2, y2 + offset_y, float(conf), int(class_id))
        )
    return detections


def iou(a: Detection, b: Detection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    return intersection / max(area_a + area_b - intersection, 1e-9)


def nms(detections: Iterable[Detection], threshold: float = 0.45) -> list[Detection]:
    pending = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []
    while pending:
        current = pending.pop(0)
        kept.append(current)
        pending = [candidate for candidate in pending if iou(current, candidate) < threshold]
    return kept


def infer_frame(
    model: YOLO,
    frame: np.ndarray,
    profile: dict[str, Any],
    conf: float,
    imgsz: int,
    device: str,
) -> list[Detection]:
    sliced = profile.get("sliced_inference", {}).get("enabled", False)
    if not sliced:
        result = model.predict(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        return yolo_to_detections(result)

    # Untuk video sangat jauh: potong vertikal agar motor kecil menjadi lebih besar.
    height = frame.shape[0]
    tile_cfg = profile["sliced_inference"]
    tile_height = min(int(tile_cfg.get("tile_height", 500)), height)
    overlap = float(tile_cfg.get("overlap", 0.30))
    step = max(1, int(tile_height * (1.0 - overlap)))
    starts = list(range(0, max(1, height - tile_height + 1), step))
    last_start = height - tile_height
    if not starts or starts[-1] != last_start:
        starts.append(last_start)

    detections: list[Detection] = []
    for y in starts:
        crop = frame[y : y + tile_height, :]
        result = model.predict(crop, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        detections.extend(yolo_to_detections(result, offset_y=y))
    return nms(detections, float(tile_cfg.get("nms_iou", 0.45)))


def scaled_polygon(points: list[list[float]], width: int, height: int) -> np.ndarray:
    return np.array(
        [[round(x * width), round(y * height)] for x, y in points], dtype=np.int32
    )


def scaled_point(point: list[float], width: int, height: int) -> np.ndarray:
    return np.array([point[0] * width, point[1] * height], dtype=np.float64)


def point_in_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon.astype(np.float32), point, False) >= 0


def put_label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.52,
) -> None:
    x, y = origin
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1
    )
    cv2.rectangle(
        frame,
        (x, y - text_height - baseline - 5),
        (x + text_width + 7, y + 3),
        color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x + 3, y - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_detection(frame: np.ndarray, detection: Detection) -> None:
    blue = (255, 90, 20)
    x1, y1, x2, y2 = map(int, (detection.x1, detection.y1, detection.x2, detection.y2))
    cv2.rectangle(frame, (x1, y1), (x2, y2), blue, 2)
    put_label(frame, f"Motor {detection.confidence:.2f}", (x1, max(18, y1)), blue, 0.46)


def calculate_gaps(
    detections: list[Detection],
    profile: dict[str, Any],
    frame_width: int,
    frame_height: int,
) -> tuple[list[GapResult], dict[str, int]]:
    """Hitung gap tanpa menggambar agar geometri dapat diuji secara terpisah."""
    width = frame_width
    height = frame_height
    gaps: list[GapResult] = []
    row_counts: dict[str, int] = {}
    tolerance = float(profile.get("gap_tolerance", 0.10))

    for row in profile.get("rows", []):
        row_id = str(row["id"])
        polygon = scaled_polygon(row["polygon"], width, height)
        members = [d for d in detections if point_in_polygon(d.bottom_center, polygon)]
        row_counts[row_id] = len(members)

        axis = np.array(row.get("axis", [1.0, 0.0]), dtype=np.float64)
        axis /= max(np.linalg.norm(axis), 1e-9)
        boundary_start = scaled_point(row["boundary_start"], width, height)
        boundary_end = scaled_point(row["boundary_end"], width, height)
        if float(np.dot(boundary_end - boundary_start, axis)) < 0:
            axis = -axis

        members.sort(key=lambda d: float(np.dot(np.array(d.bottom_center), axis)))
        pitch_ratio = row.get("reference_pitch", row.get("reference_pitch_width_ratio"))
        if pitch_ratio is None:
            raise KeyError(f"Row '{row_id}' tidak memiliki reference_pitch")
        pitch_px = float(pitch_ratio) * width
        pitch_px = max(pitch_px, 1e-9)
        edge_margin = float(row.get("edge_margin_pitch", 0.5))

        if not members:
            distance = max(0.0, float(np.dot(boundary_end - boundary_start, axis)))
            capacity = max(0, math.floor(distance / pitch_px + tolerance))
            gaps.append(
                GapResult(
                    row_id,
                    "empty_row",
                    tuple(boundary_start),
                    tuple(boundary_end),
                    distance,
                    capacity,
                )
            )
            continue

        first = np.array(members[0].bottom_center, dtype=np.float64)
        start_distance = max(0.0, float(np.dot(first - boundary_start, axis)))
        start_capacity = max(
            0, math.floor(start_distance / pitch_px + tolerance - edge_margin)
        )
        gaps.append(
            GapResult(
                row_id,
                "edge_start",
                tuple(boundary_start),
                tuple(first),
                start_distance,
                start_capacity,
            )
        )

        for left, right in zip(members, members[1:]):
            p1 = np.array(left.bottom_center, dtype=np.float64)
            p2 = np.array(right.bottom_center, dtype=np.float64)
            distance = abs(float(np.dot(p2 - p1, axis)))
            capacity = max(0, math.floor(distance / pitch_px + tolerance) - 1)
            gaps.append(
                GapResult(
                    row_id,
                    "internal",
                    tuple(p1),
                    tuple(p2),
                    distance,
                    capacity,
                )
            )

        last = np.array(members[-1].bottom_center, dtype=np.float64)
        end_distance = max(0.0, float(np.dot(boundary_end - last, axis)))
        end_capacity = max(
            0, math.floor(end_distance / pitch_px + tolerance - edge_margin)
        )
        gaps.append(
            GapResult(
                row_id,
                "edge_end",
                tuple(last),
                tuple(boundary_end),
                end_distance,
                end_capacity,
            )
        )

    return gaps, row_counts


def draw_gap_results(
    frame: np.ndarray,
    gaps: list[GapResult],
    show_tight_gaps: bool = False,
) -> None:
    colors = {
        "internal": (35, 210, 70),
        "edge_start": (0, 165, 255),
        "edge_end": (0, 165, 255),
        "empty_row": (190, 70, 200),
    }
    label_colors = {
        "internal": (25, 155, 50),
        "edge_start": (0, 120, 220),
        "edge_end": (0, 120, 220),
        "empty_row": (145, 45, 160),
    }

    for gap in gaps:
        a = tuple(np.round(gap.start).astype(int))
        b = tuple(np.round(gap.end).astype(int))
        if gap.capacity <= 0:
            if show_tight_gaps and gap.gap_type == "internal":
                cv2.line(frame, a, b, (120, 120, 120), 2, cv2.LINE_AA)
            continue

        midpoint = tuple(
            np.round((np.array(gap.start) + np.array(gap.end)) / 2.0).astype(int)
        )
        color = colors[gap.gap_type]
        cv2.line(frame, a, b, color, 6, cv2.LINE_AA)
        wording = f"{gap.gap_type}: muat {gap.capacity} motor"
        put_label(
            frame,
            wording,
            (midpoint[0] - 65, midpoint[1] - 8),
            label_colors[gap.gap_type],
            0.42,
        )


def estimate_and_draw_gaps(
    frame: np.ndarray,
    detections: list[Detection],
    profile: dict[str, Any],
) -> tuple[int, int, dict[str, int]]:
    height, width = frame.shape[:2]
    gaps, row_counts = calculate_gaps(detections, profile, width, height)

    for row in profile.get("rows", []):
        polygon = scaled_polygon(row["polygon"], width, height)
        cv2.polylines(frame, [polygon], True, (255, 200, 0), 2)
        label_at = tuple(int(v) for v in polygon[0])
        put_label(
            frame,
            str(row["id"]),
            (label_at[0], max(20, label_at[1] + 18)),
            (255, 170, 0),
            0.48,
        )

        members = [d for d in detections if point_in_polygon(d.bottom_center, polygon)]
        for detection in members:
            point = tuple(np.round(detection.bottom_center).astype(int))
            cv2.circle(frame, point, 5, (0, 230, 255), -1, cv2.LINE_AA)

    draw_gap_results(frame, gaps, bool(profile.get("show_tight_gaps", False)))
    positive_gaps = [gap for gap in gaps if gap.capacity > 0]
    return len(positive_gaps), sum(gap.capacity for gap in positive_gaps), row_counts


def draw_dashboard(
    frame: np.ndarray,
    profile_name: str,
    profile: dict[str, Any],
    motor_count: int,
    candidates: int,
    capacity: int,
    frame_index: int,
    fps: float,
) -> None:
    overlay = frame.copy()
    panel_height = 128 if profile.get("mode") == "gap" else 150
    cv2.rectangle(overlay, (8, 8), (frame.shape[1] - 8, panel_height), (15, 18, 24), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    lines = [
        f"Prototype Parkir Motor | profil: {profile_name}",
        f"Motor terdeteksi: {motor_count} | waktu: {frame_index / max(fps, 1e-9):.1f} s",
    ]
    if profile.get("mode") == "gap":
        lines.append(f"Kandidat gap: {candidates} | estimasi kapasitas tambahan: {capacity}")
        lines.append("Gap dihitung dari posisi motor; bukan kelas yang dideteksi YOLO")
    else:
        lines.append("Mode deteksi saja: estimasi gap dinonaktifkan pada POV sangat jauh/padat")
        lines.append("Beberapa bounding box dapat mewakili beberapa motor; jumlah belum reliabel")

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (18, 34 + index * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )


def mux_audio(video_without_audio: Path, original: Path, final_output: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        shutil.move(str(video_without_audio), str(final_output))
        return

    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_without_audio),
        "-i",
        str(original),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "21",
        "-c:a",
        "aac",
        "-shortest",
        "-movflags",
        "+faststart",
        str(final_output),
    ]
    subprocess.run(command, check=True)
    video_without_audio.unlink(missing_ok=True)


def process_video(args: argparse.Namespace) -> None:
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    model_path = Path(args.model).resolve()
    config_path = Path(args.config).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    profile_name, profile = select_profile(config, args.profile, input_path)
    conf = float(args.conf if args.conf is not None else profile.get("conf", 0.15))
    imgsz = int(args.imgsz if args.imgsz is not None else profile.get("imgsz", 640))
    stride = int(
        args.inference_stride
        if args.inference_stride is not None
        else profile.get("inference_stride", 2)
    )

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Video tidak dapat dibuka: {input_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    model = YOLO(str(model_path))
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=output_path.parent) as tmp:
        temporary_video = Path(tmp.name)
    writer = cv2.VideoWriter(
        str(temporary_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("VideoWriter tidak dapat dibuat. Periksa instalasi OpenCV.")

    print(f"Profil       : {profile_name}")
    print(f"Mode         : {profile.get('mode', 'detection_only')}")
    print(f"Video        : {width}x{height}, {fps:.2f} FPS, {frame_count} frame")
    print(f"YOLO         : conf={conf}, imgsz={imgsz}, stride={stride}, device={args.device}")

    latest_detections: list[Detection] = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % max(1, stride) == 0 or not latest_detections:
                latest_detections = infer_frame(
                    model, frame, profile, conf=conf, imgsz=imgsz, device=args.device
                )

            annotated = frame.copy()
            for detection in latest_detections:
                draw_detection(annotated, detection)

            candidates = capacity = 0
            if profile.get("mode") == "gap":
                candidates, capacity, _ = estimate_and_draw_gaps(
                    annotated, latest_detections, profile
                )

            draw_dashboard(
                annotated,
                profile_name,
                profile,
                len(latest_detections),
                candidates,
                capacity,
                frame_index,
                fps,
            )
            writer.write(annotated)

            if args.preview:
                preview = cv2.resize(annotated, (width // 2, height // 2))
                cv2.imshow("Parking prototype - tekan q untuk berhenti", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
            if frame_index % 50 == 0 or frame_index == frame_count:
                print(f"Memproses {frame_index}/{frame_count} frame")
    finally:
        capture.release()
        writer.release()
        if args.preview:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                # opencv-python-headless memang tidak memiliki backend jendela.
                pass

    if args.no_audio:
        shutil.move(str(temporary_video), str(output_path))
    else:
        mux_audio(temporary_video, input_path, output_path)
    print(f"Selesai       : {output_path}")


if __name__ == "__main__":
    process_video(parse_args())
