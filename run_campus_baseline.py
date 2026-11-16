"""Baseline inference best.pt untuk media kampus tanpa ROI, gap, atau training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}

DETECTION_FIELDS = [
    "source_file",
    "media_type",
    "frame_index",
    "timestamp_seconds",
    "detection_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "box_width",
    "box_height",
]

SUMMARY_FIELDS = [
    "source_file",
    "media_type",
    "width",
    "height",
    "fps",
    "total_frames",
    "processed_frames",
    "duration_seconds",
    "total_detections",
    "average_detections_per_processed_frame",
    "minimum_detections",
    "maximum_detections",
    "average_confidence",
    "minimum_confidence",
    "maximum_confidence",
    "processing_seconds",
    "device",
    "status",
    "error_message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Uji best.pt pada foto/video kampus dalam mode detection_only. "
            "Script ini tidak melakukan training dan tidak memakai ROI/gap."
        )
    )
    parser.add_argument("--input", default="data/campus_raw", help="Folder media mentah")
    parser.add_argument(
        "--output", default="outputs/campus_baseline", help="Folder hasil baseline"
    )
    parser.add_argument("--model", default="best.pt", help="Path model YOLO .pt")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence minimum")
    parser.add_argument("--imgsz", type=int, default=960, help="Ukuran inferensi YOLO")
    parser.add_argument(
        "--video-stride",
        type=int,
        default=5,
        help="Inferensi hanya setiap N frame video",
    )
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, 0, cuda:0, dan seterusnya"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Proses ulang media yang output suksesnya sudah tersedia",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_media(input_root: Path) -> tuple[list[Path], list[Path]]:
    files = sorted(path for path in input_root.rglob("*") if path.is_file())
    images = [path for path in files if path.suffix.lower() in IMAGE_EXTENSIONS]
    videos = [path for path in files if path.suffix.lower() in VIDEO_EXTENSIONS]
    return images, videos


def resolve_device(requested: str) -> tuple[str | int, str]:
    normalized = requested.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return 0, f"cuda:0 ({torch.cuda.get_device_name(0)})"
        return "cpu", "cpu"
    if normalized.isdigit():
        index = int(normalized)
        name = (
            torch.cuda.get_device_name(index)
            if torch.cuda.is_available() and index < torch.cuda.device_count()
            else f"CUDA device {index}"
        )
        return index, f"cuda:{index} ({name})"
    return requested, requested


def relative_name(source: Path, input_root: Path) -> str:
    return source.relative_to(input_root).as_posix()


def predict_detections(
    model: YOLO,
    image: np.ndarray,
    conf: float,
    imgsz: int,
    inference_device: str | int,
) -> list[dict[str, float]]:
    result = model.predict(
        image,
        conf=conf,
        imgsz=imgsz,
        device=inference_device,
        verbose=False,
    )[0]
    if result.boxes is None:
        return []

    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    confidences = result.boxes.conf.detach().cpu().numpy()
    detections: list[dict[str, float]] = []
    for box, confidence in zip(xyxy, confidences):
        x1, y1, x2, y2 = map(float, box)
        detections.append(
            {
                "confidence": float(confidence),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return detections


def draw_detections(
    image: np.ndarray,
    detections: list[dict[str, float]],
    show_count: bool,
) -> np.ndarray:
    annotated = image.copy()
    height, width = annotated.shape[:2]
    thickness = max(2, round(min(width, height) / 500))
    font_scale = max(0.48, min(width, height) / 1400)
    blue = (255, 90, 20)

    for detection in detections:
        x1, y1, x2, y2 = (
            int(round(detection["x1"])),
            int(round(detection["y1"])),
            int(round(detection["x2"])),
            int(round(detection["y2"])),
        )
        cv2.rectangle(annotated, (x1, y1), (x2, y2), blue, thickness)
        label = f"Motor {detection['confidence']:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        label_y = max(text_height + baseline + 5, y1)
        cv2.rectangle(
            annotated,
            (x1, label_y - text_height - baseline - 5),
            (x1 + text_width + 7, label_y + 2),
            blue,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 3, label_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    if show_count:
        text = f"Motor terdeteksi: {len(detections)}"
        scale = max(0.58, min(width, height) / 1100)
        text_thickness = max(1, thickness)
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, text_thickness
        )
        cv2.rectangle(
            annotated,
            (10, 10),
            (20 + text_width, 22 + text_height + baseline),
            (15, 18, 24),
            -1,
        )
        cv2.putText(
            annotated,
            text,
            (15, 16 + text_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )
    return annotated


def detection_rows(
    source_file: str,
    media_type: str,
    frame_index: int,
    timestamp_seconds: float,
    detections: list[dict[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for detection_id, detection in enumerate(detections, start=1):
        rows.append(
            {
                "source_file": source_file,
                "media_type": media_type,
                "frame_index": frame_index,
                "timestamp_seconds": round(timestamp_seconds, 6),
                "detection_id": detection_id,
                "class_name": "Motor",
                "confidence": round(detection["confidence"], 6),
                "x1": round(detection["x1"], 3),
                "y1": round(detection["y1"], 3),
                "x2": round(detection["x2"], 3),
                "y2": round(detection["y2"], 3),
                "box_width": round(detection["x2"] - detection["x1"], 3),
                "box_height": round(detection["y2"] - detection["y1"], 3),
            }
        )
    return rows


def confidence_stats(confidences: list[float]) -> tuple[float | None, float | None, float | None]:
    if not confidences:
        return None, None, None
    return (
        sum(confidences) / len(confidences),
        min(confidences),
        max(confidences),
    )


def process_image(
    source: Path,
    input_root: Path,
    image_output_root: Path,
    model: YOLO,
    conf: float,
    imgsz: int,
    inference_device: str | int,
    device_label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    source_file = relative_name(source, input_root)
    image = cv2.imread(str(source))
    if image is None:
        raise RuntimeError("Gambar tidak dapat dibaca oleh OpenCV")

    height, width = image.shape[:2]
    detections = predict_detections(model, image, conf, imgsz, inference_device)
    annotated = draw_detections(image, detections, show_count=False)
    output_path = image_output_root / source.relative_to(input_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(f"Gagal menulis gambar anotasi: {output_path}")

    confidences = [item["confidence"] for item in detections]
    average_confidence, minimum_confidence, maximum_confidence = confidence_stats(
        confidences
    )
    count = len(detections)
    summary = {
        "source_file": source_file,
        "media_type": "image",
        "width": width,
        "height": height,
        "fps": 0.0,
        "total_frames": 1,
        "processed_frames": 1,
        "duration_seconds": 0.0,
        "total_detections": count,
        "average_detections_per_processed_frame": float(count),
        "minimum_detections": count,
        "maximum_detections": count,
        "average_confidence": average_confidence,
        "minimum_confidence": minimum_confidence,
        "maximum_confidence": maximum_confidence,
        "processing_seconds": time.perf_counter() - started,
        "device": device_label,
        "status": "success",
        "error_message": "",
    }
    rows = detection_rows(source_file, "image", 0, 0.0, detections)
    return summary, rows


def resize_contact_frame(frame: np.ndarray, width: int = 480) -> np.ndarray:
    height = max(1, round(frame.shape[0] * width / frame.shape[1]))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def write_contact_sheet(
    frames: list[tuple[int, float, np.ndarray]], output_path: Path
) -> None:
    if not frames:
        raise RuntimeError("Tidak ada frame untuk contact sheet")

    tiles: list[np.ndarray] = []
    for frame_index, timestamp, frame in frames[:6]:
        tile = resize_contact_frame(frame)
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 34), (15, 18, 24), -1)
        cv2.putText(
            tile,
            f"frame {frame_index} | {timestamp:.2f} s",
            (8, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    columns = min(3, len(tiles))
    rows = math.ceil(len(tiles) / columns)
    tile_height = max(tile.shape[0] for tile in tiles)
    tile_width = max(tile.shape[1] for tile in tiles)
    sheet = np.full((rows * tile_height, columns * tile_width, 3), 28, np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        sheet[
            row * tile_height : row * tile_height + tile.shape[0],
            column * tile_width : column * tile_width + tile.shape[1],
        ] = tile

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"Gagal menulis contact sheet: {output_path}")


def process_video(
    source: Path,
    input_root: Path,
    video_output_root: Path,
    contact_output_root: Path,
    model: YOLO,
    conf: float,
    imgsz: int,
    video_stride: int,
    inference_device: str | int,
    device_label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    source_file = relative_name(source, input_root)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError("Video tidak dapat dibuka oleh OpenCV")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0 or fps <= 0:
        capture.release()
        raise RuntimeError(
            f"Metadata video tidak valid: {width}x{height}, fps={fps:.3f}"
        )

    relative = source.relative_to(input_root)
    output_path = video_output_root / relative.parent / f"{source.stem}_detected.mp4"
    contact_path = contact_output_root / relative.parent / f"{source.stem}_contact_sheet.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    contact_path.parent.mkdir(parents=True, exist_ok=True)

    output_fps = fps / video_stride
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("VideoWriter MP4 tidak dapat dibuat")

    expected_processed = (
        math.ceil(total_frames / video_stride) if total_frames > 0 else 0
    )
    target_ordinals = (
        set(
            int(value)
            for value in np.linspace(
                0, expected_processed - 1, min(6, expected_processed)
            )
        )
        if expected_processed > 0
        else set(range(6))
    )

    all_rows: list[dict[str, Any]] = []
    detection_counts: list[int] = []
    confidences: list[float] = []
    contact_frames: list[tuple[int, float, np.ndarray]] = []
    frame_index = 0
    processed_frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % video_stride != 0:
                frame_index += 1
                continue

            timestamp = frame_index / fps
            detections = predict_detections(
                model, frame, conf, imgsz, inference_device
            )
            annotated = draw_detections(frame, detections, show_count=True)
            writer.write(annotated)

            detection_counts.append(len(detections))
            confidences.extend(item["confidence"] for item in detections)
            all_rows.extend(
                detection_rows(
                    source_file, "video", frame_index, timestamp, detections
                )
            )
            if processed_frames in target_ordinals:
                contact_frames.append((frame_index, timestamp, annotated.copy()))

            processed_frames += 1
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    if processed_frames == 0:
        raise RuntimeError("Video tidak menghasilkan frame yang dapat diproses")

    write_contact_sheet(contact_frames, contact_path)
    average_confidence, minimum_confidence, maximum_confidence = confidence_stats(
        confidences
    )
    summary = {
        "source_file": source_file,
        "media_type": "video",
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "duration_seconds": total_frames / fps if total_frames > 0 else frame_index / fps,
        "total_detections": sum(detection_counts),
        "average_detections_per_processed_frame": sum(detection_counts)
        / processed_frames,
        "minimum_detections": min(detection_counts),
        "maximum_detections": max(detection_counts),
        "average_confidence": average_confidence,
        "minimum_confidence": minimum_confidence,
        "maximum_confidence": maximum_confidence,
        "processing_seconds": time.perf_counter() - started,
        "device": device_label,
        "status": "success",
        "error_message": "",
    }
    return summary, all_rows


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(
    path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def expected_outputs(
    source: Path,
    input_root: Path,
    image_root: Path,
    video_root: Path,
    contact_root: Path,
) -> list[Path]:
    relative = source.relative_to(input_root)
    if source.suffix.lower() in IMAGE_EXTENSIONS:
        return [image_root / relative]
    return [
        video_root / relative.parent / f"{source.stem}_detected.mp4",
        contact_root / relative.parent / f"{source.stem}_contact_sheet.jpg",
    ]


def error_summary(
    source_file: str, media_type: str, device_label: str, error: Exception, elapsed: float
) -> dict[str, Any]:
    summary = {field: "" for field in SUMMARY_FIELDS}
    summary.update(
        {
            "source_file": source_file,
            "media_type": media_type,
            "processing_seconds": elapsed,
            "device": device_label,
            "status": "error",
            "error_message": str(error),
        }
    )
    return summary


def manual_review_items(summaries: list[dict[str, Any]]) -> list[tuple[str, str]]:
    ranked: list[tuple[float, str, str]] = []
    for summary in summaries:
        source = str(summary["source_file"])
        if summary.get("status") != "success":
            ranked.append((10_000.0, source, f"gagal: {summary.get('error_message', '')}"))
            continue

        total = as_float(summary.get("total_detections"))
        average = as_float(summary.get("average_detections_per_processed_frame"))
        minimum = as_float(summary.get("minimum_detections"))
        maximum = as_float(summary.get("maximum_detections"))
        confidence_value = summary.get("average_confidence")
        confidence = as_float(confidence_value, 0.0)
        reasons: list[str] = []
        score = 0.0
        if total == 0:
            score += 1_000.0
            reasons.append("tidak ada deteksi")
        if confidence_value not in (None, ""):
            score += (1.0 - confidence) * 100.0
            if confidence < 0.35:
                reasons.append(f"confidence rata-rata rendah ({confidence:.3f})")
        variation = maximum - minimum
        score += variation * 5.0 + average
        if variation >= 3:
            reasons.append(f"jumlah deteksi bervariasi {minimum:.0f}-{maximum:.0f}")
        if average >= 8:
            reasons.append(f"adegan padat ({average:.2f}/frame)")
        if not reasons:
            reasons.append("sampel pemeriksaan representatif")
        ranked.append((score, source, "; ".join(reasons)))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [(source, reason) for _, source, reason in ranked[:5]]


def write_report(
    path: Path,
    summaries: list[dict[str, Any]],
    image_count: int,
    video_count: int,
    config: dict[str, Any],
    elapsed: float,
) -> list[tuple[str, str]]:
    successful = [item for item in summaries if item.get("status") == "success"]
    failed = [item for item in summaries if item.get("status") == "error"]
    detection_ranked = sorted(
        successful,
        key=lambda item: as_float(item.get("average_detections_per_processed_frame")),
    )
    confidence_ranked = sorted(
        [item for item in successful if item.get("average_confidence") not in (None, "")],
        key=lambda item: as_float(item.get("average_confidence")),
    )
    reviews = manual_review_items(summaries)

    lines = [
        "# Campus Baseline Inference Report",
        "",
        "Baseline ini hanya menguji model `best.pt` yang sudah ada dalam mode "
        "`detection_only`. Tidak ada training, ROI, perhitungan gap, P3, atau P4.",
        "",
        "> Statistik pada laporan ini **bukan ukuran accuracy, precision, recall, "
        "atau mAP** karena data belum memiliki ground truth.",
        "",
        "## Ringkasan",
        "",
        f"- Foto ditemukan: {image_count}",
        f"- Video ditemukan: {video_count}",
        f"- Berhasil diproses: {len(successful)}",
        f"- Gagal diproses: {len(failed)}",
        f"- Total waktu run: {elapsed:.2f} detik",
        "",
        "## Konfigurasi inferensi",
        "",
        f"- Model: `{config['model']}`",
        f"- SHA-256 model: `{config['model_sha256']}`",
        f"- Device: `{config['resolved_device']}`",
        f"- Confidence threshold: `{config['conf']}`",
        f"- Image size: `{config['imgsz']}`",
        f"- Video stride: `{config['video_stride']}`",
        "- Mode: `detection_only` tanpa ROI atau estimasi gap",
        "",
        "## Perbandingan file",
        "",
    ]

    if detection_ranked:
        lowest = detection_ranked[0]
        highest = detection_ranked[-1]
        lines.extend(
            [
                f"- Deteksi rata-rata tertinggi: `{highest['source_file']}` "
                f"({as_float(highest['average_detections_per_processed_frame']):.3f})",
                f"- Deteksi rata-rata terendah: `{lowest['source_file']}` "
                f"({as_float(lowest['average_detections_per_processed_frame']):.3f})",
            ]
        )
    else:
        lines.append("- Tidak ada file sukses untuk dibandingkan berdasarkan deteksi.")

    if confidence_ranked:
        lowest = confidence_ranked[0]
        highest = confidence_ranked[-1]
        lines.extend(
            [
                f"- Confidence rata-rata tertinggi: `{highest['source_file']}` "
                f"({as_float(highest['average_confidence']):.3f})",
                f"- Confidence rata-rata terendah: `{lowest['source_file']}` "
                f"({as_float(lowest['average_confidence']):.3f})",
            ]
        )
    else:
        lines.append("- Tidak ada deteksi untuk dibandingkan berdasarkan confidence.")

    lines.extend(["", "## Hasil yang perlu diperiksa manual", ""])
    for source, reason in reviews:
        lines.append(f"- `{source}` — {reason}")
    if not reviews:
        lines.append("- Tidak ada file untuk diperiksa.")

    if failed:
        lines.extend(["", "## File gagal", ""])
        for item in failed:
            lines.append(f"- `{item['source_file']}` — {item.get('error_message', '')}")

    lines.extend(
        [
            "",
            "## Catatan interpretasi",
            "",
            "Jumlah deteksi dan confidence hanya membantu inspeksi baseline dan "
            "pemilihan sampel review. Nilai tersebut tidak membuktikan bahwa box "
            "benar, tidak membuktikan objek yang terlewat, dan tidak dapat digunakan "
            "sebagai metrik kualitas tanpa anotasi ground truth.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return reviews


def main() -> int:
    args = parse_args()
    started_wall = time.perf_counter()
    started_at = datetime.now().astimezone().isoformat()
    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    model_path = Path(args.model).resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"Folder input tidak ditemukan: {input_root}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf harus berada pada rentang 0-1")
    if args.imgsz <= 0:
        raise ValueError("--imgsz harus positif")
    if args.video_stride <= 0:
        raise ValueError("--video-stride harus positif")
    try:
        output_root.relative_to(input_root)
    except ValueError:
        pass
    else:
        raise ValueError("Folder output tidak boleh berada di dalam folder input")

    image_root = output_root / "annotated_images"
    video_root = output_root / "annotated_videos"
    contact_root = output_root / "contact_sheets"
    for path in (output_root, image_root, video_root, contact_root):
        path.mkdir(parents=True, exist_ok=True)

    images, videos = discover_media(input_root)
    all_media = images + videos
    if not all_media:
        raise RuntimeError(f"Tidak ada media yang didukung di {input_root}")

    inference_device, device_label = resolve_device(args.device)
    model_hash = sha256_file(model_path)
    config: dict[str, Any] = {
        "started_at": started_at,
        "completed_at": None,
        "input": str(input_root),
        "output": str(output_root),
        "model": str(model_path),
        "model_sha256": model_hash,
        "requested_device": args.device,
        "resolved_device": device_label,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "video_stride": args.video_stride,
        "overwrite": args.overwrite,
        "mode": "detection_only",
        "roi_enabled": False,
        "gap_calculation_enabled": False,
        "found_images": len(images),
        "found_videos": len(videos),
    }
    run_config_path = output_root / "run_config.json"
    run_config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Media ditemukan : {len(images)} foto, {len(videos)} video")
    print(f"Device          : {device_label}")
    print(f"Model SHA-256   : {model_hash}")
    print("Mode            : detection_only (tanpa ROI/gap)")
    model = YOLO(str(model_path))

    detections_path = output_root / "detections.csv"
    summary_path = output_root / "file_summary.csv"
    previous_detections = read_csv_rows(detections_path) if not args.overwrite else []
    previous_summaries = read_csv_rows(summary_path) if not args.overwrite else []
    found_names = {relative_name(path, input_root) for path in all_media}
    detection_data = [
        row for row in previous_detections if row.get("source_file") in found_names
    ]
    summaries_by_source = {
        str(row["source_file"]): row
        for row in previous_summaries
        if row.get("source_file") in found_names
    }
    error_details: list[str] = []

    def checkpoint() -> None:
        ordered_summaries = [
            summaries_by_source[name] for name in sorted(summaries_by_source)
        ]
        write_csv_rows(detections_path, DETECTION_FIELDS, detection_data)
        write_csv_rows(summary_path, SUMMARY_FIELDS, ordered_summaries)

    def process_one(source: Path, smoke: bool = False) -> dict[str, Any]:
        nonlocal detection_data
        source_file = relative_name(source, input_root)
        media_type = "image" if source.suffix.lower() in IMAGE_EXTENSIONS else "video"
        prefix = "SMOKE" if smoke else "FILE"
        expected = expected_outputs(
            source, input_root, image_root, video_root, contact_root
        )
        previous = summaries_by_source.get(source_file)
        if (
            not args.overwrite
            and previous is not None
            and previous.get("status") == "success"
            and all(path.exists() and path.stat().st_size > 0 for path in expected)
        ):
            print(f"[{prefix}] RESUME {source_file}")
            return previous

        print(f"[{prefix}] PROSES {source_file}")
        detection_data = [
            row for row in detection_data if row.get("source_file") != source_file
        ]
        item_started = time.perf_counter()
        try:
            if media_type == "image":
                summary, rows = process_image(
                    source,
                    input_root,
                    image_root,
                    model,
                    args.conf,
                    args.imgsz,
                    inference_device,
                    device_label,
                )
            else:
                summary, rows = process_video(
                    source,
                    input_root,
                    video_root,
                    contact_root,
                    model,
                    args.conf,
                    args.imgsz,
                    args.video_stride,
                    inference_device,
                    device_label,
                )
            detection_data.extend(rows)
            print(
                f"[{prefix}] OK {source_file}: "
                f"{summary['total_detections']} deteksi"
            )
        except Exception as error:  # lanjutkan media berikutnya setelah smoke gate
            summary = error_summary(
                source_file,
                media_type,
                device_label,
                error,
                time.perf_counter() - item_started,
            )
            error_details.append(
                f"[{source_file}] {error}\n{traceback.format_exc()}"
            )
            print(f"[{prefix}] ERROR {source_file}: {error}")
        summaries_by_source[source_file] = summary
        checkpoint()
        return summary

    smoke_files: list[Path] = []
    if images:
        smoke_files.append(images[0])
    if videos:
        smoke_files.append(videos[0])
    smoke_ok = True
    for source in smoke_files:
        smoke_ok = process_one(source, smoke=True).get("status") == "success" and smoke_ok

    if smoke_ok:
        smoke_set = set(smoke_files)
        for source in all_media:
            if source not in smoke_set:
                process_one(source)
    else:
        print("Smoke test gagal; pemrosesan penuh tidak dilanjutkan.")

    checkpoint()
    elapsed = time.perf_counter() - started_wall
    ordered_summaries = [
        summaries_by_source[name] for name in sorted(summaries_by_source)
    ]
    reviews = write_report(
        output_root / "baseline_report.md",
        ordered_summaries,
        len(images),
        len(videos),
        config,
        elapsed,
    )
    errors_path = output_root / "errors.log"
    errors_path.write_text(
        "\n\n".join(error_details) + ("\n" if error_details else ""),
        encoding="utf-8",
    )

    successful_count = sum(
        item.get("status") == "success" for item in ordered_summaries
    )
    failed_count = sum(item.get("status") == "error" for item in ordered_summaries)
    config.update(
        {
            "completed_at": datetime.now().astimezone().isoformat(),
            "processing_seconds": elapsed,
            "successful_files": successful_count,
            "failed_files": failed_count,
            "smoke_test_passed": smoke_ok,
            "manual_review": [
                {"source_file": source, "reason": reason}
                for source, reason in reviews
            ],
        }
    )
    run_config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Berhasil         : {successful_count}")
    print(f"Gagal            : {failed_count}")
    print(f"Waktu total      : {elapsed:.2f} detik")
    print(f"Output           : {output_root}")
    return 0 if smoke_ok and failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
