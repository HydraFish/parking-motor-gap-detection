"""Siapkan pilot dataset kampus dan pre-label tanpa training atau augmentasi."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}
MANIFEST_FIELDS = [
    "image_file",
    "source_video",
    "source_frame",
    "timestamp_seconds",
    "blur_score",
    "duplicate_group",
    "prelabel_count",
    "status",
    "split_recommendation",
    "perceptual_hash",
    "change_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ekstrak pilot dataset kampus, review blur/duplikat, dan buat "
            "pre-label dari best_baseline.pt. Script ini tidak melakukan training."
        )
    )
    parser.add_argument("--input", default="data", help="Folder video sumber")
    parser.add_argument(
        "--output", default="data/campus_dataset_pilot", help="Folder dataset pilot"
    )
    parser.add_argument(
        "--model", default="best_baseline.pt", help="Model khusus pre-label"
    )
    parser.add_argument(
        "--interval-seconds", type=float, default=2.0, help="Interval ekstraksi frame"
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=60.0,
        help="Variance of Laplacian di bawah nilai ini masuk review_blurry",
    )
    parser.add_argument(
        "--duplicate-distance",
        type=int,
        default=4,
        help="Maksimum Hamming distance pHash untuk kelompok hampir duplikat",
    )
    parser.add_argument(
        "--duplicate-mad-threshold",
        type=float,
        default=1.0,
        help="Batas mean absolute difference thumbnail setelah kandidat pHash cocok",
    )
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence pre-label")
    parser.add_argument("--imgsz", type=int, default=960, help="Ukuran inferensi YOLO")
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, cuda:0")
    parser.add_argument(
        "--target-max",
        type=int,
        default=300,
        help="Jumlah maksimum frame unik sebelum seleksi perubahan visual",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Izinkan menimpa file dataset pilot yang sudah ada",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> tuple[str | int, str]:
    normalized = requested.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return 0, f"cuda:0 ({torch.cuda.get_device_name(0)})"
        return "cpu", "cpu"
    if normalized.isdigit():
        return int(normalized), f"cuda:{normalized}"
    return requested, requested


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def discover_videos(input_root: Path, output_root: Path) -> list[Path]:
    return sorted(
        path
        for path in input_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
        and not is_within(path, output_root)
    )


def safe_source_id(relative_video: Path) -> str:
    stem = relative_video.with_suffix("").as_posix()
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")


def blur_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def perceptual_hash(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    transformed = cv2.dct(np.float32(resized))[:8, :8]
    values = transformed.flatten()
    threshold = float(np.median(values[1:]))
    result = 0
    for value in values:
        result = (result << 1) | int(value > threshold)
    return result


def hamming_distance(first: int, second: int) -> int:
    return (first ^ second).bit_count()


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def move_generated_file(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    shutil.move(str(source), str(destination))
    return destination


def extract_frames(
    videos: list[Path],
    input_root: Path,
    images_root: Path,
    blurry_root: Path,
    interval_seconds: float,
    blur_threshold: float,
) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    unreadable = 0
    for video_index, video in enumerate(videos, start=1):
        relative_video = video.relative_to(input_root)
        source_id = safe_source_id(relative_video)
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            print(f"[EKSTRAK] GAGAL membuka {relative_video.as_posix()}")
            unreadable += 1
            continue
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_frames <= 0:
            capture.release()
            print(f"[EKSTRAK] Metadata tidak valid {relative_video.as_posix()}")
            unreadable += 1
            continue

        duration = total_frames / fps
        timestamps = np.arange(0.0, duration, interval_seconds)
        previous_hash: int | None = None
        extracted_for_video = 0
        for timestamp in timestamps:
            source_frame = min(total_frames - 1, int(round(float(timestamp) * fps)))
            capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                unreadable += 1
                continue

            actual_timestamp = source_frame / fps
            score = blur_score(frame)
            image_hash = perceptual_hash(frame)
            change = (
                float(hamming_distance(previous_hash, image_hash))
                if previous_hash is not None
                else 64.0
            )
            previous_hash = image_hash
            filename = (
                f"{source_id}__f{source_frame:08d}"
                f"__t{round(actual_timestamp * 1000):010d}ms.jpg"
            )
            image_path = images_root / filename
            if not cv2.imwrite(str(image_path), frame):
                unreadable += 1
                continue

            status = "candidate"
            if score < blur_threshold:
                image_path = move_generated_file(image_path, blurry_root / filename)
                status = "blurry"
            records.append(
                {
                    "image_file": image_path,
                    "source_video": relative_video.as_posix(),
                    "source_frame": source_frame,
                    "timestamp_seconds": actual_timestamp,
                    "blur_score": score,
                    "duplicate_group": "",
                    "prelabel_count": 0,
                    "status": status,
                    "split_recommendation": "",
                    "perceptual_hash": image_hash,
                    "change_score": change,
                    "_boxes": [],
                }
            )
            extracted_for_video += 1
        capture.release()
        print(
            f"[EKSTRAK] {video_index}/{len(videos)} "
            f"{relative_video.as_posix()}: {extracted_for_video} frame"
        )
    return records, unreadable


def deduplicate_frames(
    records: list[dict[str, Any]],
    duplicates_root: Path,
    duplicate_distance: int,
    duplicate_mad_threshold: float,
) -> tuple[int, int]:
    candidates = [record for record in records if record["status"] == "candidate"]
    thumbnails: list[np.ndarray] = []
    for record in candidates:
        image = cv2.imread(str(record["image_file"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            thumbnails.append(np.zeros((64, 64), dtype=np.uint8))
        else:
            thumbnails.append(
                cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
            )
    union_find = UnionFind(len(candidates))
    for first in range(len(candidates)):
        for second in range(first + 1, len(candidates)):
            hash_distance = hamming_distance(
                int(candidates[first]["perceptual_hash"]),
                int(candidates[second]["perceptual_hash"]),
            )
            if hash_distance > duplicate_distance:
                continue
            pixel_difference = float(
                np.mean(
                    np.abs(
                        thumbnails[first].astype(np.float32)
                        - thumbnails[second].astype(np.float32)
                    )
                )
            )
            if pixel_difference <= duplicate_mad_threshold:
                union_find.union(first, second)

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(candidates):
        grouped[union_find.find(index)].append(record)

    duplicate_groups = 0
    removed = 0
    for group in grouped.values():
        if len(group) == 1:
            group[0]["status"] = "selected"
            continue
        duplicate_groups += 1
        group_id = f"dup_{duplicate_groups:04d}"
        for record in group:
            record["duplicate_group"] = group_id
        best = max(
            group,
            key=lambda record: (
                float(record["blur_score"]),
                float(record["change_score"]),
                -int(record["source_frame"]),
            ),
        )
        best["status"] = "selected"
        for record in group:
            if record is best:
                continue
            current = Path(record["image_file"])
            record["image_file"] = move_generated_file(
                current, duplicates_root / current.name
            )
            record["status"] = "duplicate"
            removed += 1
    return duplicate_groups, removed


def cap_selected_frames(
    records: list[dict[str, Any]], duplicates_root: Path, target_max: int
) -> int:
    selected = [record for record in records if record["status"] == "selected"]
    if len(selected) <= target_max:
        return 0

    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in selected:
        by_video[str(record["source_video"])].append(record)
    mandatory = {
        id(
            max(
                group,
                key=lambda record: (
                    float(record["change_score"]),
                    float(record["blur_score"]),
                ),
            )
        )
        for group in by_video.values()
    }
    ranked = sorted(
        selected,
        key=lambda record: (
            id(record) not in mandatory,
            -float(record["change_score"]),
            -float(record["blur_score"]),
            str(record["source_video"]),
            int(record["source_frame"]),
        ),
    )
    keep = {id(record) for record in ranked[:target_max]}
    removed = 0
    for record in selected:
        if id(record) in keep:
            continue
        current = Path(record["image_file"])
        record["image_file"] = move_generated_file(
            current, duplicates_root / current.name
        )
        record["status"] = "not_selected_change"
        removed += 1
    return removed


def recommend_splits(videos: list[Path], input_root: Path) -> dict[str, str]:
    relative_names = [video.relative_to(input_root).as_posix() for video in videos]
    ordered = sorted(
        relative_names,
        key=lambda name: hashlib.sha256(name.encode("utf-8")).hexdigest(),
    )
    count = len(ordered)
    validation_count = max(1, round(count * 0.15)) if count >= 3 else 0
    test_count = max(1, round(count * 0.15)) if count >= 3 else 0
    train_count = count - validation_count - test_count
    result: dict[str, str] = {}
    for name in ordered[:train_count]:
        result[name] = "train"
    for name in ordered[train_count : train_count + validation_count]:
        result[name] = "validation"
    for name in ordered[train_count + validation_count :]:
        result[name] = "test"
    return result


def prelabel_selected(
    records: list[dict[str, Any]],
    labels_root: Path,
    model_path: Path,
    conf: float,
    imgsz: int,
    device: str | int,
) -> int:
    model = YOLO(str(model_path))
    total_labels = 0
    selected = [record for record in records if record["status"] == "selected"]
    for index, record in enumerate(selected, start=1):
        image_path = Path(record["image_file"])
        image = cv2.imread(str(image_path))
        if image is None:
            record["status"] = "corrupt_after_selection"
            continue
        result = model.predict(
            image, conf=conf, imgsz=imgsz, device=device, verbose=False
        )[0]
        lines: list[str] = []
        boxes_for_contact: list[tuple[float, float, float, float, float]] = []
        if result.boxes is not None:
            xywhn = result.boxes.xywhn.detach().cpu().numpy()
            confidences = result.boxes.conf.detach().cpu().numpy()
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            for normalized, confidence, pixel_box in zip(xywhn, confidences, xyxy):
                center_x, center_y, width, height = map(float, normalized)
                lines.append(
                    f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
                )
                x1, y1, x2, y2 = map(float, pixel_box)
                boxes_for_contact.append((x1, y1, x2, y2, float(confidence)))
        label_path = labels_root / f"{image_path.stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        record["prelabel_count"] = len(lines)
        record["_boxes"] = boxes_for_contact
        total_labels += len(lines)
        if index % 25 == 0 or index == len(selected):
            print(f"[PRELABEL] {index}/{len(selected)} frame")
    return total_labels


def annotate_for_contact(
    image: np.ndarray,
    boxes: list[tuple[float, float, float, float, float]],
    title: str,
) -> np.ndarray:
    annotated = image.copy()
    for x1, y1, x2, y2, confidence in boxes:
        a = (int(round(x1)), int(round(y1)))
        b = (int(round(x2)), int(round(y2)))
        cv2.rectangle(annotated, a, b, (255, 90, 20), 2)
        cv2.putText(
            annotated,
            f"Motor {confidence:.2f}",
            (a[0], max(18, a[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 90, 20),
            1,
            cv2.LINE_AA,
        )
    target_width = 480
    target_height = max(1, round(annotated.shape[0] * target_width / annotated.shape[1]))
    tile = cv2.resize(annotated, (target_width, target_height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 34), (15, 18, 24), -1)
    cv2.putText(
        tile,
        title,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def create_contact_sheets(
    records: list[dict[str, Any]], contact_root: Path
) -> int:
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["status"] == "selected":
            by_video[str(record["source_video"])].append(record)

    created = 0
    for source_video, group in sorted(by_video.items()):
        group.sort(key=lambda record: int(record["source_frame"]))
        count = min(6, len(group))
        indexes = sorted(
            set(int(value) for value in np.linspace(0, len(group) - 1, count))
        )
        tiles: list[np.ndarray] = []
        for index in indexes:
            record = group[index]
            image = cv2.imread(str(record["image_file"]))
            if image is None:
                continue
            title = (
                f"frame {record['source_frame']} | "
                f"{float(record['timestamp_seconds']):.2f} s"
            )
            tiles.append(annotate_for_contact(image, record["_boxes"], title))
        if not tiles:
            continue
        columns = min(3, len(tiles))
        rows = math.ceil(len(tiles) / columns)
        tile_height = max(tile.shape[0] for tile in tiles)
        sheet = np.full((rows * tile_height, columns * 480, 3), 28, np.uint8)
        for index, tile in enumerate(tiles):
            row, column = divmod(index, columns)
            sheet[
                row * tile_height : row * tile_height + tile.shape[0],
                column * 480 : (column + 1) * 480,
            ] = tile
        output_name = f"{safe_source_id(Path(source_video))}_contact_sheet.jpg"
        if cv2.imwrite(str(contact_root / output_name), sheet):
            created += 1
    return created


def write_manifest(
    records: list[dict[str, Any]], output_root: Path
) -> None:
    manifest_path = output_root / "source_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in sorted(
            records,
            key=lambda item: (str(item["source_video"]), int(item["source_frame"])),
        ):
            row = dict(record)
            row["image_file"] = Path(row["image_file"]).relative_to(output_root).as_posix()
            row["timestamp_seconds"] = f"{float(row['timestamp_seconds']):.6f}"
            row["blur_score"] = f"{float(row['blur_score']):.6f}"
            row["perceptual_hash"] = f"{int(row['perceptual_hash']):016x}"
            row["change_score"] = f"{float(row['change_score']):.3f}"
            writer.writerow(row)


def write_report(
    output_root: Path,
    videos: list[Path],
    input_root: Path,
    records: list[dict[str, Any]],
    unreadable: int,
    duplicate_groups: int,
    duplicate_removed: int,
    change_removed: int,
    total_prelabels: int,
    contact_sheets: int,
    splits: dict[str, str],
    config: dict[str, Any],
    elapsed: float,
) -> None:
    selected = [record for record in records if record["status"] == "selected"]
    blurry = [record for record in records if record["status"] == "blurry"]
    split_videos: dict[str, list[str]] = defaultdict(list)
    split_frames: dict[str, int] = defaultdict(int)
    for video, split in splits.items():
        split_videos[split].append(video)
    for record in selected:
        split_frames[str(record["split_recommendation"])] += 1

    lines = [
        "# Campus Dataset Pilot Preparation Report",
        "",
        "> Dataset ini belum memiliki anotasi ground truth. Label pada "
        "`labels_preliminary/` adalah pre-label model yang wajib diperiksa dan "
        "dikoreksi manual sebelum training.",
        "",
        "## Ringkasan",
        "",
        f"- Video ditemukan: {len(videos)}",
        f"- Frame awal hasil ekstraksi 2 detik: {len(records)}",
        f"- Frame rusak/tidak terbaca: {unreadable}",
        f"- Kandidat buram: {len(blurry)}",
        f"- Kelompok hampir duplikat: {duplicate_groups}",
        f"- Frame duplikat dipindahkan untuk review: {duplicate_removed}",
        f"- Frame tidak terpilih karena batas target: {change_removed}",
        f"- Frame unik terpilih: {len(selected)}",
        f"- Total pre-label: {total_prelabels}",
        f"- Contact sheet: {contact_sheets}",
        f"- Waktu persiapan: {elapsed:.2f} detik",
        "",
        "## Konfigurasi",
        "",
        f"- Interval ekstraksi: `{config['interval_seconds']}` detik",
        f"- Blur threshold: `{config['blur_threshold']}`",
        f"- pHash Hamming distance: `<= {config['duplicate_distance']}`",
        f"- Verifikasi thumbnail MAD: `<= {config['duplicate_mad_threshold']}`",
        f"- Model pre-label: `{config['model']}`",
        f"- SHA-256 model: `{config['model_sha256']}`",
        f"- Confidence pre-label: `{config['conf']}`",
        f"- Image size: `{config['imgsz']}`",
        f"- Device: `{config['device']}`",
        "- Augmentasi: tidak dilakukan",
        "- Training/fine-tuning: tidak dilakukan",
        "",
        "## Rekomendasi split berbasis video",
        "",
        "Semua frame dari satu video ditempatkan pada split yang sama untuk "
        "mencegah kebocoran frame yang sangat mirip.",
        "",
    ]
    for split in ("train", "validation", "test"):
        names = sorted(split_videos.get(split, []))
        lines.append(
            f"### {split.title()} — {len(names)} video, "
            f"{split_frames.get(split, 0)} frame"
        )
        lines.append("")
        for name in names:
            lines.append(f"- `{name}`")
        lines.append("")

    lines.extend(
        [
            "## Catatan penggunaan",
            "",
            "- Jangan memakai pre-label sebagai ground truth tanpa inspeksi manual.",
            "- Periksa terutama `review_blurry/`, `review_duplicates/`, dan contact sheet.",
            "- Split yang direkomendasikan berbasis video, bukan acak berdasarkan frame.",
            "- Script ini tidak melakukan augmentasi atau training.",
            "",
        ]
    )
    (output_root / "preparation_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    model_path = Path(args.model).resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"Folder input tidak ditemukan: {input_root}")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model pre-label tidak ditemukan: {model_path}. "
            "best.pt tidak akan digunakan sebagai pengganti."
        )
    if args.interval_seconds <= 0:
        raise ValueError("--interval-seconds harus positif")
    if not 0 <= args.duplicate_distance <= 64:
        raise ValueError("--duplicate-distance harus berada pada rentang 0-64")
    if args.duplicate_mad_threshold < 0:
        raise ValueError("--duplicate-mad-threshold tidak boleh negatif")
    if not 0 <= args.conf <= 1:
        raise ValueError("--conf harus berada pada rentang 0-1")
    if args.target_max <= 0:
        raise ValueError("--target-max harus positif")

    manifest_path = output_root / "source_manifest.csv"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Dataset pilot sudah ada: {manifest_path}. Gunakan --overwrite untuk menimpa."
        )

    videos = discover_videos(input_root, output_root)
    if not videos:
        raise RuntimeError(f"Tidak ada video yang didukung di {input_root}")

    images_root = output_root / "images_all"
    labels_root = output_root / "labels_preliminary"
    blurry_root = output_root / "review_blurry"
    duplicates_root = output_root / "review_duplicates"
    contact_root = output_root / "contact_sheets"
    for folder in (
        images_root,
        labels_root,
        blurry_root,
        duplicates_root,
        contact_root,
    ):
        folder.mkdir(parents=True, exist_ok=True)

    inference_device, device_label = resolve_device(args.device)
    print(f"Video ditemukan : {len(videos)}")
    print(f"Model pre-label : {model_path}")
    print(f"Device          : {device_label}")
    print("Training        : tidak dilakukan")

    records, unreadable = extract_frames(
        videos,
        input_root,
        images_root,
        blurry_root,
        args.interval_seconds,
        args.blur_threshold,
    )
    duplicate_groups, duplicate_removed = deduplicate_frames(
        records,
        duplicates_root,
        args.duplicate_distance,
        args.duplicate_mad_threshold,
    )
    change_removed = cap_selected_frames(records, duplicates_root, args.target_max)

    splits = recommend_splits(videos, input_root)
    for record in records:
        record["split_recommendation"] = splits[str(record["source_video"])]

    total_prelabels = prelabel_selected(
        records,
        labels_root,
        model_path,
        args.conf,
        args.imgsz,
        inference_device,
    )
    contact_sheets = create_contact_sheets(records, contact_root)
    write_manifest(records, output_root)

    elapsed = time.perf_counter() - started
    config = {
        "interval_seconds": args.interval_seconds,
        "blur_threshold": args.blur_threshold,
        "duplicate_distance": args.duplicate_distance,
        "duplicate_mad_threshold": args.duplicate_mad_threshold,
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "conf": args.conf,
        "imgsz": args.imgsz,
        "device": device_label,
    }
    write_report(
        output_root,
        videos,
        input_root,
        records,
        unreadable,
        duplicate_groups,
        duplicate_removed,
        change_removed,
        total_prelabels,
        contact_sheets,
        splits,
        config,
        elapsed,
    )

    selected_count = sum(record["status"] == "selected" for record in records)
    blurry_count = sum(record["status"] == "blurry" for record in records)
    print(f"Frame awal      : {len(records)}")
    print(f"Frame unik      : {selected_count}")
    print(f"Kandidat buram  : {blurry_count}")
    print(f"Grup duplikat   : {duplicate_groups}")
    print(f"Total pre-label : {total_prelabels}")
    print(f"Output           : {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
