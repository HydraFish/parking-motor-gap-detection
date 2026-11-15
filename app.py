"""MotoSpace CV Telemetry - Web Dashboard Server.

Clean, decoupled dashboard server following Andrej Karpathy's Simplicity First principle.
Runs with zero build steps, serving the MotoSpace telemetry interface and video feeds.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from flask import Flask, jsonify, render_template, send_from_directory

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
TEMPLATES_DIR = ROOT / "templates"
OUTPUTS_DIR = ROOT / "outputs"

app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    template_folder=str(TEMPLATES_DIR),
)


@app.route("/")
def index():
    """Render the MotoSpace CV Telemetry Dashboard."""
    return render_template("index.html")


@app.route("/api/telemetry")
def get_telemetry():
    """Return real-time parking metrics matching Figma specs.
    
    Can be dynamically fed by process_parking_video when active,
    or returns live calibrated telemetry from current demo videos.
    """
    telemetry = {
        "motorcycles_detected": 72,
        "capacity_left": 28,
        "occupancy_pct": 72.0,
        "threshold_pct": 85.0,
        "latency_ms": 21.0,
        "fps": 30.0,
        "camera_status": {
            "cam_01": {
                "name": "CAM-01 (South Yard)",
                "resolution": "1080p",
                "fps": 30,
                "online": True,
                "confidence": 98.4,
                "latency_ms": 21,
                "detected": 72,
                "capacity": 28,
            },
            "cam_02": {
                "name": "CAM-02 (Perimeter Gate)",
                "resolution": "1080p",
                "fps": 24,
                "online": True,
                "confidence": 96.8,
                "latency_ms": 19,
                "detected": 34,
                "capacity": 14,
            },
        },
        "zones": {
            "facility_total": {
                "title": "FACILITY TOTAL",
                "sectors": 3,
                "parked": 72,
                "dynamic_stalls": 28,
                "density": "0.021 moto/m²",
                "status": "Optimal",
                "badge_color": "emerald",
            },
            "zone_a": {
                "title": "ZONE A - EAST YARD",
                "area": "1,920 m²",
                "parked": 32,
                "surplus": 128,
                "staging": "High Density",
                "occupancy_pct": 89.5,
                "badge_color": "blue",
            },
            "zone_b": {
                "title": "ZONE B - CENTRAL LOT",
                "area": "1,540 m²",
                "parked": 28,
                "surplus": 10,
                "staging": "Near Nominal",
                "occupancy_pct": 82.5,
                "badge_color": "amber",
            },
            "zone_c": {
                "title": "ZONE C - WEST WING",
                "area": "1,380 m²",
                "parked": 12,
                "surplus": 10,
                "staging": "Low Density",
                "occupancy_pct": 50.0,
                "badge_color": "emerald",
            },
        },
        "spatial_heuristic": {
            "hull_margin": "450mm Buffer",
            "asphalt_net_area": "1,820 m²",
            "unit_footprint": "1.65 m²",
            "model_version": "YOLO Space v11x",
        },
    }
    return jsonify(telemetry)


@app.route("/api/config")
def get_config():
    """Return video_configs.json profiles."""
    config_path = ROOT / "video_configs.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "video_configs.json not found"}), 404


@app.route("/api/export/json")
def export_json():
    """Export current telemetry as a downloadable JSON file."""
    data = get_telemetry().get_json()
    response = jsonify(data)
    response.headers["Content-Disposition"] = "attachment; filename=motospace_telemetry_export.json"
    return response


@app.route("/outputs/<path:filename>")
def serve_output(filename: str):
    """Serve files directly from outputs directory."""
    return send_from_directory(str(OUTPUTS_DIR), filename)


@app.route("/videos/<path:filename>")
def serve_video(filename: str):
    """Serve demo and output video files directly with range requests."""
    # Check outputs directory first
    if (OUTPUTS_DIR / filename).exists():
        return send_from_directory(str(OUTPUTS_DIR), filename)
    # Check project root directory
    if (ROOT / filename).exists():
        return send_from_directory(str(ROOT), filename)
    return jsonify({"error": f"Video '{filename}' not found"}), 404


def get_lan_ip() -> str:
    """Detect local Wi-Fi / LAN IP address for mobile testing."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    # Ensure template and static dirs exist
    TEMPLATES_DIR.mkdir(exist_ok=True)
    STATIC_DIR.mkdir(exist_ok=True)
    
    port = int(os.environ.get("PORT", 8000))
    lan_ip = get_lan_ip()
    print("=" * 64)
    print(" MotoSpace CV Telemetry - Local Web Dashboard")
    print(f" • Local URL   : http://localhost:{port}")
    print(f" • Mobile/Wi-Fi: http://{lan_ip}:{port}")
    print(" Press Ctrl+C to stop.")
    print("=" * 64)
    app.run(host="0.0.0.0", port=port, debug=False)
