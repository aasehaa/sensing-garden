"""HTTP route handlers for the DOT receiver."""

import re
import json
import shutil
from pathlib import Path
from datetime import datetime
from flask import request, jsonify

from ..settings import load_config

logger = None


def register_routes(app):
    """Register all routes on the Flask application."""
    global logger
    logger = app.logger

    phone_to_dot_mapping = {}
    used_dot_slots = set()

    tracker = app.config.get("TRACKER")

    def load_bugcam_dot_ids():
        """Load DOT IDs from bugcam config file."""
        config = load_config()
        return config.get("dot_ids", [])

    def get_or_assign_dot_id(device_id, device_name):
        """Get existing DOT ID for phone, or assign an available one."""
        nonlocal phone_to_dot_mapping, used_dot_slots

        if device_id in phone_to_dot_mapping:
            return phone_to_dot_mapping[device_id]

        allowed_dot_ids = load_bugcam_dot_ids()

        if not allowed_dot_ids:
            logger.debug("No bugcam config found - accepting all phones")
            return None

        for dot_id in allowed_dot_ids:
            if dot_id not in used_dot_slots:
                phone_to_dot_mapping[device_id] = dot_id
                used_dot_slots.add(dot_id)
                logger.info(f"Assigned {dot_id} to phone {device_name} ({device_id[:8]}...)")
                return dot_id

        logger.warning(f"Rejected phone {device_name} ({device_id[:8]}...) - all DOT slots in use")
        return None

    def extract_frame_number(filename):
        """Extract frame number from filename like 'frame_000000.jpg' -> 0"""
        match = re.match(r'frame_(\d+)\.jpg', filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def parse_track_id(track_id_with_time):
        """Parse track ID with embedded time format: {track_id}_{HHMMSS}"""
        if not track_id_with_time:
            return None, None

        match = re.match(r'^([a-zA-Z0-9]+)_(\d{6})$', track_id_with_time)
        if match:
            return match.group(1), match.group(2)

        return None, None

    def find_track_folder(crops_dir_path, track_id):
        """Check if a crop folder for this track_id already exists (ignoring timestamp suffix).

        Scans crops/ for a subdirectory starting with '{track_id}_' and returns
        its full folder name (e.g., '0000002a_143052'), or None if none exists.
        Used to merge batch uploads of the same track into the first batch's folder.
        """
        if not crops_dir_path or not crops_dir_path.exists():
            return None
        prefix = f"{track_id}_"
        for entry in crops_dir_path.iterdir():
            if entry.is_dir() and entry.name.startswith(prefix):
                return entry.name
        return None

    def convert_dot_labels_to_bugcam_format(labels_data):
        """Convert DOT label format (from iPhone) to bugcam-expected format."""
        if "points" not in labels_data:
            return labels_data

        resolution = labels_data.get("resolution", {})
        res_width = resolution.get("width", 3840) if isinstance(resolution, dict) else 3840

        scale = 2.0 if res_width <= 2400 else 1.0

        frames = []
        for seq_idx, point in enumerate(labels_data.get("points", [])):
            if point.get("frameIndex") is not None:
                frames.append({
                    "frame_number": seq_idx,
                    "bbox": [
                        point.get("x", 0) * scale,
                        point.get("y", 0) * scale,
                        point.get("width", 0) * scale,
                        point.get("height", 0) * scale
                    ]
                })

        return {"frames": frames}

    input_storage = Path(app.config["INPUT_STORAGE"])

    @app.route('/upload_crops', methods=['POST'])
    def upload_crops():
        device_id = request.headers.get('X-Device-ID', 'unknown')
        device_name = request.headers.get('X-Device-Name', 'Unknown Device')
        track_id_with_time = request.headers.get('X-Track-ID', '')

        assigned_dot_id = get_or_assign_dot_id(device_id, device_name)

        if assigned_dot_id is None:
            return jsonify({"error": "All DOT slots in use. Contact administrator."}), 503

        today = datetime.now().strftime('%Y%m%d')
        dot_directory = f"{assigned_dot_id}_{today}"

        if not track_id_with_time:
            logger.error("Missing X-Track-ID header")
            return jsonify({"error": "Missing X-Track-ID header"}), 400

        track_id, time_str = parse_track_id(track_id_with_time)

        if track_id is None:
            logger.error(f"Invalid X-Track-ID format: {track_id_with_time}")
            return jsonify({"error": "Invalid X-Track-ID format. Expected: {track_id}_{HHMMSS}"}), 400

        dot_directory = Path(dot_directory).name
        track_id = Path(track_id).name

        logger.info(f"Receiving crop upload from {device_name}")
        logger.info(f"DOT Directory: {dot_directory}, Track ID: {track_id}")

        try:
            dot_dir_path = input_storage / dot_directory
            crops_dir_path = dot_dir_path / "crops"

            existing = find_track_folder(crops_dir_path, track_id)
            if existing:
                track_folder = existing
                track_dir_path = crops_dir_path / track_folder
                logger.info(f"Merging into existing folder: {track_folder}")
            else:
                track_folder = f"{track_id}_{time_str}"
                track_dir_path = crops_dir_path / track_folder
                track_dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Creating new folder: {track_folder}")

            files = request.files.getlist('files')
            if not files:
                logger.warning("No files received in upload")
                return jsonify({"error": "No files uploaded"}), 400

            frame_offset = len([f for f in track_dir_path.iterdir()
                                if f.is_file() and f.name.startswith("frame_")])

            saved_count = 0
            for file in files:
                if not file.filename:
                    continue

                filename = Path(file.filename).name
                frame_num = extract_frame_number(filename)

                if frame_num is None:
                    logger.warning(f"Skipping invalid filename: {filename}")
                    continue

                target_filename = f"frame_{frame_offset + frame_num:06d}.jpg"
                full_path = track_dir_path / target_filename
                file.save(full_path)
                saved_count += 1

            total_frames = len([f for f in track_dir_path.iterdir() if f.is_file() and f.name.startswith("frame_")])
            logger.info(f"Saved {saved_count} frames for track {track_folder} (total: {total_frames})")

            if tracker:
                tracker.update(dot_directory, track_folder, has_crops=True)

            return jsonify({
                "status": "success",
                "dot_directory": dot_directory,
                "track_folder": track_folder,
                "frames_saved": saved_count
            }), 200

        except Exception as e:
            logger.error(f"Error saving upload: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/upload_labels', methods=['POST'])
    def upload_labels():
        device_id = request.headers.get('X-Device-ID', 'unknown')
        device_name = request.headers.get('X-Device-Name', 'Unknown Device')
        track_id_with_time = request.headers.get('X-Track-ID', '')

        assigned_dot_id = get_or_assign_dot_id(device_id, device_name)

        if assigned_dot_id is None:
            return jsonify({"error": "All DOT slots in use. Contact administrator."}), 503

        today = datetime.now().strftime('%Y%m%d')
        dot_directory = f"{assigned_dot_id}_{today}"

        if not track_id_with_time:
            logger.error("Missing X-Track-ID header")
            return jsonify({"error": "Missing X-Track-ID header"}), 400

        track_id, time_str = parse_track_id(track_id_with_time)

        if track_id is None:
            logger.error(f"Invalid X-Track-ID format: {track_id_with_time}")
            return jsonify({"error": "Invalid X-Track-ID format. Expected: {track_id}_{HHMMSS}"}), 400

        dot_directory = Path(dot_directory).name
        track_id = Path(track_id).name

        labels_data = request.get_json(silent=True)
        if not labels_data:
            logger.error("No JSON data received")
            return jsonify({"error": "No JSON data received"}), 400

        labels_data = convert_dot_labels_to_bugcam_format(labels_data)

        logger.info(f"Receiving labels upload from {device_name}")
        logger.info(f"DOT Directory: {dot_directory}, Track: {track_id}")

        try:
            dot_dir_path = input_storage / dot_directory
            labels_dir_path = dot_dir_path / "labels"
            labels_dir_path.mkdir(parents=True, exist_ok=True)

            labels_filename = f"{track_id}.json"
            labels_path = labels_dir_path / labels_filename

            if labels_path.exists():
                with open(labels_path, 'r') as f:
                    existing_data = json.load(f)
                existing_frames = existing_data.get("frames", [])
            else:
                existing_frames = []

            new_data = labels_data if isinstance(labels_data, dict) else {}
            new_frames = new_data.get("frames", [])

            merged_frames = existing_frames + new_frames
            for i, frame in enumerate(merged_frames):
                frame["frame_number"] = i

            merged_data = {"frames": merged_frames}

            with open(labels_path, 'w') as f:
                json.dump(merged_data, f, indent=2)

            logger.info(f"Saved labels for track {track_id} (total frames: {len(merged_frames)})")

            actual_folder = find_track_folder(
                input_storage / dot_directory / "crops", track_id
            ) or track_id
            if tracker:
                tracker.update(dot_directory, actual_folder, has_crops=False)

            return jsonify({
                "status": "success",
                "dot_directory": dot_directory,
                "track_id": track_id,
                "labels_file": labels_filename
            }), 200

        except Exception as e:
            logger.error(f"Error saving labels: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/upload_done', methods=['POST'])
    def upload_done():
        device_id = request.headers.get('X-Device-ID', 'unknown')
        device_name = request.headers.get('X-Device-Name', 'Unknown Device')
        track_id_with_time = request.headers.get('X-Track-ID', '')

        assigned_dot_id = get_or_assign_dot_id(device_id, device_name)

        if assigned_dot_id is None:
            return jsonify({"error": "All DOT slots in use. Contact administrator."}), 503

        today = datetime.now().strftime('%Y%m%d')
        dot_directory = f"{assigned_dot_id}_{today}"

        if not track_id_with_time:
            logger.error("Missing X-Track-ID header")
            return jsonify({"error": "Missing X-Track-ID header"}), 400

        track_id, time_str = parse_track_id(track_id_with_time)

        if track_id is None:
            logger.error(f"Invalid X-Track-ID format: {track_id_with_time}")
            return jsonify({"error": "Invalid X-Track-ID format. Expected: {track_id}_{HHMMSS}"}), 400

        dot_directory = Path(dot_directory).name

        logger.info(f"Creating done marker from {device_name}")
        logger.info(f"DOT Directory: {dot_directory}, Track ID: {track_id}")

        try:
            dot_dir_path = input_storage / dot_directory
            crops_dir_path = dot_dir_path / "crops"
            existing = find_track_folder(crops_dir_path, track_id)

            if not existing:
                logger.error(f"No crop folder found for track {track_id} in {crops_dir_path}")
                return jsonify({"error": "Track directory not found"}), 400

            track_folder = existing
            track_dir_path = crops_dir_path / track_folder

            done_file_path = track_dir_path / "done.txt"
            if done_file_path.exists():
                logger.info(f"done.txt already exists for track {track_folder}")
                return jsonify({
                    "status": "success",
                    "dot_directory": dot_directory,
                    "track_folder": track_folder,
                    "marker": "done.txt",
                    "note": "already existed"
                }), 200

            with open(done_file_path, 'w') as f:
                f.write(f"Track completed at {datetime.now().isoformat()}\n")

            if tracker:
                tracker.mark_done(dot_directory, track_folder)

            logger.info(f"Created done marker for track {track_folder}")

            return jsonify({
                "status": "success",
                "dot_directory": dot_directory,
                "track_folder": track_folder,
                "marker": "done.txt"
            }), 200

        except Exception as e:
            logger.error(f"Error creating done marker: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/upload_background', methods=['POST'])
    def upload_background():
        device_id = request.headers.get('X-Device-ID', 'unknown')
        device_name = request.headers.get('X-Device-Name', 'Unknown Device')

        assigned_dot_id = get_or_assign_dot_id(device_id, device_name)

        if assigned_dot_id is None:
            return jsonify({"error": "All DOT slots in use. Contact administrator."}), 503

        today = datetime.now().strftime('%Y%m%d')
        dot_directory = f"{assigned_dot_id}_{today}"

        if 'image' not in request.files:
            logger.error("No image file in request")
            return jsonify({"error": "No image file uploaded"}), 400

        image_file = request.files['image']
        if not image_file or not image_file.filename:
            logger.error("Empty image file")
            return jsonify({"error": "Empty image file"}), 400

        dot_directory = Path(dot_directory).name

        timestamp = datetime.now().strftime('%H%M%S')
        filename = f"{timestamp}_background.jpg"

        logger.info(f"Receiving background image from {device_name}")
        logger.info(f"DOT Directory: {dot_directory}, File: {filename}")

        try:
            dot_dir_path = input_storage / dot_directory
            dot_dir_path.mkdir(parents=True, exist_ok=True)

            image_path = dot_dir_path / filename
            image_file.save(image_path)

            current_path = dot_dir_path / "current_background.jpg"
            shutil.copy2(image_path, current_path)

            file_size = image_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)

            logger.info(f"Saved background image: {filename} ({file_size_mb:.1f} MB)")

            return jsonify({
                "status": "success",
                "dot_directory": dot_directory,
                "filename": filename,
                "size_bytes": file_size
            }), 200

        except Exception as e:
            logger.error(f"Error saving background image: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/upload_video', methods=['POST'])
    def upload_video():
        device_id = request.headers.get('X-Device-ID', 'unknown')
        device_name = request.headers.get('X-Device-Name', 'Unknown Device')

        assigned_dot_id = get_or_assign_dot_id(device_id, device_name)

        if assigned_dot_id is None:
            return jsonify({"error": "All DOT slots in use. Contact administrator."}), 503

        today = datetime.now().strftime('%Y%m%d')
        dot_directory = f"{assigned_dot_id}_{today}"

        if 'video' not in request.files:
            logger.error("No video file in request")
            return jsonify({"error": "No video file uploaded"}), 400

        video_file = request.files['video']
        if not video_file or not video_file.filename:
            logger.error("Empty video file")
            return jsonify({"error": "Empty video file"}), 400

        dot_directory = Path(dot_directory).name
        filename = Path(video_file.filename).name

        if not filename.lower().endswith('.mp4'):
            logger.error(f"Invalid file format: {filename}")
            return jsonify({"error": "Invalid file format. Expected: .mp4"}), 400

        logger.info(f"Receiving video upload from {device_name}")
        logger.info(f"DOT Directory: {dot_directory}, File: {filename}")

        try:
            dot_dir_path = input_storage / dot_directory
            videos_dir_path = dot_dir_path / "videos"
            videos_dir_path.mkdir(parents=True, exist_ok=True)

            video_path = videos_dir_path / filename
            video_file.save(video_path)

            file_size = video_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)

            logger.info(f"Saved video: {filename} ({file_size_mb:.1f} MB)")

            return jsonify({
                "status": "success",
                "dot_directory": dot_directory,
                "filename": filename,
                "size_bytes": file_size,
                "size_mb": round(file_size_mb, 2)
            }), 200

        except Exception as e:
            logger.error(f"Error saving video: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/heartbeat', methods=['GET', 'POST'])
    def heartbeat():
        device_id = request.headers.get('X-Device-ID', 'unknown')
        device_name = request.headers.get('X-Device-Name', 'Unknown')

        if request.method == 'GET':
            device_id = request.args.get('device_id', device_id)
            device_name = request.args.get('device_name', device_name)

        request.get_json(silent=True)  # tolerate/consume an optional JSON body
        safe_id = device_id[:8] if len(device_id) > 8 else device_id
        logger.info(f"Heartbeat from {device_name} ({safe_id})")

        return jsonify({"status": "ok", "server": "bugcam-receiver"}), 200

    @app.route('/api/health', methods=['GET'])
    def health():
        return jsonify({
            "status": "ok",
            "service": "bugcam-receiver",
            "input_storage": str(input_storage),
            "timestamp": datetime.now().isoformat()
        }), 200

    @app.route('/api/track', methods=['POST'])
    def api_track():
        device_id = request.headers.get('X-Device-ID', 'unknown')
        device_name = request.headers.get('X-Device-Name', 'Unknown Device')
        track_id_with_time = request.headers.get('X-Track-ID', '')

        assigned_dot_id = get_or_assign_dot_id(device_id, device_name)

        if assigned_dot_id is None:
            return jsonify({"error": "All DOT slots in use. Contact administrator."}), 503

        today = datetime.now().strftime('%Y%m%d')
        dot_directory = f"{assigned_dot_id}_{today}"

        track_data = request.get_json(silent=True)
        if not track_data:
            logger.error("No JSON data received")
            return jsonify({"error": "No JSON data received"}), 400

        track_data = convert_dot_labels_to_bugcam_format(track_data)

        track_id = track_id_with_time
        if not track_id:
            track_id = str(track_data.get('track_id', ''))

        if not track_id:
            logger.error("Missing track_id in header or data")
            return jsonify({"error": "Missing track_id"}), 400

        parsed_track_id, time_str = parse_track_id(track_id)
        if parsed_track_id is None:
            parsed_track_id = str(track_id)

        logger.info(f"Receiving track telemetry from {device_name} ({device_id[:8]}...)")
        logger.info(f"Assigned DOT Directory: {dot_directory}, Track: {parsed_track_id}")

        try:
            dot_dir_path = input_storage / dot_directory
            labels_dir_path = dot_dir_path / "labels"
            labels_dir_path.mkdir(parents=True, exist_ok=True)

            labels_filename = f"{parsed_track_id}.json"
            labels_path = labels_dir_path / labels_filename

            if labels_path.exists():
                with open(labels_path, 'r') as f:
                    existing_data = json.load(f)
                existing_frames = existing_data.get("frames", [])
            else:
                existing_frames = []

            new_data = track_data if isinstance(track_data, dict) else {}
            new_frames = new_data.get("frames", [])

            merged_frames = existing_frames + new_frames
            for i, frame in enumerate(merged_frames):
                frame["frame_number"] = i

            merged_data = {"frames": merged_frames}

            with open(labels_path, 'w') as f:
                json.dump(merged_data, f, indent=2)

            logger.info(f"Saved telemetry for track {parsed_track_id} (total frames: {len(merged_frames)})")

            actual_folder = find_track_folder(
                input_storage / dot_directory / "crops", parsed_track_id
            ) or parsed_track_id
            if tracker:
                tracker.update(dot_directory, actual_folder, has_crops=False)

            return jsonify({
                "status": "success",
                "dot_directory": dot_directory,
                "track_id": parsed_track_id
            }), 200

        except Exception as e:
            logger.error(f"Error saving telemetry: {e}")
            return jsonify({"error": str(e)}), 500
