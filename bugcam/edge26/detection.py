"""
Video processor for insect detection pipeline.

Preprocessing (background subtraction, blob extraction) and tracking are
BugSpot's own responsibility (DetectionPipeline); this module wraps that for
crop/composite output and adds Hailo classification + hierarchical
aggregation on top. Re-exports BugSpot's core types below for convenience.
"""

import cv2
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional

from bugspot import (
    DetectionPipeline,
    MotionDetector,
    Detection,
    InsectTracker,
    Track,
    analyze_path_topology,
    check_track_consistency,
)

from .classification import HailoClassifier, HierarchicalClassification

__all__ = [
    "VideoProcessor",
    "DetectionPipeline",
    "MotionDetector",
    "Detection",
    "InsectTracker",
    "Track",
    "analyze_path_topology",
    "check_track_consistency",
]

logger = logging.getLogger(__name__)

_NA_PREDICTION = {
    "family": "N/A",
    "genus": "N/A",
    "species": "N/A",
    "family_confidence": None,
    "genus_confidence": None,
    "species_confidence": None,
}


class VideoProcessor:
    """
    Processes video files for insect detection and classification.
    
    Pipeline:
        1-4. Detection, Tracking, Topology, Crops & Composites (BugSpot)
        5.   Classification (Hailo)
        6.   Hierarchical Aggregation
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.detection_config = config.get("detection", {})
        self.classification_config = config.get("classification", {})
        self.tracking_config = config.get("tracking", {})
        self.output_config = config.get("output", {})
        self.model_metadata = config.get("model", {})
        
        # Build bugspot config (merge detection + tracking params)
        bugspot_config = dict(self.detection_config)
        bugspot_config["max_lost_frames"] = self.tracking_config.get("max_lost_frames", 45)
        bugspot_config["tracker_w_dist"] = self.tracking_config.get("w_dist", 0.6)
        bugspot_config["tracker_w_area"] = self.tracking_config.get("w_area", 0.4)
        bugspot_config["tracker_cost_threshold"] = self.tracking_config.get("cost_threshold", 0.3)
        
        # Core pipeline (BugSpot)
        self._pipeline = DetectionPipeline(bugspot_config)
        
        # Pipeline toggles
        pipeline_config = config.get("pipeline", {})
        self.enable_classification = pipeline_config.get("enable_classification", True)
        self.continuous_tracking = pipeline_config.get("continuous_tracking", False)
        
        # Classifier (lazy loaded)
        self._classifier: Optional[HailoClassifier] = None
        
        classify_str = "detection + classification" if self.enable_classification else "detection only"
        tracking_str = "continuous" if self.continuous_tracking else "per-video"
        logger.info(f"VideoProcessor initialized ({classify_str}, tracking: {tracking_str})")
    
    def process_video(self, video_path: Path, output_dir: Path) -> Dict:
        """
        Process a single video file.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory for crops/composites output
                        (e.g. results_dir/flick01/20260204_100000/)
        
        Returns:
            Dict with processing results
        """
        logger.info(f"Processing video: {video_path.name}")
        
        # Determine output dirs within device/timestamp structure
        save_composites_dir = None
        if self.output_config.get("save_composites", True):
            save_composites_dir = str(output_dir / "composites")
        
        save_crops_dir = None
        if self.output_config.get("save_crops", True):
            save_crops_dir = str(output_dir / "crops")
        
        # =================================================================
        # PHASES 1-4: Detection, Tracking, Topology, Crops & Composites
        # =================================================================
        result = self._pipeline.process_video(
            str(video_path),
            extract_crops=True,
            render_composites=save_composites_dir is not None,
            save_crops_dir=save_crops_dir,
            save_composites_dir=save_composites_dir,
        )
        
        logger.info(f"  BugSpot: {len(result.confirmed_tracks)} confirmed / "
                    f"{len(result.track_paths)} total tracks")
        
        # =================================================================
        # PHASE 5: Classification (confirmed tracks only)
        # =================================================================
        track_classifications: Dict[str, List[HierarchicalClassification]] = {}
        
        if self.enable_classification:
            logger.info("Phase 5: Classification (confirmed tracks only)")
            
            if result.confirmed_tracks:
                if self._classifier is None:
                    self._classifier = HailoClassifier(self.classification_config)
                
                classified_count = 0
                for track_id, track in result.confirmed_tracks.items():
                    classifications = []
                    for frame_num, crop in track.crops:
                        classification = self._classifier.classify(crop)
                        classifications.append(classification)
                        
                        # Update detection data with classification
                        for det in result.all_detections:
                            if det["track_id"] == track_id and det["frame_number"] == frame_num:
                                det["family"] = classification.family
                                det["genus"] = classification.genus
                                det["species"] = classification.species
                                det["family_confidence"] = classification.family_confidence
                                det["genus_confidence"] = classification.genus_confidence
                                det["species_confidence"] = classification.species_confidence
                                det["family_probs"] = classification.family_probs
                                det["genus_probs"] = classification.genus_probs
                                det["species_probs"] = classification.species_probs
                                break
                        
                        classified_count += 1
                    
                    if classifications:
                        track_classifications[track_id] = classifications
                
                logger.info(f"  Classified {classified_count} detections from "
                           f"{len(track_classifications)} tracks")
            
            # =================================================================
            # PHASE 6: Hierarchical Aggregation
            # =================================================================
            logger.info("Phase 6: Hierarchical Aggregation")
            
            aggregated = self._hierarchical_aggregation(
                result, track_classifications
            )
        else:
            logger.info("Phase 5-6: Classification disabled, skipping")
            aggregated = self._detection_only_aggregation(result)
        
        # Build output
        video_timestamp = self._parse_timestamp(video_path.stem)
        
        output = self._build_output(
            video_path=video_path,
            video_timestamp=video_timestamp,
            pipeline_result=result,
            aggregated=aggregated,
        )
        
        logger.info(f"  Complete: {len(aggregated)} confirmed tracks")
        return output
    
    def classify_dot_track(self, track_dir: Path, track_id: str,
                           timestamp: Optional[str] = None) -> Optional[Dict]:
        """
        Classify crops from a single DOT track directory.
        
        Args:
            track_dir: Path to track directory containing frame_*.jpg crops
            track_id: Track identifier (hash only, e.g. "a1b2c3d4")
            timestamp: Track timestamp as HHMMSS string, or None
            
        Returns:
            Dict with track classification results, or None if no valid crops
        """
        if self._classifier is None:
            self._classifier = HailoClassifier(self.classification_config)
        
        crop_files = sorted(track_dir.glob("frame_*.jpg"))
        if not crop_files:
            return None
        
        classifications = []
        frames = []
        
        for crop_path in crop_files:
            crop = cv2.imread(str(crop_path))
            if crop is None:
                logger.warning(f"Could not read crop: {crop_path}")
                continue
            
            frame_num = int(crop_path.stem.split("_")[1])
            classification = self._classifier.classify(crop)
            classifications.append(classification)
            
            frames.append({
                "frame_number": frame_num,
                "prediction": {
                    "family": classification.family,
                    "genus": classification.genus,
                    "species": classification.species,
                    "family_confidence": classification.family_confidence,
                    "genus_confidence": classification.genus_confidence,
                    "species_confidence": classification.species_confidence,
                }
            })
        
        if not classifications:
            return None
        
        final_pred = self._classifier.hierarchical_aggregate(classifications)
        if not final_pred:
            return None
        
        result = {
            "track_id": track_id,
            "timestamp": timestamp,
            "final_prediction": final_pred,
            "num_detections": len(classifications),
            "frames": frames,
        }
        return result
    
    def create_dot_composite(self, track_dir: Path, background_path: Path,
                             label_path: Path, output_path: Path) -> None:
        """
        Create a composite image matching BugSpot's visual style.
        
        Dimmed background with lighten-blended crops at their bbox
        positions, red path polyline through centroids, green start
        marker, and detection count label.
        
        Args:
            track_dir: Track directory with frame_*.jpg crops
            background_path: Background image for this DOT day
            label_path: Label JSON with per-frame bounding boxes [x, y, w, h]
            output_path: Where to save the composite
        """
        import numpy as np
        
        BG_DARKEN = 0.50
        
        background = cv2.imread(str(background_path))
        if background is None:
            raise ValueError(f"Could not read background: {background_path}")
        
        composite = background.astype(np.float64) * BG_DARKEN
        bg_h, bg_w = background.shape[:2]
        
        # Load bounding boxes: {frame_number: [x, y, w, h]}
        bboxes = {}
        if label_path.exists():
            try:
                with open(label_path) as f:
                    label_data = json.load(f)
                if isinstance(label_data, dict):
                    for frame_info in label_data.get("frames", []):
                        fn = frame_info.get("frame_number")
                        bbox = frame_info.get("bbox")
                        if fn is not None and bbox:
                            bboxes[fn] = bbox
            except Exception:
                pass
        
        crop_files = sorted(track_dir.glob("frame_*.jpg"))
        centroids = []
        n_placed = 0
        
        for crop_path in crop_files:
            crop = cv2.imread(str(crop_path))
            if crop is None:
                continue
            
            frame_num = int(crop_path.stem.split("_")[1])
            
            if frame_num not in bboxes:
                continue
            
            bbox = bboxes[frame_num]
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            resized = cv2.resize(crop, (w, h))
            
            # Clip to image bounds
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(bg_w, x + w), min(bg_h, y + h)
            cx1, cy1 = x1 - x, y1 - y
            cx2, cy2 = cx1 + (x2 - x1), cy1 + (y2 - y1)
            
            if x2 > x1 and y2 > y1:
                region = resized[cy1:cy2, cx1:cx2].astype(np.float64)
                composite[y1:y2, x1:x2] = np.maximum(
                    composite[y1:y2, x1:x2], region
                )
                centroids.append((x + w // 2, y + h // 2))
                n_placed += 1
        
        img = np.clip(composite, 0, 255).astype(np.uint8)
        
        # Path polyline (red) and start marker (green)
        if len(centroids) > 1:
            pts = np.array(centroids, dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
            cv2.circle(img, (pts[0][0], pts[0][1]), 6, (0, 255, 0), -1)
        
        cv2.putText(img, f"{n_placed} detections", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img)
    
    def _hierarchical_aggregation(
        self,
        result,
        track_classifications: Dict[str, List[HierarchicalClassification]],
    ) -> List[Dict]:
        """Aggregate per-frame classifications using hierarchical selection."""
        results = []
        
        for track_id, track in result.confirmed_tracks.items():
            if track_id not in track_classifications:
                continue
            
            classifications = track_classifications[track_id]
            if not classifications:
                continue
            
            final_pred = self._classifier.hierarchical_aggregate(classifications)
            if not final_pred:
                continue
            
            entry = {
                "track_id": track_id,
                "num_detections": track.num_detections,
                "first_frame_time": track.first_frame_time,
                "last_frame_time": track.last_frame_time,
                "duration": track.duration,
                "final_family": final_pred["family"],
                "final_genus": final_pred["genus"],
                "final_species": final_pred["species"],
                "family_confidence": final_pred["family_confidence"],
                "genus_confidence": final_pred["genus_confidence"],
                "species_confidence": final_pred["species_confidence"],
                "passes_topology": True,
                **track.topology_metrics,
            }
            results.append(entry)
            
            logger.info(f"  Track {track_id[:8]}: {final_pred['family']} / "
                       f"{final_pred['genus']} / {final_pred['species']} "
                       f"({final_pred['species_confidence']:.1%})")
        
        return results
    
    def _detection_only_aggregation(self, result) -> List[Dict]:
        """Build aggregated entries for detection-only mode (no classification)."""
        results = []
        
        for track_id, track in result.confirmed_tracks.items():
            entry = {
                "track_id": track_id,
                "num_detections": track.num_detections,
                "first_frame_time": track.first_frame_time,
                "last_frame_time": track.last_frame_time,
                "duration": track.duration,
                "final_family": _NA_PREDICTION["family"],
                "final_genus": _NA_PREDICTION["genus"],
                "final_species": _NA_PREDICTION["species"],
                "family_confidence": _NA_PREDICTION["family_confidence"],
                "genus_confidence": _NA_PREDICTION["genus_confidence"],
                "species_confidence": _NA_PREDICTION["species_confidence"],
                "passes_topology": True,
                **track.topology_metrics,
            }
            results.append(entry)
            
            logger.info(f"  Track {track_id[:8]}: N/A / N/A / N/A (detection only)")
        
        return results
    
    def _build_output(self, video_path: Path, video_timestamp: Optional[datetime],
                      pipeline_result, aggregated: List[Dict]) -> Dict:
        """Build final JSON output structure."""
        source_device, date_str, time_str = self._parse_video_identity(video_path.stem)
        
        tracks_data = []
        
        for entry in aggregated:
            track_id = entry["track_id"]
            
            # Per-frame data
            track_dets = [d for d in pipeline_result.all_detections if d["track_id"] == track_id]
            frames = []
            for det in track_dets:
                frame_data = {
                    "frame_number": det["frame_number"],
                    "timestamp_seconds": det["frame_time_seconds"],
                    "bbox": det["bbox"],
                }
                if "species" in det:
                    frame_data["prediction"] = {
                        "family": det["family"],
                        "genus": det["genus"],
                        "species": det["species"],
                        "family_confidence": det["family_confidence"],
                        "genus_confidence": det["genus_confidence"],
                        "species_confidence": det["species_confidence"],
                    }
                elif not self.enable_classification:
                    frame_data["prediction"] = dict(_NA_PREDICTION)
                frames.append(frame_data)
            
            tracks_data.append({
                "track_id": track_id,
                "timestamp": time_str,
                "final_prediction": {
                    "family": entry["final_family"],
                    "genus": entry["final_genus"],
                    "species": entry["final_species"],
                    "family_confidence": entry["family_confidence"],
                    "genus_confidence": entry["genus_confidence"],
                    "species_confidence": entry["species_confidence"],
                },
                "num_detections": entry["num_detections"],
                "first_seen_seconds": entry["first_frame_time"],
                "last_seen_seconds": entry["last_frame_time"],
                "duration_seconds": entry["duration"],
                "topology_metrics": {
                    k: entry.get(k)
                    for k in ["net_displacement", "revisit_ratio",
                             "progression_ratio", "directional_variance"]
                },
                "frames": frames,
            })
        
        vi = pipeline_result.video_info
        return {
            "source_device": source_device,
            "date": date_str,
            "model_id": self.model_metadata.get("model_id"),
            "video_file": video_path.name,
            "video_timestamp": video_timestamp.isoformat() if video_timestamp else None,
            "processing_timestamp": datetime.now(timezone.utc).isoformat(),
            "video_info": {
                "fps": vi["fps"],
                "total_frames": vi["total_frames"],
                "duration_seconds": vi["duration"],
            },
            "summary": {
                "total_detections": len(pipeline_result.all_detections),
                "total_tracks": len(pipeline_result.track_paths),
                "confirmed_tracks": len(pipeline_result.confirmed_tracks),
                "unconfirmed_tracks": len(pipeline_result.track_paths) - len(pipeline_result.confirmed_tracks),
            },
            "tracks": tracks_data,
        }

    def _parse_video_identity(self, filename_stem: str) -> tuple[str, Optional[str], Optional[str]]:
        """Parse device id and timestamp parts from old and new chunk filenames."""
        parts = filename_stem.split("_")
        if len(parts) >= 4 and len(parts[-3]) == 8 and len(parts[-2]) == 6 and len(parts[-1]) == 6:
            return "_".join(parts[:-3]), parts[-3], parts[-2]
        if len(parts) >= 3 and len(parts[-2]) == 8 and len(parts[-1]) == 6:
            return "_".join(parts[:-2]), parts[-2], parts[-1]
        return filename_stem, None, None

    def _parse_timestamp(self, filename_stem: str) -> Optional[datetime]:
        """Parse timestamp from video filename (recorder stamps stems in UTC)."""
        try:
            _, date_str, time_str = self._parse_video_identity(filename_stem)
            if date_str and time_str:
                return datetime.strptime(
                    f"{date_str}_{time_str}", "%Y%m%d_%H%M%S"
                ).replace(tzinfo=timezone.utc)
        except (ValueError, IndexError):
            pass
        return None
    
    def clear_video_detections(self) -> None:
        """
        Clean up after processing a video.
        
        Continuous tracking: calls pipeline.clear() — keeps tracker state so
            tracks can persist across video chunk boundaries.
        Per-video tracking: calls pipeline.reset() — full reset including
            tracker, so each video is processed independently.
        """
        if self.continuous_tracking:
            self._pipeline.clear()
        else:
            self._pipeline.reset()
    
    def reset_tracker(self) -> None:
        """Full reset of the BugSpot pipeline including tracker state."""
        self._pipeline.reset()
        logger.info("Tracker reset (full)")
