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
from typing import Dict, Optional

from bugspot import (
    DetectionPipeline,
    MotionDetector,
    Detection,
    InsectTracker,
    Track,
    analyze_path_topology,
    check_track_consistency,
)

from .classification import HailoClassifier

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
