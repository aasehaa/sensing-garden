"""
Output writer for detection results.

Saves JSON results to device/timestamp-organised output directories.

Output structure:
    results_dir/
        {device_id}/
            {date_time}/
                crops/         - Crop images per track (saved by BugSpot or copied from DOT)
                composites/    - Composite images (saved by BugSpot or copied from DOT)
                results.json   - Classification results
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timezone

from bugcam.edge26.sidecars import RESULTS, RESULTS_TMP

logger = logging.getLogger(__name__)


class ResultsWriter:
    """
    Writes detection results to JSON.
    
    Output JSON format includes hierarchical predictions
    (family, genus, species) and per-frame data.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the results writer.
        
        Args:
            config: Output configuration from settings.yaml
        """
        self.config = config
        self.results_dir = Path(config.get("results_dir", "output/results"))
        
        # Ensure root output directory exists
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ResultsWriter initialized: {self.results_dir}")
    
    def write_results(self, results: Dict, output_dir: Path) -> Dict[str, Path]:
        """
        Write results JSON to an output directory.
        
        Uses atomic write (temp file + rename) to prevent data loss
        if the process crashes or the Pi loses power mid-write.
        
        Args:
            results: Processing results from VideoProcessor
            output_dir: Target directory (e.g. results_dir/flick01/20260204_100000/)
            
        Returns:
            Dict with output file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = {}
        
        json_path = output_dir / RESULTS
        temp_path = output_dir / RESULTS_TMP
        
        with open(temp_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        temp_path.replace(json_path)
        
        output_paths['json'] = json_path
        
        logger.info(f"Results saved: {json_path}")
        
        # Log summary
        summary = results.get('summary', {})
        logger.info(f"  Total tracks: {summary.get('total_tracks', 0)}, "
                   f"Confirmed: {summary.get('confirmed_tracks', 0)}")
        
        return output_paths
    
    def write_summary(self, all_results: List[Dict]) -> Path:
        """
        Write a summary of all processed videos.
        
        Args:
            all_results: List of results from multiple videos
            
        Returns:
            Path to summary JSON
        """
        total_confirmed = 0
        total_tracks = 0
        species_counts = {}
        
        for result in all_results:
            summary = result.get('summary', {})
            total_confirmed += summary.get('confirmed_tracks', 0)
            total_tracks += summary.get('total_tracks', 0)
            
            for track in result.get('tracks', []):
                final_pred = track.get('final_prediction', {})
                species = final_pred.get('species', 'Unknown')
                species_counts[species] = species_counts.get(species, 0) + 1
        
        summary_data = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'total_videos': len(all_results),
            'total_tracks': total_tracks,
            'total_confirmed': total_confirmed,
            'species_counts': species_counts,
            'videos': [
                {
                    'video_file': r.get('video_file'),
                    'video_timestamp': r.get('video_timestamp'),
                    'confirmed_tracks': r.get('summary', {}).get('confirmed_tracks', 0),
                    'species_detected': [
                        t.get('final_prediction', {}).get('species')
                        for t in r.get('tracks', [])
                    ]
                }
                for r in all_results
            ]
        }
        
        summary_path = self.results_dir / 'processing_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"Summary saved: {summary_path}")
        return summary_path
