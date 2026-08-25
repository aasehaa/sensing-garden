"""Process existing media with the vendored edge26 pipeline."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from bugcam import __version__
from bugcam.config import get_input_storage_dir, get_output_storage_dir, log_startup_config
from bugcam.device_config import load_device_config, resolve_flick_id
from bugcam.runtime import build_pipeline, resolve_bundle_provenance

app = typer.Typer(help="Process existing files with edge26", invoke_without_command=True, no_args_is_help=False)
console = Console()


@app.callback()
def process(
    input_dir: Path = typer.Option(get_input_storage_dir(), "--input-dir", help="Directory containing input videos and DOT folders"),
    output_dir: Path = typer.Option(get_output_storage_dir(), "--output-dir", help="Directory for processed output"),
    model: str = typer.Option(..., "--model", help="Model bundle name or model.hef path"),
    flick_id: str | None = typer.Option(None, "--flick-id", help="FLICK device ID"),
    classification: bool = typer.Option(True, "--classification/--no-classification", help="Enable classification"),
    continuous_tracking: bool = typer.Option(False, "--continuous-tracking/--no-continuous-tracking", help="Track insects across FLICK chunks"),
    detection_config: Path | None = typer.Option(None, "--detection-config", help="Path to detection config YAML file"),
) -> None:
    """Process existing files without recording."""
    device_config = load_device_config()
    resolved_flick_id = resolve_flick_id(flick_id)
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = resolve_bundle_provenance(model)
    console.print(f"[cyan]Processing[/cyan] {input_dir} -> {output_dir}")
    console.print(f"[dim]Model[/dim] {provenance['model_id']} {provenance['model_sha256'][:12]}")
    pipeline = build_pipeline(
        flick_id=resolved_flick_id,
        dot_ids=device_config.dot_ids or [],
        input_dir=input_dir,
        output_dir=output_dir,
        model_reference=model,
        enable_recording=False,
        enable_processing=True,
        enable_classification=classification,
        continuous_tracking=continuous_tracking,
        detection_config_path=detection_config,
    )
    # setup_logging() ran inside build_pipeline(); handlers are live here.
    # Built from attributes (not dataclasses.asdict) so it also accepts a
    # mocked device_config in tests, not just a real DeviceConfig instance.
    log_startup_config(
        version=__version__,
        device_config={
            "api_url": device_config.api_url,
            "api_key": device_config.api_key,
            "s3_bucket": device_config.s3_bucket,
            "flick_id": resolved_flick_id,
            "dot_ids": device_config.dot_ids,
        },
        detection_config=pipeline.config.get("detection", {}),
        tracking_config=pipeline.config.get("tracking", {}),
        detection_config_source=pipeline.config.get("detection_config_source", "unknown"),
    )
    pipeline.start()
    pipeline.wait()
