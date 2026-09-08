"""Shared bundle resolution for BugCam models."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET
import urllib.request

from .settings import get_cache_dir

MODELS_BASE_URL = "https://scl-sensing-garden-models.s3.amazonaws.com"
BUNDLE_MODEL_FILENAME = "model.hef"
BUNDLE_LABELS_FILENAME = "labels.txt"
LOCAL_BUNDLES_DIR = Path(__file__).parent.parent / "resources"


@dataclass(frozen=True)
class ModelBundle:
    """Bundle-backed model install."""

    name: str
    root: Path
    location: str

    @property
    def model_path(self) -> Path:
        return self.root / BUNDLE_MODEL_FILENAME

    @property
    def labels_path(self) -> Path:
        return self.root / BUNDLE_LABELS_FILENAME

    @property
    def has_model(self) -> bool:
        return self.model_path.exists()

    @property
    def has_labels(self) -> bool:
        return self.labels_path.exists()

    def is_complete(self, require_labels: bool = True) -> bool:
        return self.has_model and (self.has_labels if require_labels else True)


def get_models_cache_dir() -> Path:
    """Return the models cache directory."""
    return get_cache_dir() / "models"


def get_bundle_dir(bundle_name: str, cache_dir: Optional[Path] = None) -> Path:
    """Return the bundle directory path."""
    return (cache_dir or get_models_cache_dir()) / bundle_name


def iter_bundle_roots(cache_dir: Optional[Path] = None, local_dir: Optional[Path] = None) -> list[tuple[str, Path]]:
    """Return search roots for installed model bundles."""
    return [
        ("cache", cache_dir or get_models_cache_dir()),
        ("local", local_dir or LOCAL_BUNDLES_DIR),
    ]


def get_installed_bundles(
    require_labels: bool = False,
    cache_dir: Optional[Path] = None,
    local_dir: Optional[Path] = None,
) -> list[ModelBundle]:
    """Return installed bundle directories."""
    bundles: list[ModelBundle] = []
    seen_names: set[str] = set()

    for location, root in iter_bundle_roots(cache_dir=cache_dir, local_dir=local_dir):
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            bundle = ModelBundle(name=child.name, root=child, location=location)
            if bundle.name in seen_names:
                continue
            if bundle.is_complete(require_labels=require_labels):
                bundles.append(bundle)
                seen_names.add(bundle.name)
    return bundles


def find_installed_bundle(
    bundle_name: str,
    require_labels: bool = False,
    cache_dir: Optional[Path] = None,
    local_dir: Optional[Path] = None,
) -> Optional[ModelBundle]:
    """Find an installed bundle by name."""
    for bundle in get_installed_bundles(
        require_labels=require_labels,
        cache_dir=cache_dir,
        local_dir=local_dir,
    ):
        if bundle.name == bundle_name:
            return bundle
    return None


def resolve_bundle_reference(
    reference: Optional[str],
    require_labels: bool = False,
    cache_dir: Optional[Path] = None,
    local_dir: Optional[Path] = None,
) -> Optional[ModelBundle]:
    """Resolve a bundle reference from name or directory path."""
    if reference is None:
        bundles = get_installed_bundles(
            require_labels=require_labels,
            cache_dir=cache_dir,
            local_dir=local_dir,
        )
        return bundles[0] if bundles else None

    reference_path = Path(reference)
    if reference_path.exists() and reference_path.is_dir():
        bundle = ModelBundle(name=reference_path.name, root=reference_path, location="path")
        if bundle.is_complete(require_labels=require_labels):
            return bundle
        return None

    return find_installed_bundle(
        reference,
        require_labels=require_labels,
        cache_dir=cache_dir,
        local_dir=local_dir,
    )


def resolve_model_path(reference: Optional[str]) -> Optional[Path]:
    """Resolve a path to a model.hef from a bundle reference or explicit path."""
    if reference:
        reference_path = Path(reference)
        if reference_path.exists() and reference_path.is_file() and reference_path.suffix == ".hef":
            return reference_path

    bundle = resolve_bundle_reference(reference, require_labels=False)
    if bundle:
        return bundle.model_path
    return None


def resolve_labels_path(reference: Optional[str]) -> Optional[Path]:
    """Resolve a labels.txt path from a bundle reference."""
    bundle = resolve_bundle_reference(reference, require_labels=True)
    if bundle:
        return bundle.labels_path
    return None


def get_remote_bundle_file_url(bundle_name: str, filename: str) -> str:
    """Build the URL for a file inside a remote model bundle."""
    return f"{MODELS_BASE_URL}/{bundle_name}/{filename}"


def sha256_file(path: Path) -> str:
    """Return the SHA256 for a local file."""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def list_remote_bundle_names() -> list[str]:
    """List remote bundles by scanning for */model.hef keys."""
    try:
        req = urllib.request.Request(MODELS_BASE_URL)
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()

        root = ET.fromstring(xml_data)
        namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        contents = root.findall(".//s3:Contents/s3:Key", namespace)
        if not contents:
            contents = root.findall(".//Contents/Key")

        bundles = set()
        suffix = f"/{BUNDLE_MODEL_FILENAME}"
        for key_elem in contents:
            key = key_elem.text or ""
            if key.endswith(suffix):
                bundles.add(key[: -len(suffix)])
        return sorted(bundles)
    except Exception:
        return []
