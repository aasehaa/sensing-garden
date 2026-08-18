"""
Hierarchical insect classification.

Classifies insects at 3 levels: Family, Genus, Species.
Uses Hailo HEF models for inference via the VStreams API.
Taxonomy (family/genus) is resolved from GBIF at startup.
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests

try:
    from hailo_platform import (
        HEF,
        ConfigureParams,
        FormatType,
        HailoSchedulingAlgorithm,
        HailoStreamInterface,
        InferVStreams,
        InputVStreamParams,
        OutputVStreamParams,
        VDevice,
    )
    _HAILO_IMPORT_ERROR: ImportError | None = None
except ImportError as e:  # hailo_platform is a Pi-only hardware dependency
    # Defer the failure: keep the module importable (so `bugcam --help`, the test
    # suite, and any non-inference code path work off-Pi) and raise only when the
    # classifier actually loads a model. See _load_model.
    HEF = ConfigureParams = FormatType = HailoSchedulingAlgorithm = None
    HailoStreamInterface = InferVStreams = InputVStreamParams = None
    OutputVStreamParams = VDevice = None
    _HAILO_IMPORT_ERROR = e

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HierarchicalClassification:
    """Classification result with family, genus, and species predictions."""

    family: str
    genus: str
    species: str
    family_confidence: float
    genus_confidence: float
    species_confidence: float
    family_probs: List[float] = field(default_factory=list)
    genus_probs: List[float] = field(default_factory=list)
    species_probs: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Taxonomy helper
# ---------------------------------------------------------------------------

def get_taxonomy(species_list: List[str], cache_path: Path | None = None) -> dict:
    """
    Build a hierarchical taxonomy from the GBIF API.

    Args:
        species_list: Ordered list of species names (same order as training).

    Returns:
        Dictionary with three keys:
            1 – sorted list of families
            2 – genus -> family mapping
            3 – species -> genus mapping
    """
    cached_taxonomy = _load_taxonomy_cache(cache_path, species_list)
    if cached_taxonomy is not None:
        logger.info(f"Loaded taxonomy cache from {cache_path}")
        return cached_taxonomy

    taxonomy: dict = {1: [], 2: {}, 3: {}}
    lookup_failed = False

    species_for_gbif = [s for s in species_list if s.lower() != "unknown"]
    has_unknown = len(species_for_gbif) != len(species_list)

    logger.info(f"Building taxonomy from GBIF for {len(species_for_gbif)} species")

    print("\nTaxonomy Results:")
    print("-" * 80)
    print(f"{'Species':<30} {'Family':<20} {'Genus':<20} {'Status'}")
    print("-" * 80)

    for species_name in species_for_gbif:
        family, genus, ok = _lookup_species(species_name)
        lookup_failed = lookup_failed or not ok
        taxonomy[3][species_name] = genus
        taxonomy[2][genus] = family
        if family not in taxonomy[1]:
            taxonomy[1].append(family)

    # Handle the "unknown" class without hitting GBIF
    if has_unknown:
        taxonomy[1].append("Unknown")
        taxonomy[2]["Unknown"] = "Unknown"
        taxonomy[3]["unknown"] = "Unknown"
        print(f"{'unknown':<30} {'Unknown':<20} {'Unknown':<20} OK")

    taxonomy[1] = sorted(set(taxonomy[1]))
    print("-" * 80)

    _print_taxonomy_summary(taxonomy, species_list)
    if not lookup_failed:
        _save_taxonomy_cache(cache_path, taxonomy, species_list)
    return taxonomy


def _lookup_species(species_name: str) -> Tuple[str, str, bool]:
    """Query GBIF for a single species and return (family, genus, ok)."""
    url = f"https://api.gbif.org/v1/species/match?name={species_name}&verbose=true"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
    except Exception as exc:
        return _taxonomy_fallback(species_name, f"GBIF request failed: {exc}")

    status = data.get("status")
    if status not in ("ACCEPTED", "SYNONYM"):
        return _taxonomy_fallback(
            species_name,
            f"Not found in GBIF (status={status}), check spelling",
        )

    family = data.get("family")
    genus = data.get("genus")
    if not family or not genus:
        return _taxonomy_fallback(
            species_name,
            "Found in GBIF but family/genus missing, check spelling",
        )

    print(f"{species_name:<30} {family:<20} {genus:<20} OK")
    return family, genus, True


def _taxonomy_fallback(species_name: str, message: str) -> Tuple[str, str, bool]:
    """Log a taxonomy lookup warning and fall back to Unknown labels."""
    logger.warning(f"{species_name}: {message}")
    print(f"{species_name:<30} {'Unknown':<20} {'Unknown':<20} FALLBACK")
    print(f"  {message}")
    return "Unknown", "Unknown", False


def _load_taxonomy_cache(cache_path: Path | None, species_list: List[str]) -> Optional[dict]:
    if cache_path is None or not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Could not read taxonomy cache {cache_path}: {exc}")
        return None

    if payload.get("species_list") != species_list:
        logger.info("Taxonomy cache does not match current labels; rebuilding")
        return None

    families = payload.get("family_list")
    genus_to_family = payload.get("genus_to_family")
    species_to_genus = payload.get("species_to_genus")
    if not isinstance(families, list) or not isinstance(genus_to_family, dict) or not isinstance(species_to_genus, dict):
        logger.warning(f"Invalid taxonomy cache structure in {cache_path}")
        return None

    return {
        1: [str(family) for family in families],
        2: {str(genus): str(family) for genus, family in genus_to_family.items()},
        3: {str(species): str(genus) for species, genus in species_to_genus.items()},
    }


def _save_taxonomy_cache(cache_path: Path | None, taxonomy: dict, species_list: List[str]) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "species_list": species_list,
        "family_list": taxonomy[1],
        "genus_to_family": taxonomy[2],
        "species_to_genus": taxonomy[3],
    }
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _print_taxonomy_summary(taxonomy: dict, species_list: List[str]) -> None:
    """Print a human-readable taxonomy summary."""
    print("\nFamily indices:")
    for i, family in enumerate(taxonomy[1]):
        print(f"  {i}: {family}")

    print("\nGenus indices:")
    for i, genus in enumerate(sorted(taxonomy[2].keys())):
        print(f"  {i}: {genus}")

    print("\nSpecies indices:")
    for i, species in enumerate(species_list):
        print(f"  {i}: {species}")

    logger.info(
        f"Taxonomy built: {len(taxonomy[1])} families, "
        f"{len(taxonomy[2])} genera, {len(taxonomy[3])} species"
    )


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class HailoClassifier:
    """
    Hailo-based hierarchical insect classifier.

    Uses the VStreams API to run inference on a compiled HEF model.
    Outputs predictions for family, genus, and species.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.model_path = Path(config["model"])
        self._input_size: Optional[list] = config.get("input_size")

        # Hailo components (lazy-loaded on first inference)
        self._hef: Optional[HEF] = None
        self._vdevice: Optional[VDevice] = None
        self._network_group = None
        self._network_group_params = None
        self._input_vstream_params = None
        self._output_vstream_params = None

        # Labels & taxonomy (populated by _load_labels)
        self.family_list: List[str] = []
        self.genus_list: List[str] = []
        self.species_list: List[str] = []
        self.species_to_genus: Dict[str, str] = {}
        self.genus_to_family: Dict[str, str] = {}

        logger.info(f"HailoClassifier initialised – model: {self.model_path}")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the HEF, configure the device, and build the taxonomy."""
        if _HAILO_IMPORT_ERROR is not None:
            raise ImportError(
                "hailo_platform module not found. This is a system-level dependency for Hailo AI accelerators. "
                "On Raspberry Pi OS, install it with: sudo apt install python3-hailo-tappas. "
                "For other platforms, follow Hailo's installation guide: https://hailo.ai/developer-zone/"
            ) from _HAILO_IMPORT_ERROR
        if self._hef is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Load HEF
        self._hef = HEF(str(self.model_path))

        # Create virtual device
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.NONE
        self._vdevice = VDevice(params=params)

        # Configure network group
        configure_params = ConfigureParams.create_from_hef(
            hef=self._hef, interface=HailoStreamInterface.PCIe,
        )
        network_groups = self._vdevice.configure(self._hef, configure_params)
        self._network_group = network_groups[0]
        self._network_group_params = self._network_group.create_params()

        # VStream params – dequantised float32 in/out
        self._input_vstream_params = InputVStreamParams.make(
            self._network_group, quantized=False, format_type=FormatType.FLOAT32,
        )
        self._output_vstream_params = OutputVStreamParams.make(
            self._network_group, quantized=False, format_type=FormatType.FLOAT32,
        )

        # Labels & taxonomy
        self._load_labels()

        logger.info(
            f"Model loaded: {len(self.family_list)} families, "
            f"{len(self.genus_list)} genera, {len(self.species_list)} species"
        )

    # ------------------------------------------------------------------
    # Label / taxonomy loading
    # ------------------------------------------------------------------

    def _load_labels(self) -> None:
        """Load species from a plain-text file and build taxonomy via GBIF."""
        labels_path = self.config.get("labels")
        taxonomy_cache_path = self.config.get("taxonomy_cache")

        if not labels_path:
            logger.warning("No 'labels' in classification config – using numeric indices")
            self._load_labels_fallback()
            return

        labels_path = Path(labels_path)
        if not labels_path.exists():
            logger.warning(f"Labels file not found: {labels_path} – using numeric indices")
            self._load_labels_fallback()
            return

        # One species per line, same order as training
        with open(labels_path) as fh:
            self.species_list = [line.strip() for line in fh if line.strip()]

        logger.info(f"Loaded {len(self.species_list)} species from {labels_path}")

        # Resolve family / genus from GBIF
        taxonomy = get_taxonomy(
            self.species_list,
            Path(str(taxonomy_cache_path)) if taxonomy_cache_path else None,
        )
        self.family_list = taxonomy[1]
        self.genus_list = sorted(taxonomy[2].keys())
        self.species_to_genus = taxonomy[3]
        self.genus_to_family = taxonomy[2]

    def _load_labels_fallback(self) -> None:
        """Generate numeric placeholder labels from the model output shapes."""
        output_infos = self._hef.get_output_vstream_infos()

        for i, info in enumerate(output_infos):
            n = info.shape[-1]
            if i == 0:
                self.family_list = [f"family_{j}" for j in range(n)]
            elif i == 1:
                self.genus_list = [f"genus_{j}" for j in range(n)]
            else:
                self.species_list = [f"class_{j}" for j in range(n)]

        # Single-head model → treat as species-only
        if len(output_infos) == 1:
            n = output_infos[0].shape[-1]
            self.species_list = [f"class_{j}" for j in range(n)]
            self.family_list = list(self.species_list)
            self.genus_list = list(self.species_list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, crop: np.ndarray) -> HierarchicalClassification:
        """
        Classify a single BGR image crop.

        Returns a HierarchicalClassification with family/genus/species
        predictions and their softmax probabilities.
        """
        self._load_model()

        preprocessed = self._preprocess(crop)
        raw_outputs = self._run_inference(preprocessed)

        family_probs, genus_probs, species_probs = self._parse_outputs(raw_outputs)

        family_idx = int(np.argmax(family_probs))
        genus_idx = int(np.argmax(genus_probs))
        species_idx = int(np.argmax(species_probs))

        return HierarchicalClassification(
            family=self._safe_label(self.family_list, family_idx, "Family"),
            genus=self._safe_label(self.genus_list, genus_idx, "Genus"),
            species=self._safe_label(self.species_list, species_idx, "Species"),
            family_confidence=float(family_probs[family_idx]),
            genus_confidence=float(genus_probs[genus_idx]),
            species_confidence=float(species_probs[species_idx]),
            family_probs=family_probs.tolist(),
            genus_probs=genus_probs.tolist(),
            species_probs=species_probs.tolist(),
        )

    def classify_batch(self, crops: List[np.ndarray]) -> List[HierarchicalClassification]:
        """Classify a list of BGR image crops."""
        return [self.classify(crop) for crop in crops]

    def hierarchical_aggregate(
        self, classifications: List[HierarchicalClassification],
    ) -> Optional[Dict]:
        """
        Aggregate per-frame classifications using hierarchical selection.

        1. Average probabilities across frames.
        2. Pick the best family.
        3. Pick the best genus *within* that family.
        4. Pick the best species *within* that genus.
        """
        if not classifications:
            return None

        family_avg = np.mean([c.family_probs for c in classifications], axis=0)
        genus_avg = np.mean([c.genus_probs for c in classifications], axis=0)
        species_avg = np.mean([c.species_probs for c in classifications], axis=0)

        # Best family
        family_idx = int(np.argmax(family_avg))
        best_family = self._safe_label(self.family_list, family_idx, "Family")

        # Best genus within that family
        genus_idx = self._best_within(
            self.genus_list, self.genus_to_family, best_family, genus_avg,
        )
        best_genus = self._safe_label(self.genus_list, genus_idx, "Genus")

        # Best species within that genus
        species_idx = self._best_within(
            self.species_list, self.species_to_genus, best_genus, species_avg,
        )
        best_species = self._safe_label(self.species_list, species_idx, "Species")

        return {
            "family": best_family,
            "genus": best_genus,
            "species": best_species,
            "family_confidence": float(family_avg[family_idx]),
            "genus_confidence": float(genus_avg[genus_idx]),
            "species_confidence": float(species_avg[species_idx]),
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_families(self) -> int:
        self._load_model()
        return len(self.family_list)

    @property
    def num_genera(self) -> int:
        self._load_model()
        return len(self.genus_list)

    @property
    def num_species(self) -> int:
        self._load_model()
        return len(self.species_list)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Resize a BGR crop to the model's input size and return float32 RGB."""
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        h, w = self._get_input_hw()
        resized = cv2.resize(rgb, (w, h))

        return resized.astype(np.float32)

    def _get_input_hw(self) -> Tuple[int, int]:
        """Return the model's expected (height, width)."""
        if self._input_size and len(self._input_size) >= 2:
            return int(self._input_size[0]), int(self._input_size[1])

        self._load_model()
        shape = self._hef.get_input_vstream_infos()[0].shape  # (H, W, C)
        return int(shape[0]), int(shape[1])

    def _run_inference(self, preprocessed: np.ndarray) -> List[np.ndarray]:
        """Run a single forward pass through the Hailo VStreams pipeline."""
        input_info = self._hef.get_input_vstream_infos()[0]
        output_infos = self._hef.get_output_vstream_infos()

        # Batch dimension required by InferVStreams
        input_data = {input_info.name: np.expand_dims(preprocessed, axis=0)}

        with InferVStreams(
            self._network_group,
            self._input_vstream_params,
            self._output_vstream_params,
        ) as pipeline:
            with self._network_group.activate(self._network_group_params):
                results = pipeline.infer(input_data)

        # Strip batch dimension and ensure float32
        return [results[info.name][0].astype(np.float32) for info in output_infos]

    def _parse_outputs(
        self, outputs: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert raw model outputs into softmax probabilities."""
        if len(outputs) >= 3:
            family_logits = outputs[0].flatten()
            genus_logits = outputs[1].flatten()
            species_logits = outputs[2].flatten()
        else:
            species_logits = outputs[0].flatten()
            family_logits = np.zeros(len(self.family_list)) if self.family_list else species_logits
            genus_logits = np.zeros(len(self.genus_list)) if self.genus_list else species_logits

        return (
            self._softmax(family_logits),
            self._softmax(genus_logits),
            self._softmax(species_logits),
        )

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically-stable softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    @staticmethod
    def _safe_label(labels: List[str], idx: int, prefix: str) -> str:
        """Return the label at *idx*, or a fallback string."""
        return labels[idx] if idx < len(labels) else f"{prefix}_{idx}"

    @staticmethod
    def _best_within(
        labels: List[str],
        mapping: Dict[str, str],
        parent: str,
        probs: np.ndarray,
    ) -> int:
        """Pick the highest-probability index whose label maps to *parent*."""
        candidates = [i for i, lbl in enumerate(labels) if mapping.get(lbl) == parent]
        if candidates:
            return candidates[int(np.argmax(probs[candidates]))]
        return int(np.argmax(probs))
