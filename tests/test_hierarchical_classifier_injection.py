"""Pins the ModelRunner split (interfaces.py): HierarchicalClassifier must
run end-to-end -- preprocessing, output parsing, taxonomy fallback, and
hierarchical aggregation -- against any ModelRunner, with no Hailo
dependency at all. HailoClassifier is just HierarchicalClassifier bound to
HailoModelRunner, not a separate implementation."""
import numpy as np

from bugcam.edge26.classification import HailoClassifier, HailoModelRunner, HierarchicalClassifier


class _FakeModelRunner:
    """3-head runner (family/genus/species) with fixed, deterministic
    logits -- no model file, no hardware, no GBIF network calls needed
    since labels stay as numeric fallback."""

    def __init__(self) -> None:
        self.load_calls = 0

    def load(self) -> None:
        self.load_calls += 1

    def input_hw(self):
        return (4, 4)

    def output_head_sizes(self):
        return [2, 2, 3]  # family, genus, species head sizes

    def run(self, preprocessed):
        return [
            np.array([0.1, 5.0]),        # family: index 1 wins
            np.array([3.0, 0.2]),        # genus: index 0 wins
            np.array([0.1, 0.1, 4.0]),   # species: index 2 wins
        ]


def _crop() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


def test_classify_runs_end_to_end_against_a_fake_runner():
    classifier = HierarchicalClassifier({}, runner_factory=_FakeModelRunner)

    result = classifier.classify(_crop())

    assert result.family == "family_1"
    assert result.genus == "genus_0"
    assert result.species == "class_2"
    assert 0.0 < result.family_confidence <= 1.0


def test_runner_is_constructed_once_and_loaded_once():
    runner = _FakeModelRunner()
    classifier = HierarchicalClassifier({}, runner_factory=lambda: runner)

    classifier.classify(_crop())
    classifier.classify(_crop())

    assert runner.load_calls == 1


def test_hierarchical_aggregate_picks_consistent_taxonomy_path():
    classifier = HierarchicalClassifier({}, runner_factory=_FakeModelRunner)

    classifications = [classifier.classify(_crop()) for _ in range(3)]
    aggregated = classifier.hierarchical_aggregate(classifications)

    assert aggregated == {
        "family": "family_1",
        "genus": "genus_0",
        "species": "class_2",
        "family_confidence": aggregated["family_confidence"],
        "genus_confidence": aggregated["genus_confidence"],
        "species_confidence": aggregated["species_confidence"],
    }


def test_hailo_classifier_is_hierarchical_classifier_bound_to_hailo_runner():
    classifier = HailoClassifier({"model": "some/model.hef"})

    assert isinstance(classifier, HierarchicalClassifier)
    runner = classifier._runner_factory()
    assert isinstance(runner, HailoModelRunner)
    assert runner.model_path.name == "model.hef"
