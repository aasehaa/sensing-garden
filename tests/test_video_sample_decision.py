"""Sample-video selection for the FLIK upload path.

Default behavior prioritizes the first video in a batch of N with confirmed
tracks over a later plain one (falling back to the batch's last video if none
had detections), so the uploaded "sample" is really "best of N". --random-sampling
disables that bias so the saved sample is always the batch's actual Nth video,
regardless of content.
"""
from bugcam.edge26.pipeline import _video_sample_decision


# --- default mode (random_sampling=False): confirmed tracks win ---

def test_default_saves_on_confirmed_tracks_before_batch_end():
    should_save, reason = _video_sample_decision(
        sample_saved=False, is_last_in_batch=False, confirmed_count=2, random_sampling=False,
    )
    assert (should_save, reason) == (True, "detections")


def test_default_does_not_save_mid_batch_without_confirmed_tracks():
    should_save, reason = _video_sample_decision(
        sample_saved=False, is_last_in_batch=False, confirmed_count=0, random_sampling=False,
    )
    assert (should_save, reason) == (False, None)


def test_default_falls_back_to_last_video_when_no_confirmed_tracks_all_batch():
    should_save, reason = _video_sample_decision(
        sample_saved=False, is_last_in_batch=True, confirmed_count=0, random_sampling=False,
    )
    assert (should_save, reason) == (True, "fallback")


def test_default_last_video_with_confirmed_tracks_reports_detections_not_fallback():
    should_save, reason = _video_sample_decision(
        sample_saved=False, is_last_in_batch=True, confirmed_count=1, random_sampling=False,
    )
    assert (should_save, reason) == (True, "detections")


def test_default_does_not_save_twice_in_one_batch():
    should_save, reason = _video_sample_decision(
        sample_saved=True, is_last_in_batch=True, confirmed_count=5, random_sampling=False,
    )
    assert (should_save, reason) == (False, None)


# --- random-sampling mode: confirmed-track bias disabled, always the Nth video ---

def test_random_sampling_ignores_confirmed_tracks_mid_batch():
    should_save, reason = _video_sample_decision(
        sample_saved=False, is_last_in_batch=False, confirmed_count=3, random_sampling=True,
    )
    assert (should_save, reason) == (False, None)


def test_random_sampling_saves_last_video_regardless_of_confirmed_tracks():
    should_save, reason = _video_sample_decision(
        sample_saved=False, is_last_in_batch=True, confirmed_count=4, random_sampling=True,
    )
    assert (should_save, reason) == (True, "interval")


def test_random_sampling_saves_last_video_with_no_confirmed_tracks():
    should_save, reason = _video_sample_decision(
        sample_saved=False, is_last_in_batch=True, confirmed_count=0, random_sampling=True,
    )
    assert (should_save, reason) == (True, "interval")


def test_random_sampling_respects_already_saved_latch():
    should_save, reason = _video_sample_decision(
        sample_saved=True, is_last_in_batch=True, confirmed_count=0, random_sampling=True,
    )
    assert (should_save, reason) == (False, None)
