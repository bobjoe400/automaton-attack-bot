"""Settings persistence and resolution scaling."""

from automaton_attack_bot.core.config import REFERENCE_RESOLUTION, Settings


def test_defaults_match_the_measured_geometry():
    settings = Settings()
    assert settings.geometry.panel == (481, 57, 1444, 1022)
    assert len(settings.geometry.hud_boxes) == 3


def test_reference_resolution_is_returned_unchanged():
    settings = Settings()
    assert settings.for_resolution(*REFERENCE_RESOLUTION) is settings


def test_geometry_scales_with_resolution():
    scaled = Settings().for_resolution(3840, 2160)
    assert scaled.geometry.panel == (962, 114, 2888, 2044)
    # blob filters must scale too, or nothing passes at 4K
    assert scaled.blobs.min_width == Settings().blobs.min_width * 2
    assert scaled.blobs.min_area == Settings().blobs.min_area * 4


def test_hud_boxes_scale_with_resolution():
    scaled = Settings().for_resolution(960, 540)
    assert scaled.geometry.hud_boxes[0] == (20, 0, 110, 55)


def test_round_trip_through_json(tmp_path):
    original = Settings()
    path = original.save(tmp_path / "automaton.json")
    assert Settings.load(path) == original


def test_round_trip_preserves_tuples(tmp_path):
    """JSON has no tuples; lists coming back would break equality."""
    path = Settings().save(tmp_path / "automaton.json")
    loaded = Settings.load(path)
    assert isinstance(loaded.geometry.panel, tuple)
    assert isinstance(loaded.geometry.hud_boxes[0], tuple)
    assert isinstance(loaded.color.hsv_lo, tuple)


def test_edited_config_is_applied(tmp_path):
    path = Settings().save(tmp_path / "automaton.json")
    import json

    data = json.loads(path.read_text())
    data["color"]["hsv_lo"] = [30, 40, 120]
    data["safe_mode"] = True
    path.write_text(json.dumps(data))

    loaded = Settings.load(path)
    assert loaded.color.hsv_lo == (30, 40, 120)
    assert loaded.safe_mode is True


def test_unknown_keys_are_ignored(tmp_path):
    """An old config with a since-renamed field should not crash the bot."""
    import json

    path = tmp_path / "automaton.json"
    path.write_text(json.dumps({"blobs": {"min_area": 900,
                                          "retired_option": 1}}))
    assert Settings.load(path).blobs.min_area == 900
