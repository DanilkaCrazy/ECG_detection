"""Stage 3.6.14 — no conflicting SNOMED dictionaries remain in the codebase."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from ecgdet import config  # noqa: E402
from ecgdet.labels import map_chapman_dx_to_rhythm, map_ptbxl_scp_to_rhythm  # noqa: E402
from ecgdet.label_resolution import CANONICAL_SNOMED, MAPPING_VERSION  # noqa: E402


def test_all_snomed_dicts_agree_on_afib_and_afl():
    assert config.CHAPMAN_SNOMED_TO_ACRONYM["164889003"] == "AF"
    assert config.CHAPMAN_SNOMED_TO_ACRONYM["164890007"] == "AFL"
    assert config.SNOMED_TO_RHYTHM["164889003"] == "AF"
    assert config.SNOMED_TO_RHYTHM["164890007"] == "AFL"
    assert CANONICAL_SNOMED["164889003"] == ("AFIB", 1)
    assert CANONICAL_SNOMED["164890007"] == ("AFL", 0)


def test_ptbxl_both_codes_is_never_silently_afib():
    assert map_ptbxl_scp_to_rhythm({"AFIB": 100}) == "AF"
    assert map_ptbxl_scp_to_rhythm({"AFLT": 100}) == "AFL"
    assert map_ptbxl_scp_to_rhythm({"AFIB": 100, "AFLT": 100}) is None


def test_chapman_mixed_codes_are_not_first_code_wins():
    assert map_chapman_dx_to_rhythm(["164889003", "164890007"]) is None
    assert map_chapman_dx_to_rhythm(["164890007", "164889003"]) is None
    assert map_chapman_dx_to_rhythm(["164889003", "59118001"]) == "AF"


def test_mapping_version_is_stable_string():
    assert MAPPING_VERSION == "1.0"
