"""Stage 3.6.14 — explicit label resolution tests (TDD)."""
import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from ecgdet.diagnosis_provenance import DiagnosisRecord  # noqa: E402
from ecgdet.label_resolution import (  # noqa: E402
    AFIB_CODES, AFL_CODES, CANONICAL_SNOMED, MAPPING_VERSION,
    resolve_binary, resolve_diagnosis_codes, resolve_chapman_acronym,
)


def rec(codes, source="chapman"):
    return DiagnosisRecord(source=source, record_id="X", raw_diagnosis_codes=list(codes),
                           raw_diagnosis_values={}, primary_diagnosis=codes[0] if codes else None,
                           rhythm_codes=list(codes), secondary_diagnoses=[])


def test_canonical_snomed_mapping_is_unambiguous():
    assert CANONICAL_SNOMED["164889003"] == ("AFIB", 1)
    assert CANONICAL_SNOMED["164890007"] == ("AFL", 0)
    assert MAPPING_VERSION == "1.1"


@pytest.mark.parametrize("codes,status,label", [
    (["164889003"], "AFIB", 1),
    (["164890007"], "AFL", 0),
    (["AFIB"], "AFIB", 1),
    (["AFLT"], "AFL", 0),
    (["164889003", "164890007"], "AMBIGUOUS", None),
    (["AFIB", "AFLT"], "AMBIGUOUS", None),
    (["AFLT", "AFIB"], "AMBIGUOUS", None),   # order must NOT matter
    ([], "UNKNOWN", None),
    (["59118001"], "OTHER", None),
])
def test_resolve_diagnosis_codes(codes, status, label):
    r = resolve_diagnosis_codes(codes)
    assert r.label_status == status
    assert r.label == label


def test_resolve_binary_mutates_record_with_full_provenance():
    d = rec(["164889003", "59118001"])
    out = resolve_binary(d)
    assert out is d
    assert out.canonical_rhythm == "AFIB"
    assert out.label == 1
    assert out.label_status == "AFIB"
    assert out.mapping_version == MAPPING_VERSION
    assert out.raw_diagnosis_codes == ["164889003", "59118001"]


def test_zero_likelihood_target_is_not_a_diagnosis():
    from ecgdet.diagnosis_provenance import DiagnosisRecord
    r = DiagnosisRecord(source="ptbxl", record_id="1",
                        raw_diagnosis_codes=["AFIB", "NORM"],
                        raw_diagnosis_values={"AFIB": 0.0, "NORM": 100.0},
                        primary_diagnosis="AFIB", rhythm_codes=["AFIB", "NORM"],
                        secondary_diagnoses=[])
    out = resolve_binary(r)
    assert out.label is None
    assert out.label_status == "OTHER"
    assert any("likelihood 0" in n for n in out.notes)


def test_positive_likelihood_afib_still_resolves():
    from ecgdet.diagnosis_provenance import DiagnosisRecord
    r = DiagnosisRecord(source="ptbxl", record_id="2",
                        raw_diagnosis_codes=["AFIB", "NORM"],
                        raw_diagnosis_values={"AFIB": 100.0, "NORM": 0.0},
                        primary_diagnosis="AFIB", rhythm_codes=["AFIB", "NORM"],
                        secondary_diagnoses=[])
    out = resolve_binary(r)
    assert out.label_status == "AFIB" and out.label == 1


def test_t_wave_change_is_not_afib():
    """SNOMED 164934002 = T wave Change (TWC), NOT atrial fibrillation."""
    assert "164934002" not in AFIB_CODES
    out = resolve_diagnosis_codes(["164934002"], source="chapman")
    assert out.label_status == "OTHER" and out.label is None


def test_t_wave_change_does_not_turn_a_clear_afib_into_ambiguous():
    out = resolve_diagnosis_codes(["164889003", "164934002"], source="chapman")
    assert out.label_status == "AFIB" and out.label == 1


def test_chapman_raw_acronym_af_means_atrial_flutter():
    """Raw Chapman 'AF' = Atrial Flutter; internal harmonised 'AF' = AFIB."""
    assert resolve_chapman_acronym("AF") == "AFL"
    assert resolve_chapman_acronym("AFIB") == "AFIB"
    assert resolve_chapman_acronym("AFL") == "AFL"
    assert "AF" not in AFIB_CODES  # never accept the bare acronym without namespace
    assert "164890007" in AFL_CODES


def test_ambiguous_records_never_receive_a_binary_label():
    for codes in (["164889003", "164890007"], ["AFIB", "AFLT"]):
        out = resolve_diagnosis_codes(codes)
        assert out.label is None
        assert out.label_status == "AMBIGUOUS"
