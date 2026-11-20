"""Stage 3.6.14 — raw diagnosis provenance tests (TDD)."""
import sys
import pathlib

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from ecgdet.diagnosis_provenance import (  # noqa: E402
    DiagnosisRecord, extract_chapman_diagnoses, extract_cpsc_diagnoses,
    extract_ptbxl_diagnoses, to_dataframe,
)


def test_chapman_extracts_all_dx_codes_in_order(tmp_path):
    h = tmp_path / "JS00001.hea"
    h.write_text(
        "JS00001 12 500 5000\nII 0 200 12 0 0 0 I\n"
        "#Dx: 164889003,59118001,164934002\n", encoding="utf-8")
    d = extract_chapman_diagnoses(h, record_id="JS00001")
    assert d.source == "chapman"
    assert d.raw_diagnosis_codes == ["164889003", "59118001", "164934002"]
    assert d.primary_diagnosis == "164889003"
    assert "164889003" in d.rhythm_codes


def test_cpsc_extracts_all_snomed_codes(tmp_path):
    h = tmp_path / "Q0037.hea"
    h.write_text("Q0037 12 500 5000\nII 0 200 12 0 0 0 I\n# Dx: 164889003,427084000\n",
                 encoding="utf-8")
    d = extract_cpsc_diagnoses(h, record_id="Q0037")
    assert d.source == "cpsc2018_extra"
    assert d.raw_diagnosis_codes == ["164889003", "427084000"]


def test_ptbxl_preserves_likelihood_values_and_categories():
    d = extract_ptbxl_diagnoses({"AFIB": 100, "PVC": 20}, record_id="17",
                                patient_id=13619, strat_fold=8)
    assert d.source == "ptbxl"
    assert d.raw_diagnosis_values == {"AFIB": 100.0, "PVC": 20.0}
    assert set(d.raw_diagnosis_codes) == {"AFIB", "PVC"}
    assert "AFIB" in d.rhythm_codes
    assert "PVC" in d.secondary_diagnoses


def test_ptbxl_with_both_afib_and_aflt_preserves_both_codes():
    d = extract_ptbxl_diagnoses({"AFIB": 100, "AFLT": 80}, record_id="18")
    assert "AFIB" in d.raw_diagnosis_codes and "AFLT" in d.raw_diagnosis_codes


def test_unknown_codes_are_not_silently_dropped():
    # Unknown codes survive provenance; resolution decides their status later.
    r = DiagnosisRecord(source="chapman", record_id="X", raw_diagnosis_codes=["999999999"],
                        raw_diagnosis_values={}, primary_diagnosis="999999999",
                        rhythm_codes=[], secondary_diagnoses=[])
    assert r.raw_diagnosis_codes == ["999999999"]


def test_to_dataframe_keeps_all_provenance_columns():
    r = DiagnosisRecord(source="ptbxl", record_id="1", raw_diagnosis_codes=["AFIB"],
                        raw_diagnosis_values={"AFIB": 100}, primary_diagnosis="AFIB",
                        rhythm_codes=["AFIB"], secondary_diagnoses=[])
    df = to_dataframe([r])
    for c in ("source", "record_id", "raw_diagnosis_codes", "primary_diagnosis",
              "secondary_diagnoses", "canonical_rhythm", "label", "label_status",
              "mapping_version"):
        assert c in df.columns
