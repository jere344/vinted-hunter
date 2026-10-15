"""Hard-requirement comparators: present and absent (yes/no features), the tightened contains, and the abbreviations."""

import pytest
from helpers import make_detail, make_summary

from vinted_hunter.models import Extraction, HardRequirement, ModelSpecs, SearchPlan
from vinted_hunter.pipeline import (
    _extract_context,
    _extraction_digest,
    clean_model_label,
    requirements_status,
    spec_verdict,
)
from vinted_hunter.prompts import EXTRACT_SYSTEM


def _specs(value, key="active noise cancellation", variant_match="exact", lookup_mode="web"):
    return ModelSpecs(model_key="k", specs={key: value}, variant_match=variant_match, lookup_mode=lookup_mode)


def _req(spec_key, comparator, target=None):
    return HardRequirement(text="requirement", spec_key=spec_key, comparator=comparator, target=target)


@pytest.mark.parametrize(
    "value, comparator, verdict",
    [
        ("Yes", "present", "met"),
        ("Up to 25 dB", "present", "met"),
        ("yes, hybrid", "present", "met"),
        ("no", "present", "unmet"),
        ("None", "present", "unmet"),
        ("Non", "present", "unmet"),
        ("not supported", "present", "unmet"),
        ("Not available", "present", "unmet"),
        ("sans", "present", "unmet"),
        ("n/a", "present", "unmet"),
        ("absent", "present", "unmet"),
        ("no, not included", "present", "unmet"),
        ("non-stick", "present", "met"),  # "non" glued to a word is not the negative word
        ("", "present", "unknown"),
        ("unknown", "present", "unknown"),
        ("non renseigné", "present", "unknown"),
        ("yes / no", "present", "unknown"),  # two duplicate values disagree: fail closed
        ("Yes / yes", "present", "met"),
        ("no", "absent", "met"),
        ("n/a", "absent", "met"),
        ("yes", "absent", "unmet"),
        ("Up to 25 dB", "absent", "unmet"),
        ("yes / no", "absent", "unknown"),
    ],
)
def test_present_and_absent_read_a_yes_no_value(value, comparator, verdict):
    assert spec_verdict(_specs(value), _req("active noise cancellation", comparator)) == verdict


def test_present_is_unknown_when_the_spec_key_is_missing():
    specs = _specs("yes", key="battery life")
    assert spec_verdict(specs, _req("active noise cancellation", "present")) == "unknown"


def test_present_needs_no_target():
    assert spec_verdict(_specs("yes"), _req("active noise cancellation", "present", target=None)) == "met"


@pytest.mark.parametrize(
    "value, target, verdict",
    [
        # The Realme case: "ANC" against a value that spells out the same feature must never be unmet.
        ("Up to 25 dB active noise cancellation", "ANC", "met"),
        ("active noise cancellation", "ANC", "met"),
        ("ANC", "active noise cancellation", "met"),
        ("BT 5.0", "Bluetooth 5", "met"),
        ("Wi-Fi 6", "WiFi", "met"),
        # An explicit negative is unmet.
        ("no", "ANC", "unmet"),
        ("Non", "active noise cancellation", "unmet"),
        # A short word of the target may be an abbreviation the value does not spell out: unknown, not unmet.
        ("Micro USB", "USB", "met"),  # the target's word is in the value
        ("HDMI", "USB", "unknown"),
        ("Textile", "cuir", "unmet"),  # four letters, a clean value with another word: unmet as before
    ],
)
def test_contains_is_met_or_unknown_for_abbreviations_and_never_unmet_by_a_word_mismatch(value, target, verdict):
    specs = _specs(value, key="active noise cancellation")
    assert spec_verdict(specs, _req("active noise cancellation", "contains", target)) == verdict


def test_the_realme_case_is_met_through_the_requirement_status():
    plan = SearchPlan(
        product_summary="earbuds",
        queries=["ecouteurs"],
        budget_eur=15,
        spec_keys=["anc"],
        hard_requirements=[_req("anc", "contains", "ANC")],
    )
    extraction = Extraction(item_id="r1", model_confidence=0.9)
    specs = _specs("Up to 25 dB active noise cancellation", key="anc")
    assert spec_verdict(specs, plan.hard_requirements[0]) == "met"
    assert requirements_status(plan, extraction, specs) == "verified"


def test_a_present_requirement_marks_a_listing_failed_when_the_exact_specs_say_no():
    plan = SearchPlan(
        product_summary="earbuds",
        queries=["ecouteurs"],
        budget_eur=15,
        spec_keys=["active noise cancellation"],
        hard_requirements=[_req("active noise cancellation", "present")],
    )
    extraction = Extraction(item_id="r1", model_confidence=0.9)
    assert requirements_status(plan, extraction, _specs("no")) == "failed"
    assert requirements_status(plan, extraction, _specs("yes")) == "verified"


def test_present_requirement_from_base_model_specs_changes_nothing():
    plan = SearchPlan(
        product_summary="earbuds",
        queries=["ecouteurs"],
        budget_eur=15,
        spec_keys=["active noise cancellation"],
        hard_requirements=[_req("active noise cancellation", "present")],
    )
    extraction = Extraction(item_id="r1", model_confidence=0.9)
    assert requirements_status(plan, extraction, _specs("no", variant_match="base_model")) == "unverified"


@pytest.mark.parametrize(
    "model, expected",
    [
        ("EH-TW650", "EH-TW650"),
        ("EH-TW650 (variant not confirmed)", "EH-TW650"),
        ("Realme Buds Air 5 Pro - probably gen 2", "Realme Buds Air 5 Pro"),
        ("Buds Air 5 Pro ?", "Buds Air 5 Pro"),
        ("Galaxy Buds2 (probable)", "Galaxy Buds2"),
        ("Acme X1 (edition 2)", "Acme X1 (edition 2)"),  # a real qualifier stays
        (None, ""),
    ],
)
def test_model_labels_drop_doubt_qualifiers_before_a_search(model, expected):
    assert clean_model_label(model) == expected


def test_extraction_digest_changes_with_the_product_type():
    # The product type is part of the buyer context of the vision step, so a plan change is a new extraction.
    detail = make_detail(make_summary("d1", "Ecouteurs", 10.0))
    plan_a = SearchPlan(product_summary="p", queries=["q"], budget_eur=10, product_type="earbuds")
    plan_b = SearchPlan(product_summary="p", queries=["q"], budget_eur=10, product_type="headphones")
    system_a = f"{EXTRACT_SYSTEM}\n{_extract_context(plan_a)}"
    system_b = f"{EXTRACT_SYSTEM}\n{_extract_context(plan_b)}"
    assert _extraction_digest(system_a, plan_a, detail, []) != _extraction_digest(system_b, plan_b, detail, [])
