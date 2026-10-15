import pytest

from vinted_hunter.config import load_settings


def test_research_limit_is_unlimited_by_default_and_the_ttl_is_seventy_two_hours():
    settings = load_settings(environ={}, env_file=None)
    assert settings.specs_max_per_hunt == 0
    assert settings.specs_nonexact_ttl_hours == 72.0


def test_research_limit_and_ttl_are_read_from_the_environment():
    settings = load_settings(
        environ={"SPECS_MAX_PER_HUNT": "3", "SPECS_NONEXACT_TTL_HOURS": "24"}, env_file=None
    )
    assert settings.specs_max_per_hunt == 3
    assert settings.specs_nonexact_ttl_hours == 24.0


def test_zero_research_limit_is_accepted_as_unlimited():
    assert load_settings(environ={"SPECS_MAX_PER_HUNT": "0"}, env_file=None).specs_max_per_hunt == 0


def test_a_negative_research_limit_is_rejected():
    with pytest.raises(ValueError, match="SPECS_MAX_PER_HUNT"):
        load_settings(environ={"SPECS_MAX_PER_HUNT": "-1"}, env_file=None)


def test_a_negative_ttl_is_rejected():
    with pytest.raises(ValueError, match="SPECS_NONEXACT_TTL_HOURS"):
        load_settings(environ={"SPECS_NONEXACT_TTL_HOURS": "-2"}, env_file=None)
