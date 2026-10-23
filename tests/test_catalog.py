# -*- coding: utf-8 -*-
"""Tests for the generic archetype catalog and its deprecated branded aliases."""

import os
import sys
import warnings

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from energysim.models import catalog, presets  # noqa: E402

ARCHETYPES = [
    "BESS_4h_2MW", "BESS_2h_2MW", "Alkaline_2MW", "PEM_5MW", "SOEC_1MW", "CCGT_570MW",
    "Gas_Turbine_60MW", "Gas_Engine_10MW", "Solar_Farm", "Wind_Turbine_4MW",
]

# alias -> (archetype, expected attributes with the old numbers, class-name of result)
ALIASES = {
    "Tesla_Megapack_2XL": ("BESS_2h_2MW", {"capacity_mwh": 3.916, "p_max_mw": 1.929}),
    "Fluence_Gridstack": ("BESS_4h_2MW", {"capacity_mwh": 8.0, "p_max_mw": 2.0}),
    "Nel_A1000": ("Alkaline_2MW", {"p_nominal_mw": 2.2}),
    "ITM_Power_Trident": ("PEM_5MW", {"p_nominal_mw": 5.0}),
    "Bloom_Energy_SOEC": ("SOEC_1MW", {"p_nominal_mw": 1.0, "spec_elec_kwh_kg": 37.5}),
    "GE_7HA_02": ("CCGT_570MW", {"p_rated_mw": 573.0}),
    "Siemens_SGT_800": ("Gas_Turbine_60MW", {"p_rated_mw": 62.0}),
    "Wartsila_34SG": ("Gas_Engine_10MW", {"p_rated": 9.7}),
    "CanadianSolar_HiKu7": ("Solar_Farm", {}),
    "Vestas_V150_4MW": ("Wind_Turbine_4MW", {"p_rated_mw": 4.2}),
}


@pytest.mark.parametrize("name", ARCHETYPES)
def test_archetypes_build_and_are_documented(name):
    factory = getattr(catalog, name)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # archetypes must not warn
        model = factory()
    assert model is not None and model.name
    assert "illustrative archetype, not calibrated to an oem datasheet" in factory.__doc__.lower()
    assert name in catalog.PROVENANCE and catalog.PROVENANCE[name]
    assert getattr(presets, name) is factory


def test_archetype_values_and_overrides():
    bess = catalog.BESS_4h_2MW()
    assert (bess.capacity_mwh, bess.p_max_mw) == (8.0, 2.0)
    assert catalog.BESS_2h_2MW().capacity_mwh == 4.0
    assert catalog.BESS_2h_2MW(capacity_mwh=10.0, name="x").capacity_mwh == 10.0
    assert catalog.Alkaline_2MW().p_nominal_mw == 2.0
    assert catalog.CCGT_570MW().p_rated_mw == 570.0
    assert catalog.Solar_Farm(p_mw=20.0).inverter_p_ac_max_mw == 20.0


@pytest.mark.parametrize("alias", sorted(ALIASES))
def test_deprecated_aliases_warn_and_keep_numbers(alias):
    archetype, expected = ALIASES[alias]
    factory = getattr(catalog, alias)
    with pytest.warns(DeprecationWarning, match=archetype):
        model = factory()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = getattr(catalog, archetype)()
    assert type(model) is type(reference)
    for attr, value in expected.items():
        assert getattr(model, attr) == pytest.approx(value)
    assert alias in catalog.PROVENANCE and "DEPRECATED" in catalog.PROVENANCE[alias]
    assert getattr(presets, alias) is factory
    # custom name is honoured
    with pytest.warns(DeprecationWarning):
        assert factory(name="custom").name == "custom"


def test_nel_alias_keeps_suspect_2_2_mw_and_notes_it():
    with pytest.warns(DeprecationWarning):
        assert catalog.Nel_A1000().p_nominal_mw == 2.2
    assert "2.2" in catalog.PROVENANCE["Nel_A1000"]
    assert catalog.Alkaline_2MW().p_nominal_mw == 2.0


def test_provenance_covers_everything():
    for name in ARCHETYPES + list(ALIASES):
        assert name in catalog.PROVENANCE
    assert set(catalog.__all__) >= set(ARCHETYPES) | set(ALIASES) | {"PROVENANCE"}
