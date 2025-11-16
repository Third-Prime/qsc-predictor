"""
Unit Tests for QSC Predictor
=============================

Ported from: qsc_calculator.test.ts (TypeScript)
Test count: 27 tests

Run with: pytest tests/
Or: pytest tests/test_predictor.py -v
"""

import pytest
import numpy as np
from qsc_predictor import (
    predict_fdm,
    calculate_discharge_timescale,
    classify_regime,
)


# ============================================================================
# TEST 1: DISCHARGE TIMESCALES
# ============================================================================

def test_field_discharge_timescale():
    """Field discharge should be ~6.67 Gyr"""
    tau = calculate_discharge_timescale('field')
    assert 6.0 < tau < 7.5, f"Field discharge timescale {tau:.2f} not in expected range [6.0, 7.5]"


def test_isolated_infinite_discharge():
    """Isolated should have infinite discharge timescale"""
    tau = calculate_discharge_timescale('isolated')
    assert np.isinf(tau), "Isolated discharge timescale should be infinite"


def test_cluster_core_fast_discharge():
    """Cluster core should have fast discharge (~2 Gyr)"""
    tau = calculate_discharge_timescale('cluster_core')
    assert 1.5 < tau < 2.5, f"Cluster core discharge {tau:.2f} not in expected range [1.5, 2.5]"


def test_environment_accelerates_discharge():
    """Environment should accelerate discharge"""
    tau_field = calculate_discharge_timescale('field')
    tau_cluster = calculate_discharge_timescale('cluster_core')
    
    assert tau_cluster < tau_field, "Cluster should discharge faster than field"


def test_interactions_accelerate_discharge():
    """Interactions should accelerate discharge"""
    tau_isolated = calculate_discharge_timescale('field', interaction=0.0)
    tau_merger = calculate_discharge_timescale('field', interaction=1.0)
    
    assert tau_merger < tau_isolated, "Merger should discharge faster than isolated"


def test_old_galaxies_discharge_faster():
    """Old galaxies should discharge faster"""
    tau_young = calculate_discharge_timescale('field', interaction=0, age_gyr=3.0)
    tau_old = calculate_discharge_timescale('field', interaction=0, age_gyr=12.0)
    
    assert tau_old < tau_young, "Old galaxies should discharge faster"


# ============================================================================
# TEST 2: REGIME CLASSIFICATION
# ============================================================================

def test_active_sf_classification():
    """Active SF should classify as charging"""
    regime = classify_regime(1e-9, 3.5, 'field')
    assert regime == 'charging', f"Active SF classified as {regime}, expected 'charging'"


def test_post_starburst_classification():
    """Post-starburst should classify as discharge"""
    regime = classify_regime(1e-12, 8.0, 'field')
    assert regime == 'discharge', f"Post-starburst classified as {regime}, expected 'discharge'"


def test_ancient_quiescent_storage():
    """Ancient quiescent field should classify as storage"""
    regime = classify_regime(1e-15, 13.0, 'field')
    assert regime == 'storage', f"Ancient quiescent classified as {regime}, expected 'storage'"


def test_cluster_quiescent_discharge():
    """Cluster quiescent should classify as discharge"""
    regime = classify_regime(1e-12, 10.0, 'cluster_core')
    assert regime == 'discharge', f"Cluster quiescent classified as {regime}, expected 'discharge'"


def test_isolated_quiescent_storage():
    """Isolated quiescent should classify as storage"""
    regime = classify_regime(1e-13, 12.0, 'isolated')
    assert regime == 'storage', f"Isolated quiescent classified as {regime}, expected 'storage'"


# ============================================================================
# TEST 3: FULL PREDICTIONS - EDGE CASES
# ============================================================================

def test_very_young_galaxy():
    """Very young galaxy should have reasonable f_DM"""
    f_DM = predict_fdm(M_star=1e10, sSFR=1e-8, age=0.5, environment='field', morphology='spiral')
    assert 0.1 <= f_DM <= 1.0, f"Very young galaxy f_DM={f_DM:.3f} out of valid range"


def test_ancient_elliptical_high_fdm():
    """Ancient elliptical should have high f_DM"""
    f_DM = predict_fdm(M_star=1e11, sSFR=1e-15, age=12.0, environment='field', morphology='elliptical')
    assert f_DM > 0.6, f"Ancient elliptical f_DM={f_DM:.3f}, expected >0.6"


def test_post_starburst_regime():
    """Post-starburst should classify as discharge"""
    _, _, _, regime = predict_fdm(
        M_star=1e10, sSFR=1e-12, age=8.0, environment='field',
        morphology='spiral', return_components=True
    )
    assert regime == 'discharge', f"Post-starburst classified as {regime}, expected 'discharge'"


def test_isolated_dwarf_very_high_fdm():
    """Isolated dwarf should have very high f_DM"""
    f_DM = predict_fdm(M_star=1e7, sSFR=1e-13, age=12.0, environment='isolated', morphology='spiral')
    assert f_DM > 0.8, f"Isolated dwarf f_DM={f_DM:.3f}, expected >0.8"


def test_extreme_environment_rapid_discharge():
    """Extreme environment should show rapid discharge"""
    f_DM_field = predict_fdm(M_star=1e10, sSFR=1e-10, age=10.5, environment='field', morphology='spiral')
    f_DM_extreme = predict_fdm(M_star=1e10, sSFR=1e-10, age=10.5, environment='extreme', morphology='spiral')
    
    assert f_DM_extreme < f_DM_field, "Extreme environment should have lower f_DM (more discharge)"


# ============================================================================
# TEST 4: PHYSICAL CONSTRAINTS
# ============================================================================

def test_fdm_bounds():
    """f_DM should always be between 0 and 1"""
    test_cases = [
        {'M_star': 1e7, 'sSFR': 1e-15, 'age': 12.0, 'environment': 'isolated'},
        {'M_star': 1e12, 'sSFR': 1e-9, 'age': 1.0, 'environment': 'field'},
        {'M_star': 1e10, 'sSFR': 1e-10, 'age': 10.5, 'environment': 'extreme'},
        {'M_star': 1e9, 'sSFR': 1e-11, 'age': 8.0, 'environment': 'cluster_core'},
    ]
    
    for params in test_cases:
        f_DM = predict_fdm(**params, morphology='spiral')
        assert 0 <= f_DM <= 1, f"f_DM={f_DM:.3f} out of bounds for {params}"


def test_lower_mass_higher_fdm():
    """Lower mass should have higher f_DM"""
    f_DM_low_mass = predict_fdm(M_star=1e7, sSFR=1e-9, age=3.5, environment='field', morphology='spiral')
    f_DM_high_mass = predict_fdm(M_star=1e12, sSFR=1e-9, age=3.5, environment='field', morphology='spiral')
    
    assert f_DM_low_mass > f_DM_high_mass, "Lower mass should have higher f_DM"


def test_storage_preserves_fdm():
    """Storage regime should preserve f_DM"""
    f_DM_young = predict_fdm(M_star=1e10, sSFR=1e-15, age=13.0, environment='field', morphology='spiral')
    f_DM_old = predict_fdm(M_star=1e10, sSFR=1e-15, age=14.0, environment='field', morphology='spiral')
    
    # f_DM should be very similar (within 5%)
    assert abs(f_DM_young - f_DM_old) < 0.05, "Storage should preserve f_DM over time"


# ============================================================================
# TEST 5: REGRESSION - KNOWN VALUES
# ============================================================================

def test_kmos3d_like_galaxy():
    """KMOS3D-like galaxy should give f_DM ~ 0.4-0.5"""
    f_DM = predict_fdm(M_star=1e10, sSFR=1e-9, age=3.5, environment='field', morphology='spiral')
    assert 0.35 <= f_DM <= 0.55, f"KMOS3D-like galaxy f_DM={f_DM:.3f}, expected 0.35-0.55"


def test_manga_elliptical():
    """MaNGA elliptical should give f_DM ~ 0.65-0.80"""
    f_DM = predict_fdm(M_star=1e11, sSFR=1e-12, age=12.0, environment='field', morphology='elliptical')
    assert 0.65 <= f_DM <= 0.80, f"MaNGA elliptical f_DM={f_DM:.3f}, expected 0.65-0.80"


def test_dsph_very_high_fdm():
    """dSph should have f_DM > 0.85"""
    f_DM = predict_fdm(M_star=1e7, sSFR=1e-15, age=12.0, environment='isolated', morphology='spiral')
    assert f_DM > 0.85, f"dSph f_DM={f_DM:.3f}, expected >0.85"


# ============================================================================
# TEST 6: ARRAY INPUT
# ============================================================================

def test_array_input():
    """Should handle array inputs correctly"""
    M_stars = np.array([1e10, 1e11, 1e7, 1e10])
    sSFRs = np.array([1e-9, 1e-12, 1e-13, 1e-12])
    ages = np.array([3.5, 12.0, 12.0, 5.0])
    envs = np.array(['field', 'field', 'isolated', 'cluster_outskirts'])
    
    f_DMs = predict_fdm(M_stars, sSFRs, ages, envs, morphology='spiral')
    
    assert len(f_DMs) == 4, "Should return 4 values"
    assert all(0 <= f <= 1 for f in f_DMs), "All f_DM values should be in [0, 1]"
    
    # Check specific expectations
    assert 0.35 <= f_DMs[0] <= 0.55, "KMOS3D-like should be 0.35-0.55"
    assert f_DMs[2] > 0.8, "dSph-like should be >0.8"


def test_scalar_and_array_mixed():
    """Should handle mixed scalar and array inputs"""
    M_stars = np.array([1e10, 1e11])
    sSFR_scalar = 1e-10  # Single value for both
    ages = np.array([3.5, 12.0])
    
    f_DMs = predict_fdm(M_stars, sSFR_scalar, ages, environment='field', morphology='spiral')
    
    assert len(f_DMs) == 2, "Should return 2 values"
    assert all(0 <= f <= 1 for f in f_DMs), "All f_DM values should be in [0, 1]"


# ============================================================================
# TEST 7: RETURN COMPONENTS
# ============================================================================

def test_return_components():
    """Should return components when requested"""
    result = predict_fdm(
        M_star=1e10, sSFR=1e-10, age=8.0, environment='field',
        morphology='spiral', return_components=True
    )
    
    assert len(result) == 4, "Should return 4 components"
    f_DM, f_baseline, f_QSC, regime = result
    
    assert 0 <= f_DM <= 1, "f_DM should be in [0, 1]"
    assert 0 <= f_baseline <= 1, "f_baseline should be in [0, 1]"
    assert regime in ['charging', 'storage', 'discharge'], f"Unknown regime: {regime}"


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, '-v'])

