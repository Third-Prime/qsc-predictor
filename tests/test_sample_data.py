"""
Sample Data Validation Tests
=============================

Tests the QSC predictor on a curated sample of 50 galaxies
to ensure it reproduces published results.

Run with: pytest tests/test_sample_data.py -v
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from qsc_predictor import predict_fdm
from scipy.stats import spearmanr


# ============================================================================
# FIXTURE: LOAD SAMPLE DATA
# ============================================================================

@pytest.fixture
def sample_data():
    """Load sample galaxies from CSV"""
    data_path = Path(__file__).parent.parent / 'examples' / 'sample_galaxies.csv'
    
    if not data_path.exists():
        pytest.skip(f"Sample data not found: {data_path}")
    
    df = pd.read_csv(data_path)
    return df


# ============================================================================
# TEST 1: DATA INTEGRITY
# ============================================================================

def test_sample_data_loads(sample_data):
    """Sample data file exists and loads correctly"""
    assert len(sample_data) > 0, "Sample data is empty"
    assert len(sample_data) >= 40, f"Expected at least 40 galaxies, got {len(sample_data)}"


def test_required_columns(sample_data):
    """Sample data has all required columns"""
    required = ['galaxy_id', 'M_star', 'sSFR', 'age', 'environment', 'morphology', 'f_DM_observed']
    
    for col in required:
        assert col in sample_data.columns, f"Missing required column: {col}"


def test_valid_values(sample_data):
    """Sample data has valid physical values"""
    # Mass should be positive
    assert (sample_data['M_star'] > 0).all(), "M_star should be positive"
    
    # sSFR should be positive
    assert (sample_data['sSFR'] > 0).all(), "sSFR should be positive"
    
    # Age should be 0-14 Gyr
    assert (sample_data['age'] >= 0).all(), "Age should be non-negative"
    assert (sample_data['age'] <= 14).all(), "Age should be <= 14 Gyr"
    
    # f_DM should be 0-1
    assert (sample_data['f_DM_observed'] >= 0).all(), "f_DM should be >= 0"
    assert (sample_data['f_DM_observed'] <= 1).all(), "f_DM should be <= 1"


# ============================================================================
# TEST 2: PREDICTIONS
# ============================================================================

def test_predictions_run(sample_data):
    """Predictions complete without errors"""
    f_DM_pred = predict_fdm(
        M_star=sample_data['M_star'].values,
        sSFR=sample_data['sSFR'].values,
        age=sample_data['age'].values,
        environment=sample_data['environment'].values,
        morphology=sample_data['morphology'].values
    )
    
    assert len(f_DM_pred) == len(sample_data), "Should return one prediction per galaxy"
    assert np.isfinite(f_DM_pred).all(), "All predictions should be finite"
    assert (f_DM_pred >= 0).all(), "All predictions should be >= 0"
    assert (f_DM_pred <= 1).all(), "All predictions should be <= 1"


def test_prediction_accuracy(sample_data):
    """Predictions match observed f_DM within tolerance"""
    f_DM_pred = predict_fdm(
        M_star=sample_data['M_star'].values,
        sSFR=sample_data['sSFR'].values,
        age=sample_data['age'].values,
        environment=sample_data['environment'].values,
        morphology=sample_data['morphology'].values
    )
    
    f_DM_obs = sample_data['f_DM_observed'].values
    
    # Calculate RMS error
    errors = f_DM_pred - f_DM_obs
    rms = np.sqrt(np.mean(errors**2))
    
    # Calculate bias
    bias = np.mean(errors)
    
    # RMS should be < 0.06 (6% of f_DM range)
    assert rms < 0.06, f"RMS error {rms:.4f} exceeds threshold 0.06"
    
    # Bias should be < 0.05 (5%)
    assert abs(bias) < 0.05, f"Bias {bias:.4f} exceeds threshold 0.05"


def test_correlation_with_observations(sample_data):
    """Predictions correlate with observations"""
    f_DM_pred = predict_fdm(
        M_star=sample_data['M_star'].values,
        sSFR=sample_data['sSFR'].values,
        age=sample_data['age'].values,
        environment=sample_data['environment'].values,
        morphology=sample_data['morphology'].values
    )
    
    f_DM_obs = sample_data['f_DM_observed'].values
    
    # Calculate Spearman correlation
    rho, p = spearmanr(f_DM_pred, f_DM_obs)
    
    # Correlation should be positive and significant
    assert rho > 0.3, f"Correlation ρ={rho:.3f} too weak (expected >0.3)"
    assert p < 0.01, f"Correlation p-value {p:.3e} not significant"


# ============================================================================
# TEST 3: PHYSICAL TRENDS
# ============================================================================

def test_spiral_vs_elliptical(sample_data):
    """Spirals should have different f_DM than ellipticals"""
    # Filter by morphology
    spirals = sample_data[sample_data['morphology'] == 'spiral']
    ellipticals = sample_data[sample_data['morphology'] == 'elliptical']
    
    if len(spirals) >= 5 and len(ellipticals) >= 5:
        # Get predictions
        f_DM_spiral = predict_fdm(
            M_star=spirals['M_star'].values,
            sSFR=spirals['sSFR'].values,
            age=spirals['age'].values,
            environment=spirals['environment'].values,
            morphology='spiral'
        )
        
        f_DM_elliptical = predict_fdm(
            M_star=ellipticals['M_star'].values,
            sSFR=ellipticals['sSFR'].values,
            age=ellipticals['age'].values,
            environment=ellipticals['environment'].values,
            morphology='elliptical'
        )
        
        # Means should be different
        mean_spiral = np.mean(f_DM_spiral)
        mean_elliptical = np.mean(f_DM_elliptical)
        
        # Allow for variability, but they shouldn't be identical
        assert abs(mean_spiral - mean_elliptical) > 0.05, \
            "Spiral and elliptical f_DM should differ by >0.05"


def test_sf_vs_quiescent(sample_data):
    """Active SF galaxies should have different f_DM than quiescent"""
    # Split by sSFR
    active = sample_data[sample_data['sSFR'] > 1e-10]
    quiescent = sample_data[sample_data['sSFR'] < 1e-11]
    
    if len(active) >= 5 and len(quiescent) >= 5:
        # Get predictions
        f_DM_active = predict_fdm(
            M_star=active['M_star'].values,
            sSFR=active['sSFR'].values,
            age=active['age'].values,
            environment=active['environment'].values,
            morphology=active['morphology'].values
        )
        
        f_DM_quiescent = predict_fdm(
            M_star=quiescent['M_star'].values,
            sSFR=quiescent['sSFR'].values,
            age=quiescent['age'].values,
            environment=quiescent['environment'].values,
            morphology=quiescent['morphology'].values
        )
        
        # Means should be different
        mean_active = np.mean(f_DM_active)
        mean_quiescent = np.mean(f_DM_quiescent)
        
        assert abs(mean_active - mean_quiescent) > 0.05, \
            "Active and quiescent f_DM should differ by >0.05"


# ============================================================================
# TEST 4: SUBSAMPLE ACCURACY
# ============================================================================

def test_kmos3d_subsample(sample_data):
    """KMOS3D-like galaxies should have good accuracy"""
    kmos3d = sample_data[sample_data['source'] == 'KMOS3D'] if 'source' in sample_data.columns else sample_data[
        (sample_data['sSFR'] > 1e-10) & (sample_data['age'] < 5)
    ]
    
    if len(kmos3d) >= 5:
        f_DM_pred = predict_fdm(
            M_star=kmos3d['M_star'].values,
            sSFR=kmos3d['sSFR'].values,
            age=kmos3d['age'].values,
            environment=kmos3d['environment'].values,
            morphology=kmos3d['morphology'].values
        )
        
        f_DM_obs = kmos3d['f_DM_observed'].values
        errors = f_DM_pred - f_DM_obs
        rms = np.sqrt(np.mean(errors**2))
        
        # KMOS3D should have good accuracy
        assert rms < 0.08, f"KMOS3D subsample RMS {rms:.4f} too high"


def test_manga_subsample(sample_data):
    """MaNGA-like galaxies should have good accuracy"""
    manga = sample_data[sample_data['source'] == 'MaNGA'] if 'source' in sample_data.columns else sample_data[
        (sample_data['sSFR'] < 1e-11) & (sample_data['age'] > 8)
    ]
    
    if len(manga) >= 5:
        f_DM_pred = predict_fdm(
            M_star=manga['M_star'].values,
            sSFR=manga['sSFR'].values,
            age=manga['age'].values,
            environment=manga['environment'].values,
            morphology=manga['morphology'].values
        )
        
        f_DM_obs = manga['f_DM_observed'].values
        errors = f_DM_pred - f_DM_obs
        rms = np.sqrt(np.mean(errors**2))
        
        # MaNGA should have good accuracy
        assert rms < 0.08, f"MaNGA subsample RMS {rms:.4f} too high"


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, '-v'])

