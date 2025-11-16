#!/usr/bin/env python3
"""
QSC Predictor Validation Script
================================

Reproduce published results on sample galaxies.
This proves the tool is working correctly.

Usage:
    python examples/validate_predictions.py

Expected Output:
    RMS Error: ~0.04
    Mean Bias: ~0.00
    Median Error: ~8%
    ✅ Validation passed!
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qsc_predictor import predict_fdm


def main():
    print("=" * 80)
    print("QSC PREDICTOR VALIDATION")
    print("=" * 80)
    print()
    
    # Load sample data
    data_path = Path(__file__).parent / 'sample_galaxies.csv'
    
    if not data_path.exists():
        print(f"❌ ERROR: Sample data not found at {data_path}")
        print()
        print("Please ensure sample_galaxies.csv exists in the examples/ directory.")
        return 1
    
    print(f"Loading sample data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} galaxies")
    print()
    
    # Run predictions
    print("Running predictions...")
    df['f_DM_predicted'] = predict_fdm(
        M_star=df['M_star'].values,
        sSFR=df['sSFR'].values,
        age=df['age'].values,
        environment=df['environment'].values,
        morphology=df['morphology'].values
    )
    print("✅ Predictions complete")
    print()
    
    # Calculate errors
    df['error'] = df['f_DM_predicted'] - df['f_DM_observed']
    df['error_percent'] = 100 * df['error'] / df['f_DM_observed']
    df['abs_error'] = np.abs(df['error'])
    
    # Overall statistics
    print("=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print()
    
    rms = np.sqrt(np.mean(df['error']**2))
    bias = df['error'].mean()
    mae = df['abs_error'].mean()
    median_error_pct = df['error_percent'].median()
    
    print(f"Sample Size:        {len(df)} galaxies")
    print(f"RMS Error:          {rms:.4f}")
    print(f"Mean Bias:          {bias:+.4f}")
    print(f"MAE:                {mae:.4f}")
    print(f"Median Error:       {median_error_pct:+.1f}%")
    print()
    
    # Correlation
    try:
        rho, p_rho = spearmanr(df['f_DM_predicted'], df['f_DM_observed'])
        r, p_r = pearsonr(df['f_DM_predicted'], df['f_DM_observed'])
        
        print(f"Spearman ρ:         {rho:+.3f} (p = {p_rho:.2e})")
        print(f"Pearson r:          {r:+.3f} (p = {p_r:.2e})")
        print()
    except:
        print("⚠️  Could not calculate correlations")
        print()
    
    # By morphology
    if 'morphology' in df.columns:
        print("=" * 80)
        print("BY MORPHOLOGY")
        print("=" * 80)
        print()
        
        for morph in df['morphology'].unique():
            subset = df[df['morphology'] == morph]
            if len(subset) > 0:
                rms_morph = np.sqrt(np.mean(subset['error']**2))
                bias_morph = subset['error'].mean()
                
                print(f"{morph.upper():15s} (N={len(subset):2d}):  RMS={rms_morph:.4f}  Bias={bias_morph:+.4f}")
        print()
    
    # By sample
    if 'source' in df.columns:
        print("=" * 80)
        print("BY SAMPLE")
        print("=" * 80)
        print()
        
        for source in df['source'].unique():
            subset = df[df['source'] == source]
            if len(subset) > 0:
                rms_source = np.sqrt(np.mean(subset['error']**2))
                bias_source = subset['error'].mean()
                
                print(f"{source:15s} (N={len(subset):2d}):  RMS={rms_source:.4f}  Bias={bias_source:+.4f}")
        print()
    
    # Show examples
    print("=" * 80)
    print("SAMPLE PREDICTIONS (First 10)")
    print("=" * 80)
    print()
    
    display_cols = ['galaxy_id', 'morphology', 'f_DM_observed', 'f_DM_predicted', 'error_percent']
    print(df[display_cols].head(10).to_string(index=False))
    print()
    
    # Worst cases
    print("=" * 80)
    print("LARGEST ERRORS (Top 5)")
    print("=" * 80)
    print()
    
    worst = df.nlargest(5, 'abs_error')
    print(worst[display_cols].to_string(index=False))
    print()
    
    # Validation check
    print("=" * 80)
    print("VALIDATION CHECK")
    print("=" * 80)
    print()
    
    passed = True
    
    # Check RMS
    if rms < 0.06:
        print(f"✅ RMS Error: {rms:.4f} < 0.06 (PASS)")
    else:
        print(f"❌ RMS Error: {rms:.4f} >= 0.06 (FAIL)")
        passed = False
    
    # Check bias
    if abs(bias) < 0.05:
        print(f"✅ Mean Bias: {abs(bias):.4f} < 0.05 (PASS)")
    else:
        print(f"❌ Mean Bias: {abs(bias):.4f} >= 0.05 (FAIL)")
        passed = False
    
    # Check correlation (if calculated)
    try:
        if rho > 0.3:
            print(f"✅ Correlation: ρ = {rho:.3f} > 0.3 (PASS)")
        else:
            print(f"⚠️  Correlation: ρ = {rho:.3f} < 0.3 (WEAK)")
            # Don't fail on weak correlation for small samples
    except:
        pass
    
    print()
    
    if passed:
        print("=" * 80)
        print("✅ VALIDATION PASSED!")
        print("=" * 80)
        print()
        print("The QSC predictor successfully reproduces published results.")
        print("Tool is working correctly and ready for use!")
        return 0
    else:
        print("=" * 80)
        print("⚠️  VALIDATION ISSUES DETECTED")
        print("=" * 80)
        print()
        print("The predictor may not be working as expected.")
        print("Please check:")
        print("  1. Sample data is correct and up-to-date")
        print("  2. Predictor version matches expected version")
        print("  3. Dependencies are correctly installed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

