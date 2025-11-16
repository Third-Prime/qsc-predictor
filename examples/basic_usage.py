#!/usr/bin/env python3
"""
QSC Predictor - Basic Usage Examples
=====================================

Simple examples demonstrating how to use the QSC predictor.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qsc_predictor import predict_fdm, predict_kmos3d, predict_manga, predict_dsph


print("=" * 80)
print("QSC PREDICTOR - BASIC USAGE EXAMPLES")
print("=" * 80)
print()

# ============================================================================
# EXAMPLE 1: Single Galaxy
# ============================================================================

print("EXAMPLE 1: Predict f_DM for a single galaxy")
print("-" * 80)

f_DM = predict_fdm(
    M_star=1e10,          # 10 billion solar masses
    sSFR=1e-10,           # Specific star formation rate (per year)
    age=8.0,              # 8 Gyr old
    environment='field',  # Field galaxy (not in cluster)
    morphology='spiral'   # Spiral morphology
)

print(f"Galaxy properties:")
print(f"  M_star  = 10^10 M☉")
print(f"  sSFR    = 10^-10 yr^-1")
print(f"  Age     = 8 Gyr")
print(f"  Type    = Field spiral")
print()
print(f"Predicted f_DM = {f_DM:.3f}")
print()
print()

# ============================================================================
# EXAMPLE 2: Multiple Galaxies (Array Input)
# ============================================================================

print("EXAMPLE 2: Batch prediction for multiple galaxies")
print("-" * 80)

# Define properties for 4 galaxies
M_stars = np.array([1e10, 1e11, 1e9, 1e10])
sSFRs = np.array([1e-9, 1e-12, 1e-10, 1e-12])
ages = np.array([3.5, 12.0, 5.0, 8.0])
environments = np.array(['field', 'field', 'field', 'cluster_outskirts'])
morphologies = np.array(['spiral', 'elliptical', 'irregular', 'elliptical'])
interactions = np.array([0.0, 0.0, 0.0, 0.0])  # No interactions

# Predict f_DM for all at once
f_DMs = predict_fdm(M_stars, sSFRs, ages, environments, interactions, morphology=morphologies)

print(f"{'Galaxy':<10s} {'M_star':<12s} {'sSFR':<12s} {'Age':>6s} {'Type':<15s} {'f_DM':>8s}")
print("-" * 80)
for i in range(len(M_stars)):
    m_str = f"{M_stars[i]:.1e}"
    s_str = f"{sSFRs[i]:.1e}"
    morph_str = f"{morphologies[i]}"
    print(f"Galaxy {i+1:<3d} {m_str:<12s} {s_str:<12s} {ages[i]:>6.1f} {morph_str:<15s} {f_DMs[i]:>8.3f}")
print()
print()

# ============================================================================
# EXAMPLE 3: Different Galaxy Types
# ============================================================================

print("EXAMPLE 3: Predictions for different galaxy types")
print("-" * 80)

# Young star-forming galaxy (like KMOS3D)
f_kmos3d = predict_fdm(M_star=1e10, sSFR=1e-9, age=3.5, 
                       environment='field', morphology='spiral')
print(f"Young SF galaxy (KMOS3D-like):     f_DM = {f_kmos3d:.3f}")

# Old quiescent galaxy (like MaNGA)
f_manga = predict_fdm(M_star=1e11, sSFR=1e-12, age=12.0,
                      environment='field', morphology='elliptical')
print(f"Old quiescent (MaNGA-like):        f_DM = {f_manga:.3f}")

# Dwarf spheroidal
f_dsph = predict_fdm(M_star=1e7, sSFR=1e-13, age=12.0,
                     environment='isolated', morphology='elliptical')
print(f"Dwarf spheroidal:                  f_DM = {f_dsph:.3f}")

# Post-starburst in cluster
f_psb = predict_fdm(M_star=1e10, sSFR=1e-12, age=5.0,
                    environment='cluster_outskirts', regime='discharge',
                    morphology='elliptical')
print(f"Post-starburst (cluster):          f_DM = {f_psb:.3f}")

print()
print()

# ============================================================================
# EXAMPLE 4: Get Component Breakdown
# ============================================================================

print("EXAMPLE 4: Get component breakdown")
print("-" * 80)

f_DM, f_baseline, f_QSC, regime = predict_fdm(
    M_star=1e10, sSFR=1e-10, age=8.0,
    environment='field', morphology='spiral',
    return_components=True
)

print(f"Galaxy: M_star=10^10 M☉, sSFR=10^-10 yr^-1, age=8 Gyr")
print()
print(f"Regime:              {regime}")
print(f"f_DM (total):        {f_DM:.3f}")
print(f"f_baseline:          {f_baseline:.3f}")
print(f"f_QSC:               {f_QSC:.3f}")
print()
print(f"QSC effect:          {(f_QSC - f_baseline):.3f} ({(f_QSC/f_baseline - 1)*100:+.1f}%)")
print()
print()

# ============================================================================
# EXAMPLE 5: Convenience Functions
# ============================================================================

print("EXAMPLE 5: Convenience functions for specific samples")
print("-" * 80)

# KMOS3D-like galaxy
f1 = predict_kmos3d(M_star=1e10, sSFR=1e-9, age=3.5)
print(f"KMOS3D-like galaxy:    f_DM = {f1:.3f}")

# MaNGA-like galaxy
f2 = predict_manga(M_star=1e10, sSFR=1e-12, age=12.0)
print(f"MaNGA-like galaxy:     f_DM = {f2:.3f}")

# Dwarf spheroidal
f3 = predict_dsph(M_star=1e7, age=12.0)
print(f"Dwarf spheroidal:      f_DM = {f3:.3f}")

print()
print()

# ============================================================================
# EXAMPLE 6: Environmental Effects
# ============================================================================

print("EXAMPLE 6: Environmental effects on discharge")
print("-" * 80)

# Same galaxy in different environments
M_test = 1e10
s_test = 1e-12
a_test = 10.0

environments_test = ['isolated', 'field', 'group', 'cluster_outskirts', 'cluster_core']

print(f"Same galaxy (M=10^10 M☉, sSFR=10^-12, age=10 Gyr) in different environments:")
print()

for env in environments_test:
    f_env = predict_fdm(M_test, s_test, a_test, environment=env, morphology='elliptical')
    print(f"  {env:<20s}: f_DM = {f_env:.3f}")

print()
print("Note: Isolated preserves f_DM, cluster_core causes rapid discharge")
print()

# ============================================================================
# DONE
# ============================================================================

print("=" * 80)
print("✅ Examples complete!")
print("=" * 80)
print()
print("For more information, see README.md or run:")
print("  python examples/validate_predictions.py")
print()

