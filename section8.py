##############################################################
# SECTION 8: 10-DISTRICT DYNAMIC SUPPORT MODEL (REALISTIC)
# - swing vs safe seats
# - heterogeneous volatility
# - structural partisan lean
# - nonlinear campaign responsiveness
# - emergent national result
##############################################################

import numpy as np
import pandas as pd

# -----------------------------
# INPUT SAFETY
# -----------------------------
support_std = np.clip(support_std, 0.005, 0.05)
recent_poll_trend = np.clip(recent_poll_trend, -0.01, 0.01)

# -----------------------------
# CONFIGURATION
# -----------------------------
NUM_DISTRICTS = 10
np.random.seed(42)

district_ids = np.arange(NUM_DISTRICTS)

# ============================================================
# 1. STRUCTURAL PARTISAN LEAN (SAFE vs SWING FOUNDATION)
# ============================================================

district_lean = np.random.normal(0, 0.12, NUM_DISTRICTS)

district_baseline = (
    latent_support
    + district_lean
    + np.random.normal(0, 0.02, NUM_DISTRICTS)
)

district_baseline = np.clip(district_baseline, 0.01, 0.99)

# ------------------------------------------------------------
# SWING SCORE (competitiveness of each district)
# ------------------------------------------------------------
margin = np.abs(district_baseline - 0.5)
swing_score = np.exp(-margin * 10)

# ============================================================
# 2. DISTRICT UNCERTAINTY (HIGH IN SWING SEATS)
# ============================================================

district_uncertainty = np.clip(
    0.01 + 0.04 * swing_score + np.random.normal(0, 0.003, NUM_DISTRICTS),
    0.005,
    0.06
)

# ============================================================
# 3. DISTRICT SENSITIVITY (STABILITY MEASURE)
# ============================================================

district_sensitivity = 1 / (1 + district_uncertainty * 25)

# ============================================================
# 4. NONLINEAR CAMPAIGN RESPONSIVENESS
# ============================================================

responsiveness = 0.3 + 1.7 * swing_score

fund_effect_d = fund_effect * district_sensitivity * responsiveness
field_effect_d = field_effect * district_sensitivity * responsiveness
momentum_effect_d = momentum_effect * district_sensitivity * responsiveness
early_vote_d = early_vote_boost * district_sensitivity * responsiveness

# ============================================================
# 5. LOCAL SHOCKS (DISTRICT-LEVEL RANDOMNESS)
# ============================================================

local_shocks = np.random.normal(
    0,
    district_uncertainty,
    NUM_DISTRICTS
)

# ============================================================
# 6. FINAL DISTRICT SUPPORT
# ============================================================

district_support = (
    district_baseline
    + fund_effect_d
    + field_effect_d
    + momentum_effect_d
    + early_vote_d
    + local_shocks
)

district_support = np.clip(district_support, 0.01, 0.99)

# ============================================================
# 7. DISTRICT CLASSIFICATION (SAFE / LEAN / SWING)
# ============================================================

district_type = np.where(
    swing_score > 0.7, "swing",
    np.where(
        district_baseline > 0.6, "safe_pro",
        np.where(district_baseline < 0.4, "safe_opp", "lean")
    )
)

# ============================================================
# 8. SUMMARY TABLE
# ============================================================

district_df = pd.DataFrame({
    "district": district_ids,
    "baseline_support": district_baseline,
    "lean": district_lean,
    "swing_score": swing_score,
    "uncertainty": district_uncertainty,
    "sensitivity": district_sensitivity,
    "responsiveness": responsiveness,
    "final_support": district_support,
    "type": district_type
})

# ============================================================
# 9. NATIONAL RESULTS (EMERGENT FROM DISTRICTS)
# ============================================================

national_support_from_districts = district_support.mean()
district_wins = (district_support > 0.5).sum()

# ============================================================
# 10. OUTPUT
# ============================================================

print("\n==============================")
print("SECTION 8: 10-DISTRICT MODEL (REALISTIC)")
print("==============================\n")

print("National latent support (Section 5):",
      round(latent_support * 100, 2), "%")

print("National adjusted support (Section 6):",
      round(base_support * 100, 2), "%")

print("National support (from districts):",
      round(national_support_from_districts * 100, 2), "%")

print("\nDistrict wins (>50%):",
      int(district_wins), "/ 10")

print("\nDistrict breakdown:")
print(district_df)

print("\nInterpretation:")
print("- Districts now have structural partisan lean (safe vs swing).")
print("- Swing districts drive volatility and responsiveness.")
print("- Campaign effects are stronger where elections are competitive.")
print("- National outcome emerges from heterogeneous local systems.")
print("==============================\n")
