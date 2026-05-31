import pandas as pd
import numpy as np

##############################################################
# SECTION 1: Raw polls added
##############################################################

polls = pd.DataFrame([
    {"pollster": "A", "support": 0.48, "days_old": 2, "sample": 1000, "quality_rating": 0.9,
     "house_effect": 0.01, "mode_adjustment": -0.005},

    {"pollster": "B", "support": 0.52, "days_old": 10, "sample": 800, "quality_rating": 0.8,
     "house_effect": -0.02, "mode_adjustment": 0.01},

    {"pollster": "C", "support": 0.50, "days_old": 5, "sample": 1200, "quality_rating": 0.95,
     "house_effect": 0.00, "mode_adjustment": 0.00}
])
#here we take the support value subtract house effect and add mode adjustment like tv phone online
polls["adjusted_support"] = (
    polls["support"]
    - polls["house_effect"]
    + polls["mode_adjustment"]
)
##############################################################
# SECTION 2: POLL WEIGHTING
##############################################################

def calculate_poll_weights(df):

    # newer polls matter more
    recency = 1 / (1 + df["days_old"])

    # bigger samples matter more
    sample = np.sqrt(df["sample"])

    # higher-quality pollsters matter more
    quality = df["quality_rating"]

    weights = recency * sample * quality

    # normalize weights
    return weights / weights.sum()


polls["weight"] = calculate_poll_weights(polls)

weighted_support = np.average(
    polls["adjusted_support"],
    weights=polls["weight"]
)

print("Weighted poll support:",
      round(weighted_support * 100, 2), "%")

##############################################################
# SECTION 3: MOMENTUM 
##############################################################

# -----------------------------------
# Prepare time + polling data
# -----------------------------------

# Convert poll age into time direction
# Smaller days_old = newer poll
# We flip sign so newer polls are "larger" in time
x = -polls["days_old"].values

# Poll support values after bias correction
y = polls["adjusted_support"].values


# -----------------------------------
# Weight recent polls more heavily
# -----------------------------------

weights = 1 / (1 + polls["days_old"].values)


# -----------------------------------
# Fit linear trend line
# y = a + bx
#
# slope = trend over time
# intercept = baseline level
# -----------------------------------

slope, intercept = np.polyfit(
    x,
    y,
    1,
    w=weights
)


# -----------------------------------
# Momentum signal
# -----------------------------------

recent_poll_trend = slope

recent_poll_trend = np.clip(recent_poll_trend, -0.01, 0.01)
print("\n========================")
print("SECTION 3 RESULTS")
print("========================")
print("Momentum trend:", round(recent_poll_trend * 100, 4), "%")


##############################################################
# SECTION 4: UNDECIDED VOTERS (DYNAMIC VERSION)
##############################################################

# -----------------------------------
# Estimated undecided voters
# Example:
# 0.05 = 5%
# -----------------------------------

undecided = 5.0 / 100


# -----------------------------------
# Polling lean
#
# If weighted support > 50%,
# undecideds lean slightly toward us
# -----------------------------------

lean = weighted_support - 0.5


# -----------------------------------
# Dynamic undecided split
#
# Competitive races:
# closer to 50/50 split
#
# Stronger candidates:
# capture slightly more undecideds
# -----------------------------------

expected_split = np.clip(
    0.5 + lean * 0.5,
    0.35,
    0.65
)


# -----------------------------------
# Final undecided gain
# -----------------------------------

undecided_gain = undecided * expected_split


print("\n========================")
print("SECTION 4 RESULTS")
print("========================")
print("Expected undecided split:",
      round(expected_split * 100, 2), "%")

print("Undecided gain:",
      round(undecided_gain * 100, 2), "%")
weighted_support = np.clip(weighted_support, 0.01, 0.99)
##############################################################
# SECTION 5: LATENT POLLING ESTIMATOR (BAYESIAN-STYLE)
##############################################################
undecided_gain = np.clip(undecided_gain, 0, 0.05)

def calculate_poll_precision(df):
    """
    Higher value = more reliable poll
    """

    # ---- Recency decay (information staleness) ----
    time_decay = np.exp(-df["days_old"] / 30)

    # ---- Sample-driven precision (statistical reliability) ----
    # sqrt is replaced with log-stabilized growth
    sample_precision = np.log1p(df["sample"]) / np.log1p(df["sample"]).max()

    # ---- Quality acts as variance reduction ----
    quality_precision = np.clip(df["quality_rating"], 0.1, 1.5)

    # ---- Combined precision (interpretable as 1/variance proxy) ----
    precision = time_decay * sample_precision * quality_precision

    return precision


polls["precision"] = calculate_poll_precision(polls)

# Normalize precision into weights
polls["weight"] = polls["precision"] / polls["precision"].sum()

# ---- Posterior estimate of latent support ----
latent_support = np.sum(
    polls["adjusted_support"] * polls["weight"]
)

# ---- Uncertainty estimate (important upgrade) ----
support_variance = np.sum(
    polls["weight"] * (polls["adjusted_support"] - latent_support) ** 2
)

support_std = np.sqrt(support_variance)
##############################################################
# SECTION 7: FULL MODEL INTERPRETATION & DIAGNOSTIC SUMMARY
##############################################################

print("\n==============================")
print("ELECTION MODEL SUMMARY (FULL PIPELINE)")
print("==============================\n")


# -----------------------------
# SECTION 2: POLLS (MEASUREMENT LAYER)
# -----------------------------
print("🟦 SECTION 2: POLLING DATA")
print("------------------------------")
print("Weighted poll support:", round(weighted_support * 100, 2), "%")
print("Interpretation: Best noisy estimate of current public opinion.\n")


# -----------------------------
# SECTION 3: MOMENTUM (DIRECTIONAL SIGNAL)
# -----------------------------
print("🟨 SECTION 3: MOMENTUM (TREND)")
print("------------------------------")
print("Momentum trend:", round(recent_poll_trend * 100, 4), "%")

if recent_poll_trend > 0:
    print("Interpretation: Support is trending upward over time.\n")
else:
    print("Interpretation: Support is trending downward over time.\n")


# -----------------------------
# SECTION 4: UNDECIDED VOTERS (FLOW MODEL)
# -----------------------------
print("🟧 SECTION 4: UNDECIDED VOTERS")
print("------------------------------")
print("Undecided pool size:", round(undecided * 100, 2), "%")
print("Expected undecided split:", round(expected_split * 100, 2), "%")
print("Undecided gain contribution:", round(undecided_gain * 100, 2), "%")

print("Interpretation: Small voter group redistributes based on current competitiveness.\n")


# -----------------------------
# SECTION 5: LATENT SUPPORT (STRUCTURAL TRUTH ESTIMATE)
# -----------------------------
print("🟪 SECTION 5: LATENT SUPPORT ESTIMATION")
print("------------------------------")
print("Latent support (poll-based truth estimate):", round(latent_support * 100, 2), "%")
print("Uncertainty (standard deviation):", round(support_std * 100, 2), "%")

print("Interpretation:")
print("- This is the best estimate of true underlying support.")
print("- Uncertainty reflects disagreement + noise across polls.\n")


# -----------------------------
# SECTION 6: CAMPAIGN EFFECTS (CAUSAL SHOCK MODEL)
# -----------------------------
print("🟥 SECTION 6: CAMPAIGN EFFECTS (SCALED + STRUCTURED)")
print("------------------------------")

print("Fundraising effect (scaled):", round(fund_effect * 100, 4), "%")
print("Field campaign effect (scaled):", round(field_effect * 100, 4), "%")
print("Momentum effect (scaled):", round(momentum_effect * 100, 4), "%")
print("Early vote effect (scaled):", round(early_vote_boost * 100, 4), "%")

print("\nTotal campaign adjustment:", round(total_campaign_effect * 100, 4), "%")

print("Interpretation:")
print("- Effects shown here are ALREADY adjusted by sensitivity.")
print("- Sensitivity reflects how volatile/competitive the race is.")
print("- Campaign impact shrinks in stable races and grows in uncertain ones.\n")


# -----------------------------
# FINAL OUTPUT (INTEGRATED SYSTEM RESULT)
# -----------------------------
print("🟩 FINAL RESULT (INTEGRATED MODEL)")
print("------------------------------")

print("Final estimated support:", round(base_support * 100, 2), "%")

win_prob = 1 / (1 + np.exp(-((base_support - 0.5) * 20)))

print("Estimated win probability:", round(win_prob * 100, 2), "%")

print("\nInterpretation:")
print("- Final support combines: polling + voter behavior + campaign effects.")
print("- Win probability converts margin into likelihood of victory.")
print("- District model (Section 8) will now spatially redistribute this signal.\n")

print("==============================\n")
##############################################################
# SECTION 8: 10-DISTRICT DYNAMIC SUPPORT MODEL
# (STRUCTURED, SCALING-BASED, READY FOR MONTE CARLO)
##############################################################

# Ensure all core signals are in fraction space
support_std = np.clip(support_std, 0.005, 0.05)
recent_poll_trend = np.clip(recent_poll_trend, -0.01, 0.01)
# -----------------------------
# CONFIGURATION
# -----------------------------
NUM_DISTRICTS = 10

# Split national latent support into district baselines
# (introduces heterogeneity instead of single national number)

np.random.seed(42)

district_ids = np.arange(NUM_DISTRICTS)

# -----------------------------
# 1. CREATE DISTRICT BASELINES
# -----------------------------
# National support is center; districts vary around it

district_baseline = latent_support + np.random.normal(
    0, 0.03, NUM_DISTRICTS
)

district_baseline = np.clip(district_baseline, 0.01, 0.99)


# -----------------------------
# 2. DISTRICT UNCERTAINTY (HETEROGENEOUS NOISE)
# -----------------------------
# Some districts are stable, others volatile

district_uncertainty = np.random.uniform(
    support_std * 0.8,
    support_std * 1.5,
    NUM_DISTRICTS
)


# -----------------------------
# 3. DISTRICT SENSITIVITY (LEARNED VIA NORMALIZATION IDEA)
# -----------------------------
# Instead of hardcoding coefficients,
# we scale sensitivity based on local volatility

district_sensitivity = 1 / (1 + district_uncertainty * 25)


# -----------------------------
# 4. ALLOCATE NATIONAL EFFECTS TO DISTRICTS
# -----------------------------
# Campaign effects influence all districts,
# but impact differs by sensitivity

fund_effect_d = fund_effect * district_sensitivity
field_effect_d = field_effect * district_sensitivity
momentum_effect_d = momentum_effect * district_sensitivity
early_vote_d = early_vote_boost * district_sensitivity


# -----------------------------
# 5. APPLY DISTRICT-LEVEL SHOCKS
# -----------------------------
# Each district has its own local variation

local_shocks = np.random.normal(
    0,
    district_uncertainty,
    NUM_DISTRICTS
)


# -----------------------------
# 6. COMPUTE FINAL DISTRICT SUPPORT
# -----------------------------
district_support = (
    district_baseline
    + fund_effect_d
    + field_effect_d
    + momentum_effect_d
    + early_vote_d
    + local_shocks
)

district_support = np.clip(district_support, 0.01, 0.99)


# -----------------------------
# 7. SUMMARY TABLE
# -----------------------------
district_df = pd.DataFrame({
    "district": district_ids,
    "baseline_support": district_baseline,
    "uncertainty": district_uncertainty,
    "sensitivity": district_sensitivity,
    "final_support": district_support
})


# -----------------------------
# 8. NATIONAL RESULT FROM DISTRICTS
# -----------------------------
national_support_from_districts = district_support.mean()

# simple win condition proxy (for now)
district_wins = (district_support > 0.5).sum()


# -----------------------------
# OUTPUT
# -----------------------------
print("\n==============================")
print("SECTION 8: 10-DISTRICT MODEL")
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
print("- Each district behaves like a semi-independent system.")
print("- Campaign effects are scaled by local sensitivity.")
print("- Local noise creates realistic divergence between regions.")
print("- National result is now emergent, not imposed.")
print("==============================\n")
##############################################################
# SECTION 9: MONTE CARLO ELECTION SIMULATION
# (NO CHANGES REQUIRED TO SECTIONS 1–8)
##############################################################

import numpy as np

NUM_SIMULATIONS = 10000

# ------------------------------------------------------------
# STORAGE
# ------------------------------------------------------------
national_results = []
district_wins_results = []

# ------------------------------------------------------------
# MONTE CARLO LOOP
# ------------------------------------------------------------
for _ in range(NUM_SIMULATIONS):

    # -----------------------------
    # STAGE 1: PERTURB NATIONAL SIGNALS
    # -----------------------------

    sim_latent_support = np.clip(
        latent_support + np.random.normal(0, support_std),
        0.01, 0.99
    )

    sim_base_support = np.clip(
        base_support + np.random.normal(0, support_std * 1.2),
        0.01, 0.99
    )

    # recompute campaign uncertainty effect
    sim_campaign_noise = np.random.normal(0, 0.002)


    # -----------------------------
    # STAGE 2: GENERATE DISTRICTS (FROM SECTION 8 STRUCTURE)
    # -----------------------------

    sim_districts = sim_base_support + np.random.normal(
        0, support_std, NUM_DISTRICTS
    )

    sim_districts = np.clip(sim_districts, 0.01, 0.99)


    # -----------------------------
    # STAGE 3: APPLY DISTRICT NOISE (LOCAL VARIATION)
    # -----------------------------

    sim_districts += np.random.normal(
        0,
        support_std * 0.5,
        NUM_DISTRICTS
    )

    sim_districts = np.clip(sim_districts, 0.01, 0.99)


    # -----------------------------
    # STAGE 4: DETERMINE WINS
    # -----------------------------

    district_wins = np.sum(sim_districts > 0.5)

    national_result = np.mean(sim_districts)

    # optional campaign shock effect
    national_result += sim_campaign_noise

    national_results.append(national_result)
    district_wins_results.append(district_wins)


# ------------------------------------------------------------
# SUMMARY STATISTICS
# ------------------------------------------------------------

national_results = np.array(national_results)
district_wins_results = np.array(district_wins_results)

win_probability = np.mean(national_results > 0.5)

avg_district_wins = np.mean(district_wins_results)

std_national = np.std(national_results)

print("\n==============================")
print("SECTION 9: MONTE CARLO RESULTS")
print("==============================\n")

print("Win probability:", round(win_probability * 100, 2), "%")
print("Average national support:", round(np.mean(national_results) * 100, 2), "%")
print("National uncertainty (std dev):", round(std_national * 100, 2), "%")

print("\nAverage district wins:", round(avg_district_wins, 2), "/ 10")

print("\nInterpretation:")
print("- Each simulation represents a possible election outcome.")
print("- Districts fluctuate due to local + national noise.")
print("- Final results form a probability distribution, not a single value.")
print("- Win probability is the fraction of simulated wins.")
print("==============================\n")
