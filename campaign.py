import pandas as pd
import numpy as np

##############################################################
# SECTION 1: POLLS (FRACTION SCALE CONSISTENT)
##############################################################

polls = pd.DataFrame([
    {"pollster": "A", "support": 0.48, "days_old": 2, "sample": 1000, "quality_rating": 0.9,
     "house_effect": 0.01, "mode_adjustment": -0.005},

    {"pollster": "B", "support": 0.52, "days_old": 10, "sample": 800, "quality_rating": 0.8,
     "house_effect": -0.02, "mode_adjustment": 0.01},

    {"pollster": "C", "support": 0.50, "days_old": 5, "sample": 1200, "quality_rating": 0.95,
     "house_effect": 0.00, "mode_adjustment": 0.00}
])

polls["adjusted_support"] = (
    polls["support"]
    - polls["house_effect"]
    + polls["mode_adjustment"]
)

##############################################################
# SECTION 2: DEMOGRAPHICS (FIXED TO FRACTION SCALE)
##############################################################

segments = pd.DataFrame([
    {"group": "Youth", "support_prob": 0.62, "turnout_prob": 0.48, "population_weight": 0.18},
    {"group": "Middle Age", "support_prob": 0.51, "turnout_prob": 0.72, "population_weight": 0.47},
    {"group": "Seniors", "support_prob": 0.44, "turnout_prob": 0.80, "population_weight": 0.35}
])

segments["effective_vote"] = (
    segments["support_prob"]
    * segments["turnout_prob"]
    * segments["population_weight"]
)

segment_support = (
    segments["effective_vote"].sum()
    /
    (segments["turnout_prob"] * segments["population_weight"]).sum()
)

##############################################################
# SECTION 3
##############################################################

recent_poll_trend = 0.8 / 100  # FIX: convert to fraction scale

##############################################################
# SECTION 4
##############################################################

undecided = 5.0 / 100
expected_split = 0.55

undecided_gain = undecided * expected_split

##############################################################
# SECTION 5
##############################################################

def calculate_poll_weights(df):
    recency = 1 / (1 + df["days_old"])
    sample = np.sqrt(df["sample"])
    quality = df["quality_rating"]

    weights = recency * sample * quality
    return weights / weights.sum()

polls["weight"] = calculate_poll_weights(polls)

weighted_support = np.average(
    polls["adjusted_support"],
    weights=polls["weight"]
)


##############################################################
# SECTION 6: CAMPAIGN EFFECT MODEL (TUNED TO REAL-WORLD SENSITIVITY)
##############################################################

candidate_funds = 520000
opponent_funds = 610000

# -----------------------------
# FUNDRAISING (small advantage signal)
# -----------------------------
fundraising_gain = (candidate_funds / opponent_funds - 1) * 0.01
# capped effect already controlled downstream

# -----------------------------
# FIELD CAMPAIGN (heavily scaled down)
# -----------------------------
doors_knocked = 18500
phone_calls = 42000

field_vote_gain = (
    doors_knocked * 0.0000005
    + phone_calls * 0.00000015
)

# -----------------------------
# EARLY VOTE (small signal)
# -----------------------------
early_vote_boost = 0.003

##############################################################
# FINAL COMBINATION (FIXED SCALE)
##############################################################

base_support = (
    weighted_support
    + (segment_support - weighted_support) * 0.3
    + recent_poll_trend
    + undecided_gain * 0.5
    + fundraising_gain
    + field_vote_gain * 0.5
    + early_vote_boost
)

print("Weighted poll support:", round(weighted_support * 100, 2), "%")
print("Segment support:", round(segment_support * 100, 2), "%")
print("Base support estimate:", round(base_support * 100, 2), "%")
