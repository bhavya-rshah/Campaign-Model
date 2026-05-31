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
# SECTION 3: MOMENTUM (FULLY DATA-DRIVEN)
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
