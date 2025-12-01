"""
fraud_score_schemas.py
Defines schema-like constants for fraud scoring parameters.
"""
 
# Risk direction:
#   "high"   = higher value → higher risk
#   "low"    = lower value → higher risk
#   "binary" = direct 0/1 risk flag
RISK_DIRECTION = {
    "username_digit_count": "high",
    "username_special_count": "high",
    "username_entropy": "high",
    "username_numeric_seq_count": "high", 
    "inv_semantic_score": "high",
    "domain_freq": "low",
    "total_apps": "high",
    "browser_suspicious": "binary",
    "country_language_mismatch": "binary",
}
 
# Relative feature importance weights (sum ≤ 1)
FEATURE_WEIGHTS = {
    "username_digit_count": .2,
    "username_special_count": .2,
    "username_entropy": .2,
    "username_numeric_seq_count": .2, 
    "inv_semantic_score": .2,
    "domain_freq": .2,
    "total_apps": .2,
    "browser_suspicious": .2,
    "country_language_mismatch": .2,
}