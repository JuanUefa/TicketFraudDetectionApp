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
    "inv_semantic_score": "high",
    "domain_freq": "low",
    "browser_suspicious": "binary",
}
 
# Relative feature importance weights (sum ≤ 1)
FEATURE_WEIGHTS = {
    "username_digit_count": 0.1,
    "username_special_count": 0.1,
    "username_entropy": 0.15,
    "inv_semantic_score": 0.25,
    "domain_freq": 0.2,
    "browser_suspicious": 0.2,
}