from dotenv import load_dotenv
import os
 
load_dotenv()
 
# -----------------------------------------------------------------------------
# Logging Config
# -----------------------------------------------------------------------------
LOGS_FILE = os.getenv("LOGS_FILE", "logs/app.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
 
# -----------------------------------------------------------------------------
# Data Paths
# -----------------------------------------------------------------------------
INPUT_PATH = os.getenv("INPUT_PATH", "data/input")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "data/output")
PLOTS_PATH = os.getenv("PLOTS_PATH", "data/output/plots")
 
# -----------------------------------------------------------------------------
# Snowflake Config
# -----------------------------------------------------------------------------
 
# Auth method will be dynamically chosen (token or username)
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_AUTHENTICATOR = os.getenv("SNOWFLAKE_AUTHENTICATOR", "externalbrowser")
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")
SNOWFLAKE_ROLE = os.getenv("SNOWFLAKE_ROLE")
 
# Optional OAuth token (if present, overrides other login methods)
SNOWFLAKE_OAUTH_TOKEN = os.getenv("SNOWFLAKE_OAUTH_TOKEN") 
 
# -----------------------------------------------------------------------------
# App Behavior
# -----------------------------------------------------------------------------
GET_SAMPLE_DATA = os.getenv("GET_SAMPLE_DATA")
SERVICE_HOST = os.getenv("SERVICE_HOST", "0.0.0.0")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", 8000))