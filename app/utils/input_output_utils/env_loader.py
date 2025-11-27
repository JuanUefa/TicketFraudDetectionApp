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
SNOWFLAKE_AUTHENTICATOR = os.getenv("SNOWFLAKE_AUTHENTICATOR")
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")
SNOWFLAKE_ROLE = os.getenv("SNOWFLAKE_ROLE")
SNOWFLAKE_HOST = os.getenv("SNOWFLAKE_HOST")
PRIVATE_KEY_PATH = os.getenv("PRIVATE_KEY_PATH")
 
# Optional OAuth token (if present, overrides other login methods)
SNOWFLAKE_OAUTH_TOKEN = os.getenv("SNOWFLAKE_OAUTH_TOKEN") 
 