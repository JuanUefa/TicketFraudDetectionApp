from dotenv import load_dotenv
import os
 
load_dotenv()

# Logging config
LOGS_FILE = os.getenv("LOGS_FILE", "logs/app.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Data paths
INPUT_PATH = os.getenv("INPUT_PATH", "data/input")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "data/output")

# Snowflake config
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_AUTHENTICATOR = os.getenv("SNOWFLAKE_AUTHENTICATOR", "externalbrowser")
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")

GET_SAMPLE_DATA = os.getenv("GET_SAMPLE_DATA")

SERVICE_HOST = os.getenv("SERVICE_HOST")
SERVICE_PORT = int(os.getenv("SERVICE_PORT"))
PLOTS_PATH = os.getenv("PLOTS_PATH")
