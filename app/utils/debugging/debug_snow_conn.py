import snowflake.connector
import os


def debug_snowflake_connection():
    """
    Attempts to connect to Snowflake using environment variables.
    Prints success or error messages for debugging.
    """
 
    ctx = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        authenticator=os.getenv("SNOWFLAKE_AUTHENTICATOR"),
        private_key_file=os.getenv("PRIVATE_KEY_PATH"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        role=os.getenv("SNOWFLAKE_ROLE"),
    )
    print("[DEBUG] Connected to Snowflake")