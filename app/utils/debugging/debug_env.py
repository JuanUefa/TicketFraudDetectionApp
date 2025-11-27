import os

def debug_env_vars():
 
    print("[ENV DEBUG] SNOWFLAKE_ACCOUNT:", os.getenv("SNOWFLAKE_ACCOUNT"))
    print("[ENV DEBUG] SNOWFLAKE_USER:", os.getenv("SNOWFLAKE_USER"))
    print("[ENV DEBUG] SNOWFLAKE_AUTHENTICATOR:", os.getenv("SNOWFLAKE_AUTHENTICATOR"))
    print("[ENV DEBUG] PRIVATE_KEY_PATH:", os.getenv("PRIVATE_KEY_PATH"))

