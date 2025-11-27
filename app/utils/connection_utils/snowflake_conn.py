import snowflake.connector
from contextlib import contextmanager
from snowflake.snowpark import Session
from cryptography.hazmat.primitives import serialization
import os
 
# Load environment variables
from utils.input_output_utils.env_loader import *
 
# -----------------------------------------------------------------------------
# Load RSA Private Key (for SNOWFLAKE_JWT)
# -----------------------------------------------------------------------------
 
def load_private_key():
    """
    Loads the RSA private key used for JWT authentication.
    Requires PRIVATE_KEY_PATH to be set in .env.
    """
    if not PRIVATE_KEY_PATH:
        raise ValueError("PRIVATE_KEY_PATH is not set in .env")
 
    if not os.path.exists(PRIVATE_KEY_PATH):
        raise FileNotFoundError(f"Private key not found at: {PRIVATE_KEY_PATH}")
 
    with open(PRIVATE_KEY_PATH, "rb") as key_file:
        return serialization.load_pem_private_key(
            key_file.read(),
            password=None,
        )
 
# -----------------------------------------------------------------------------
# Classic Snowflake Connector (for raw SQL)
# -----------------------------------------------------------------------------
 
@contextmanager
def get_snowflake_connection():
    """
    Returns a Snowflake connection using JWT + private key.
    Automatically closes the connection after use.
    """
    private_key = load_private_key()
 
    conn = snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        authenticator=SNOWFLAKE_AUTHENTICATOR,  # SNOWFLAKE_JWT
        account=SNOWFLAKE_ACCOUNT,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA,
        role=SNOWFLAKE_ROLE,
        host=SNOWFLAKE_HOST,
        private_key=private_key,                # <-- Required for JWT
    )
    try:
        yield conn
    finally:
        conn.close()
 
# -----------------------------------------------------------------------------
# Snowpark Session
# -----------------------------------------------------------------------------
 
def get_snowflake_session():
    """
    Creates a Snowpark Session using JWT + private key.
    """
    private_key = load_private_key()
 
    connection_parameters = {
        "account": SNOWFLAKE_ACCOUNT,
        "user": SNOWFLAKE_USER,
        "role": SNOWFLAKE_ROLE,
        "database": SNOWFLAKE_DATABASE,
        "schema": SNOWFLAKE_SCHEMA,
        "warehouse": SNOWFLAKE_WAREHOUSE,
        "authenticator": SNOWFLAKE_AUTHENTICATOR,
        "host": SNOWFLAKE_HOST,
        "private_key": private_key,   # <-- Required for JWT
    }
 
    return Session.builder.configs(connection_parameters).create()
 
# -----------------------------------------------------------------------------
# Execute SQL and return results
# -----------------------------------------------------------------------------
 

## Classic connector
"""def run_query(query: str):
    #Executes SQL using the classic connector.
    #Returns rows and column names.
    with get_snowflake_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            columns = [col[0] for col in cur.description]
    return rows, columns"""


## Snowpark Connector
 
def run_query(query: str):

    #Executes a SQL query using Snowpark and returns:
    #- rows: list of tuples
    #- columns: list of column names (lowercase)
    
    session = get_snowflake_session()
 
    try:

        # Ejecuta la query y convierte a pandas
        snowpark_df = session.sql(query)
        pandas_df = snowpark_df.to_pandas()
        pandas_df.columns = [col.strip().lower() for col in pandas_df.columns]
 
        # Columnas en minúsculas
        columns = [col.lower() for col in pandas_df.columns]
        # Filas como tuplas
        rows = [tuple(row) for row in pandas_df.itertuples(index=False, name=None)]
 
        return rows, columns
 
    except Exception as e:
        print(f"[ERROR] Snowpark query failed: {e}")
        raise