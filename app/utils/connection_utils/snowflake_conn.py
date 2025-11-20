import snowflake.connector
from contextlib import contextmanager
from snowflake.snowpark import Session
 
# Import environment variables (these should be defined in env_loader)
from utils.input_output_utils.env_loader import *
 
# -----------------------------------------------------------------------------
# Classic Snowflake Connection (for raw SQL execution)
# -----------------------------------------------------------------------------
 
@contextmanager
def get_snowflake_connection():
    """
    Context manager that yields a Snowflake connection using the classic 
    Python connector. Ensures the connection is properly closed after use.
    """
    conn = snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        authenticator=SNOWFLAKE_AUTHENTICATOR,
        account=SNOWFLAKE_ACCOUNT,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA,
    )
    try:
        yield conn
    finally:
        conn.close()
 
# -----------------------------------------------------------------------------
# Snowpark Session (for DataFrames and SPCS apps)
# -----------------------------------------------------------------------------
 
def get_snowflake_session():
    """
    Creates and returns a Snowpark Session object using the environment 
    variables defined in env_loader. This is the preferred method when working 
    with Snowpark DataFrames in SPCS or data pipelines.
    """
    connection_parameters = {
        "account": SNOWFLAKE_ACCOUNT,
        "user": SNOWFLAKE_USER,
        "role": SNOWFLAKE_ROLE,
        "database": SNOWFLAKE_DATABASE,
        "schema": SNOWFLAKE_SCHEMA,
        "warehouse": SNOWFLAKE_WAREHOUSE,
        "authenticator": SNOWFLAKE_AUTHENTICATOR
    }
 
    session = Session.builder.configs(connection_parameters).create()
    return session
 
# -----------------------------------------------------------------------------
# Execute a SQL Query (with results)
# -----------------------------------------------------------------------------
 
def run_query(query: str):
    """
    Executes a SQL query using the classic connector and returns:
    - rows: list of tuples with the result data
    - columns: list of column names
    Useful for simple queries or validation, without needing Snowpark.
    """
    with get_snowflake_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            columns = [col[0] for col in cur.description]
    return rows, columns