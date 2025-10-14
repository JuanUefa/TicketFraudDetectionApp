import snowflake.connector
from contextlib import contextmanager
from snowflake.snowpark import Session

from utils.input_output_utils.env_loader import (
    SNOWFLAKE_USER,
    SNOWFLAKE_AUTHENTICATOR,
    SNOWFLAKE_ACCOUNT,
    SNOWFLAKE_WAREHOUSE,
    SNOWFLAKE_DATABASE,
    SNOWFLAKE_SCHEMA,
)
 
 
@contextmanager
def get_snowflake_connection():
    """
    Context manager that yields a Snowflake connection and ensures it closes.
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

    
def get_snowflake_session():
    connection_parameters = {"account": "a40937.eu-central-1", "user": "jpra@uefa.ch", "role":"ML_SPECIALIST_ROLE", "database":"UEFA_DEV_DWH", "schema":"ML_SANDBOX", "warehouse": "DA_ML_VWH", "authenticator":"externalbrowser"}
    session = Session.builder.configs(connection_parameters).create()

    return session
 
 
def run_query(query: str):
    """
    Executes a SQL query and returns (rows, columns).
    """
    with get_snowflake_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            columns = [col[0] for col in cur.description]
    return rows, columns