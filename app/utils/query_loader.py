from pathlib import Path


def load_query(filename: str) -> str:
    """
    Load a SQL query from the data/sql_queries folder.
    
    Args:
        filename (str): Name of the .sql file (e.g., "lottery_cleaning.sql")
    
    Returns:
        str: SQL query as a string
    """
    sql_path = Path("data/sql_queries") / filename
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")
    
    with open(sql_path, "r", encoding="utf-8") as f:
        return f.read()
