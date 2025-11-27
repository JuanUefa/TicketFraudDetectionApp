import os
import socket
 
def debug_dns():
    account = os.getenv("SNOWFLAKE_ACCOUNT")
    full_host = f"{account}.snowflakecomputing.com"
 
    print(f"[DEBUG] Attempting DNS resolution for: {full_host}")
    try:
        ip = socket.gethostbyname(full_host)
        print(f"[DEBUG] SUCCESS: Resolved '{full_host}' to {ip}")
    except Exception as e:
        print(f"[ERROR] DNS resolution failed for '{full_host}': {e}")