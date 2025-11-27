from pathlib import Path
from utils.input_output_utils.env_loader import PRIVATE_KEY_PATH

def debug_private_key():
 
    key_path = Path(PRIVATE_KEY_PATH)
    if not key_path.exists():
        print(f"[ERROR] RSA key not found at: {key_path.resolve()}")
    else:
        print(f"[DEBUG] RSA key loaded from: {key_path.resolve()}")