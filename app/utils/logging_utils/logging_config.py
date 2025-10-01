import logging
import os
 
def setup_logging(log_file: str, level: str = "INFO"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
 
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler()
        ]
    )
