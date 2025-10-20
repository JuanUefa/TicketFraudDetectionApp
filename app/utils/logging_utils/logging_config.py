import logging
import sys
 
def setup_logging(log_file="app.log", level=logging.INFO):
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
 
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    console_handler.stream.reconfigure(encoding="utf-8") 
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
 
    logging.basicConfig(
        level=level,
        handlers=[console_handler, file_handler],
        force=True
    )