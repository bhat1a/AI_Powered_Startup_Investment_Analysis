import logging
import os
from datetime import datetime

def get_logger(name: str) -> logging.Logger:
    
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")
    
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger  # avoid duplicate handlers on reuse
    
    logger.setLevel(logging.DEBUG)
    
    # File handler — writes everything DEBUG and above
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler — writes INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger