from utils.debugging.debug_dns import *  # DNS debugging utility
from utils.debugging.debug_env import *  
from utils.debugging.debug_pk import *  

def debug_infrastructure():
    
    print("=== DEBUGGING INFRASTRUCTURE ===")

    debug_dns()
    debug_env_vars()
    debug_private_key()