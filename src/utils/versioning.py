import os
from datetime import datetime

def get_next_version(base_name):

    if not os.path.exists(base_name):
        return base_name
    
    name, ext = os.path.splitext(base_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_name = f"{name}_{timestamp}{ext}"
    
    counter = 1
    while os.path.exists(new_name):
        new_name = f"{name}_{timestamp}_{counter}{ext}"
        counter += 1
        
    return new_name
