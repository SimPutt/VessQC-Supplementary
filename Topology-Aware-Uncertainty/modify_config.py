#!/usr/bin/env python3
"""
Helper script to modify JSON config files with datetime stamps. Thus the results are stored in a timestamped directory.
"""

import json
import sys
from datetime import datetime
import os

def modify_segmentation_config(config_path, datetime_stamp):
    """Modify segmentation config.json file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Replace "results/seg" with "results/datetime_stamp/seg"
        if "output_folder" in config:
            config["output_folder"] = f"../results/{datetime_stamp}/seg" 
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"Successfully modified {config_path}")
        return True
    except Exception as e:
        print(f"Error modifying {config_path}: {e}")
        return False

def modify_uncertainty_config(config_path, datetime_stamp):
    """Modify uncertainty config.json file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Replace "../results/seg" with "../results/datetime_stamp/seg"
        if "npy_seg_dir" in config:
            config["npy_seg_dir"] = f"../results/{datetime_stamp}/seg"
        
        # Replace "../results/skel_uncertainty" with "../results/datetime_stamp/skel_uncertainty"
        if "output_folder" in config:
            config["output_folder"] = f"../results/{datetime_stamp}/skel_uncertainty"
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"Successfully modified {config_path}")
        return True
    except Exception as e:
        print(f"Error modifying {config_path}: {e}")
        return False

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 modify_config.py <action> <config_path> <datetime_stamp>")
        print("Actions: segmentation, uncertainty")
        sys.exit(1)
    
    action = sys.argv[1]
    config_path = sys.argv[2]
    datetime_stamp = sys.argv[3]
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    if action == "segmentation":
        success = modify_segmentation_config(config_path, datetime_stamp)
    elif action == "uncertainty":
        success = modify_uncertainty_config(config_path, datetime_stamp)
    else:
        print(f"Unknown action: {action}")
        sys.exit(1)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
