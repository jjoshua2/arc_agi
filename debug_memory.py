#!/usr/bin/env python3
"""
Quick memory debugging for fast sweep issues.
Run this to check system memory and set conservative settings.
"""

import os
import sys

def check_memory():
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"Total memory: {mem.total / (1024**3):.1f} GB")
        print(f"Available: {mem.available / (1024**3):.1f} GB")
        print(f"Used: {mem.used / (1024**3):.1f} GB")
        print(f"Percentage used: {mem.percent:.1f}%")
        return mem.available / (1024**3)
    except ImportError:
        print("psutil not available, cannot check memory")
        return None

def recommend_settings():
    available_gb = check_memory()
    
    print("\n=== FAST SWEEP MEMORY SETTINGS ===")
    
    if available_gb is None:
        print("Cannot determine available memory")
        print("Conservative settings:")
        print("export ARC_FAST_SWEEP_WORKERS=1")
        print("export ARC_FAST_SWEEP_BATCH_SIZE=5")
        return
    
    if available_gb > 20:
        print("High memory system - can use aggressive settings:")
        print("export ARC_FAST_SWEEP_WORKERS=4")
        print("export ARC_FAST_SWEEP_BATCH_SIZE=50")
    elif available_gb > 10:
        print("Medium memory system - balanced settings:")
        print("export ARC_FAST_SWEEP_WORKERS=3") 
        print("export ARC_FAST_SWEEP_BATCH_SIZE=25")
    elif available_gb > 5:
        print("Low memory system - conservative settings:")
        print("export ARC_FAST_SWEEP_WORKERS=2")
        print("export ARC_FAST_SWEEP_BATCH_SIZE=10")
    else:
        print("Very low memory - disable fast sweep:")
        print("export ARC_FAST_SWEEP=0")
    
    print("\nIf memory issues persist:")
    print("export ARC_FAST_SWEEP=0  # Disable optimization")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "disable":
        print("Disabling fast sweep...")
        os.environ["ARC_FAST_SWEEP"] = "0"
        print("Set ARC_FAST_SWEEP=0")
    else:
        recommend_settings()