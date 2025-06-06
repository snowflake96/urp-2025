#!/usr/bin/env python3
"""
Monitor the progress of the 3D numerical analysis running in the background
"""

import time
import os
from pathlib import Path
import subprocess
import json

def monitor_analysis_progress():
    """Monitor the ongoing analysis progress"""
    
    print("🔬 MONITORING 3D NUMERICAL ANALYSIS WITH PYANSYS")
    print("=" * 70)
    print("Analysis Type: Real ANSYS MAPDL 3D Numerical Analysis")
    print("CPU Cores: 8 parallel workers")
    print("Target: All 168 patient scans (68 boundary condition files)")
    print("=" * 70)
    
    results_dir = Path("/home/jiwoo/urp/data/uan/realistic_numerical_results")
    
    start_time = time.time()
    
    while True:
        try:
            # Count completed analysis files
            if results_dir.exists():
                completed_files = list(results_dir.glob("patient_*/*_realistic_numerical_results.json"))
                completed_count = len(completed_files)
            else:
                completed_count = 0
            
            # Check for any log files
            log_files = list(Path(".").glob("*.log"))
            
            elapsed_time = time.time() - start_time
            
            print(f"\r⏱️  Time: {elapsed_time/60:.1f}min | ✅ Completed: {completed_count}/68 cases | 📊 Progress: {completed_count/68*100:.1f}%", end="", flush=True)
            
            # Check if analysis is complete
            if completed_count >= 68:
                print(f"\n🎉 ANALYSIS COMPLETE! All {completed_count} cases processed in {elapsed_time/60:.1f} minutes")
                break
            
            # Brief progress summary every 30 seconds
            if int(elapsed_time) % 30 == 0 and elapsed_time > 1:
                print(f"\n📈 Progress Update: {completed_count}/68 cases completed ({completed_count/68*100:.1f}%)")
                if completed_count > 0:
                    avg_time_per_case = elapsed_time / completed_count
                    remaining_cases = 68 - completed_count
                    estimated_remaining = remaining_cases * avg_time_per_case
                    print(f"⏰ Estimated time remaining: {estimated_remaining/60:.1f} minutes")
                print("-" * 50)
            
            time.sleep(2)  # Check every 2 seconds
            
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped by user")
            print(f"Analysis is still running in background. Current progress: {completed_count}/68")
            break
        except Exception as e:
            print(f"\n❌ Error monitoring: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_analysis_progress() 