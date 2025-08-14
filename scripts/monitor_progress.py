#!/usr/bin/env python3
"""
Live Progress Monitor for RML Ultimate Training
Shows real-time progress updates
"""

import os
import time
import psutil
import subprocess
import threading
from datetime import datetime

def get_process_info(pid):
    """Get process information"""
    try:
        process = psutil.Process(pid)
        return {
            'cpu_percent': process.cpu_percent(),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'status': process.status(),
            'create_time': process.create_time()
        }
    except:
        return None

def monitor_training_progress():
    """Monitor training progress in real-time"""
    
    print("📊 RML Ultimate Training - LIVE PROGRESS MONITOR")
    print("="*60)
    print("🔄 Press Ctrl+C to stop monitoring")
    print("="*60)
    
    # Find the training process
    training_pid = None
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'rml_ultimate_trainer.py' in ' '.join(proc.info['cmdline'] or []):
                training_pid = proc.info['pid']
                break
        except:
            continue
    
    if not training_pid:
        print("❌ No training process found!")
        return
    
    print(f"🟢 Training Process ID: {training_pid}")
    print(f"⏰ Started: {datetime.fromtimestamp(psutil.Process(training_pid).create_time()).strftime('%H:%M:%S')}")
    print()
    
    start_time = time.time()
    
    try:
        while True:
            # Get process info
            proc_info = get_process_info(training_pid)
            if not proc_info:
                print("❌ Training process stopped!")
                break
            
            # Calculate runtime
            runtime = time.time() - start_time
            hours = int(runtime // 3600)
            minutes = int((runtime % 3600) // 60)
            seconds = int(runtime % 60)
            
            # Get system memory
            system_memory = psutil.virtual_memory()
            
            # Clear screen and show progress
            os.system('clear')
            print("📊 RML Ultimate Training - LIVE PROGRESS MONITOR")
            print("="*60)
            print(f"🟢 Status: RUNNING (PID: {training_pid})")
            print(f"⏱️ Runtime: {hours:02d}:{minutes:02d}:{seconds:02d}")
            print(f"📊 CPU Usage: {proc_info['cpu_percent']:.1f}%")
            print(f"💾 Process Memory: {proc_info['memory_mb']:.1f} MB")
            print(f"🖥️ System Memory: {system_memory.percent:.1f}% ({system_memory.available / 1024**3:.1f}GB free)")
            print(f"📈 Process Status: {proc_info['status']}")
            print()
            
            # Check for log updates
            log_file = "/Volumes/MEGA/R-LLM-ultimate-trained/training.log"
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            print("📝 Recent Log Entries:")
                            print("-" * 40)
                            for line in lines[-5:]:  # Last 5 lines
                                print(f"   {line.strip()}")
                except:
                    pass
            
            # Check for progress file
            progress_file = "/Volumes/MEGA/training_progress.pkl"
            if os.path.exists(progress_file):
                print("\n📂 Progress file found - training can be resumed")
            
            # Check MEGA drive space
            if os.path.exists("/Volumes/MEGA"):
                try:
                    disk_usage = psutil.disk_usage("/Volumes/MEGA")
                    print(f"\n💾 MEGA Drive: {disk_usage.free / 1024**3:.1f}GB free")
                except:
                    pass
            
            print("\n🔄 Monitoring... (Press Ctrl+C to stop)")
            time.sleep(2)  # Update every 2 seconds
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

def show_quick_status():
    """Show quick status without continuous monitoring"""
    
    print("📊 RML Ultimate Training - QUICK STATUS")
    print("="*40)
    
    # Find training process
    training_pid = None
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'rml_ultimate_trainer.py' in ' '.join(proc.info['cmdline'] or []):
                training_pid = proc.info['pid']
                break
        except:
            continue
    
    if training_pid:
        proc_info = get_process_info(training_pid)
        if proc_info:
            runtime = time.time() - proc_info['create_time']
            hours = int(runtime // 3600)
            minutes = int((runtime % 3600) // 60)
            
            print(f"🟢 Status: RUNNING (PID: {training_pid})")
            print(f"⏱️ Runtime: {hours}h {minutes}m")
            print(f"📊 CPU: {proc_info['cpu_percent']:.1f}%")
            print(f"💾 Memory: {proc_info['memory_mb']:.1f} MB")
            print(f"📈 Status: {proc_info['status']}")
        else:
            print("🔴 Status: Process not responding")
    else:
        print("🔴 Status: No training process found")
    
    # Check log file
    log_file = "/Volumes/MEGA/R-LLM-ultimate-trained/training.log"
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    print(f"\n📝 Last log entry: {lines[-1].strip()}")
        except:
            pass

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "live":
        monitor_training_progress()
    else:
        show_quick_status() 