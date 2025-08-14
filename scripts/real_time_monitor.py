#!/usr/bin/env python3
"""
Real-time Monitor for RML Training
Shows actual file processing activity
"""

import os
import time
import psutil
import subprocess
from datetime import datetime

def monitor_training_activity():
    """Monitor actual training activity in real-time"""
    
    print("📊 RML Training - REAL-TIME ACTIVITY MONITOR")
    print("="*60)
    print("🔄 Press Ctrl+C to stop monitoring")
    print("="*60)
    
    # Find training process - check for complete trainer
    training_pid = None
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'rml_ultimate_trainer_complete.py' in cmdline:
                training_pid = proc.info['pid']
                break
        except:
            continue
    
    if not training_pid:
        print("❌ No complete RML training process found!")
        return
    
    print(f"🟢 Complete RML Training Process: PID {training_pid}")
    print()
    
    try:
        while True:
            # Get process info
            try:
                process = psutil.Process(training_pid)
                cpu_percent = process.cpu_percent()
                memory_mb = process.memory_info().rss / 1024 / 1024
                memory_gb = memory_mb / 1024
                
                # Clear screen
                os.system('clear')
                
                print("📊 RML Training - REAL-TIME ACTIVITY MONITOR")
                print("="*60)
                print(f"🟢 Process: PID {training_pid}")
                print(f"📊 CPU Usage: {cpu_percent:.1f}%")
                print(f"💾 Memory Usage: {memory_gb:.1f} GB ({memory_mb:.1f} MB)")
                print()
                
                # Activity indicators
                if cpu_percent > 50:
                    print("🚀 HIGH ACTIVITY - Processing files intensively!")
                elif cpu_percent > 20:
                    print("⚡ MODERATE ACTIVITY - Processing files")
                elif cpu_percent > 5:
                    print("🔄 LOW ACTIVITY - Between processing phases")
                else:
                    print("⏸️ IDLE - Waiting or between phases")
                
                print()
                
                # Check for log updates
                log_file = "/Users/elite/R-LLM/rml-ultimate-trained/training.log"
                if os.path.exists(log_file):
                    try:
                        with open(log_file, 'r') as f:
                            lines = f.readlines()
                            if lines:
                                print("📝 Recent Log Activity:")
                                print("-" * 40)
                                for line in lines[-8:]:  # Last 8 lines
                                    print(f"   {line.strip()}")
                    except:
                        pass
                
                print()
                
                # Check for any output files
                output_dir = "/Users/elite/R-LLM/rml-ultimate-trained"
                if os.path.exists(output_dir):
                    files = os.listdir(output_dir)
                    if files:
                        print("📁 Output Directory Contents:")
                        print("-" * 40)
                        for file in files:
                            file_path = os.path.join(output_dir, file)
                            if os.path.isfile(file_path):
                                size = os.path.getsize(file_path)
                                print(f"   {file}: {size:,} bytes")
                
                print()
                print("🔄 Monitoring... (Press Ctrl+C to stop)")
                print(f"⏰ Last update: {datetime.now().strftime('%H:%M:%S')}")
                
                time.sleep(2)
                
            except psutil.NoSuchProcess:
                print("❌ Training process stopped!")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

def show_current_status():
    """Show current training status"""
    
    print("📊 RML Training - CURRENT STATUS")
    print("="*40)
    
    # Find training process - check for complete trainer
    training_pid = None
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'rml_ultimate_trainer_complete.py' in cmdline:
                training_pid = proc.info['pid']
                break
        except:
            continue
    
    if training_pid:
        try:
            process = psutil.Process(training_pid)
            cpu_percent = process.cpu_percent()
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_gb = memory_mb / 1024
            
            print(f"🟢 Status: COMPLETE RML TRAINING RUNNING (PID: {training_pid})")
            print(f"📊 CPU: {cpu_percent:.1f}%")
            print(f"💾 Memory: {memory_gb:.1f} GB")
            
            # Activity level
            if cpu_percent > 50:
                print("🚀 Activity: HIGH - Processing intensively")
            elif cpu_percent > 20:
                print("⚡ Activity: MODERATE - Processing files")
            else:
                print("🔄 Activity: LOW - Between phases")
                
        except:
            print("🔴 Status: Process not responding")
    else:
        print("🔴 Status: No complete RML training process found")
    
    # Check data directory
    data_dir = "/Users/elite/R-LLM/data"
    if os.path.exists(data_dir):
        total_size = 0
        file_count = 0
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.jsonl'):
                    file_path = os.path.join(root, file)
                    try:
                        total_size += os.path.getsize(file_path)
                        file_count += 1
                    except:
                        pass
        
        print(f"\n📁 Dataset: {file_count:,} JSONL files")
        print(f"💾 Total Size: {total_size / 1024**3:.1f} GB")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "live":
        monitor_training_activity()
    else:
        show_current_status() 