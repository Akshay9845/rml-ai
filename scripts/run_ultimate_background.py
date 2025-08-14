#!/usr/bin/env python3
"""
Background Runner for RML Ultimate Training
Runs training in background, continues even when user chats
"""

import os
import sys
import subprocess
import time
import signal
import psutil
import threading
from pathlib import Path

def run_ultimate_training():
    """Run the ultimate training in background"""
    
    print("🚀 Starting RML Ultimate Training in BACKGROUND...")
    print("="*60)
    print("📊 This will train on your ENTIRE 200GB dataset")
    print("💾 Output: /Volumes/MEGA/R-LLM-ultimate-trained")
    print("🔄 Checkpoints: /Volumes/MEGA/R-LLM-checkpoints")
    print("📝 Logs: /Volumes/MEGA/R-LLM-ultimate-trained/training.log")
    print("="*60)
    
    # Check if MEGA volume is available
    if not os.path.exists("/Volumes/MEGA"):
        print("❌ ERROR: /Volumes/MEGA not found!")
        print("Please connect your MEGA drive first.")
        return False
    
    # Create output directories
    os.makedirs("/Volumes/MEGA/R-LLM-ultimate-trained", exist_ok=True)
    os.makedirs("/Volumes/MEGA/R-LLM-checkpoints", exist_ok=True)
    
    # Start the training process
    cmd = [
        "python3", "src/rml_ultimate_trainer.py",
        "--output", "/Volumes/MEGA/R-LLM-ultimate-trained"
    ]
    
    print(f"🔧 Running command: {' '.join(cmd)}")
    print("🚀 Training started in background...")
    print("💬 You can continue chatting - training will continue!")
    print("📊 Monitor progress with: tail -f /Volumes/MEGA/R-LLM-ultimate-trained/training.log")
    print("🛑 To stop: pkill -f rml_ultimate_trainer.py")
    
    try:
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Print process info
        print(f"📊 Process ID: {process.pid}")
        print(f"📊 Process started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Monitor the process
        monitor_process(process)
        
    except Exception as e:
        print(f"❌ Error starting training: {e}")
        return False
    
    return True

def monitor_process(process):
    """Monitor the training process"""
    
    print("📊 Monitoring training process...")
    
    try:
        # Read output in real-time
        for line in iter(process.stdout.readline, ''):
            if line:
                print(f"[TRAINING] {line.strip()}")
                
                # Check for completion
                if "ULTIMATE training completed" in line:
                    print("✅ Training completed successfully!")
                    break
                elif "Training error" in line:
                    print("❌ Training encountered an error!")
                    break
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code == 0:
            print("✅ Training process completed successfully!")
        else:
            print(f"❌ Training process failed with return code: {return_code}")
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        process.terminate()
        process.wait()
        print("✅ Training process terminated")

def check_training_status():
    """Check if training is currently running"""
    
    try:
        # Check for running training processes
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'rml_ultimate_trainer.py' in ' '.join(proc.info['cmdline'] or []):
                    return proc.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    except Exception:
        return None

def show_status():
    """Show current training status"""
    
    print("📊 RML Ultimate Training Status")
    print("="*40)
    
    # Check if training is running
    pid = check_training_status()
    if pid:
        print(f"🟢 Training is RUNNING (PID: {pid})")
        
        # Show log tail
        log_file = "/Volumes/MEGA/R-LLM-ultimate-trained/training.log"
        if os.path.exists(log_file):
            print("\n📝 Recent log entries:")
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-10:]:  # Last 10 lines
                        print(f"   {line.strip()}")
            except Exception as e:
                print(f"   Error reading log: {e}")
    else:
        print("🔴 Training is NOT running")
        
        # Check for progress file
        progress_file = "/Volumes/MEGA/training_progress.pkl"
        if os.path.exists(progress_file):
            print("📂 Found progress file - training can be resumed")
    
    # Show disk usage
    if os.path.exists("/Volumes/MEGA"):
        try:
            disk_usage = psutil.disk_usage("/Volumes/MEGA")
            print(f"\n💾 MEGA Drive: {disk_usage.free / 1024**3:.1f}GB free")
        except Exception:
            pass

def stop_training():
    """Stop the training process"""
    
    print("🛑 Stopping RML Ultimate Training...")
    
    pid = check_training_status()
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"✅ Sent termination signal to process {pid}")
            
            # Wait a bit, then force kill if needed
            time.sleep(5)
            if check_training_status():
                os.kill(pid, signal.SIGKILL)
                print(f"🔄 Force killed process {pid}")
                
        except Exception as e:
            print(f"❌ Error stopping process: {e}")
    else:
        print("ℹ️ No training process found")

def main():
    """Main function"""
    
    if len(sys.argv) < 2:
        print("🧠 RML Ultimate Training Background Runner")
        print("="*50)
        print("Usage:")
        print("  python3 scripts/run_ultimate_background.py start    # Start training")
        print("  python3 scripts/run_ultimate_background.py status   # Check status")
        print("  python3 scripts/run_ultimate_background.py stop     # Stop training")
        print("  python3 scripts/run_ultimate_background.py logs     # Show recent logs")
        return
    
    command = sys.argv[1].lower()
    
    if command == "start":
        # Check if already running
        if check_training_status():
            print("⚠️ Training is already running!")
            show_status()
            return
        
        # Start training
        run_ultimate_training()
        
    elif command == "status":
        show_status()
        
    elif command == "stop":
        stop_training()
        
    elif command == "logs":
        log_file = "/Volumes/MEGA/R-LLM-ultimate-trained/training.log"
        if os.path.exists(log_file):
            print("📝 Recent training logs:")
            print("-" * 40)
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-20:]:  # Last 20 lines
                        print(line.strip())
            except Exception as e:
                print(f"❌ Error reading logs: {e}")
        else:
            print("❌ No log file found")
    
    else:
        print(f"❌ Unknown command: {command}")

if __name__ == "__main__":
    main() 