#!/usr/bin/env python3
"""
Detailed Progress Monitor for RML Ultimate Training
Shows dataset loading progress, files processed, remaining files, and estimated time
"""

import os
import time
import psutil
import json
import glob
from datetime import datetime, timedelta

def count_total_files():
    """Count total JSONL files in the data directory"""
    data_root = "/Users/elite/R-LLM/data"
    total_files = 0
    
    print("🔍 Scanning for all JSONL files...")
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith('.jsonl'):
                total_files += 1
    
    return total_files

def get_processed_files():
    """Get list of files that have been processed"""
    progress_file = "/Volumes/MEGA/training_progress.pkl"
    if os.path.exists(progress_file):
        try:
            import pickle
            with open(progress_file, 'rb') as f:
                progress = pickle.load(f)
                return progress.get('processed_files', set())
        except:
            pass
    return set()

def estimate_remaining_time(processed_files, total_files, start_time):
    """Estimate remaining time based on current progress"""
    if processed_files == 0:
        return "Calculating..."
    
    elapsed_time = time.time() - start_time
    files_per_second = processed_files / elapsed_time
    remaining_files = total_files - processed_files
    
    if files_per_second > 0:
        remaining_seconds = remaining_files / files_per_second
        remaining_hours = remaining_seconds / 3600
        return f"{remaining_hours:.1f} hours"
    else:
        return "Calculating..."

def show_detailed_progress():
    """Show detailed progress information"""
    
    print("📊 RML Ultimate Training - DETAILED PROGRESS")
    print("="*60)
    
    # Find training process
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
    
    # Get process info
    try:
        process = psutil.Process(training_pid)
        start_time = process.create_time()
        runtime = time.time() - start_time
        hours = int(runtime // 3600)
        minutes = int((runtime % 3600) // 60)
        seconds = int(runtime % 60)
        
        print(f"🟢 Training Process: PID {training_pid}")
        print(f"⏰ Started: {datetime.fromtimestamp(start_time).strftime('%H:%M:%S')}")
        print(f"⏱️ Runtime: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"📊 CPU: {process.cpu_percent():.1f}%")
        print(f"💾 Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")
        print()
        
    except Exception as e:
        print(f"❌ Error getting process info: {e}")
        return
    
    # Count total files
    print("📁 Dataset Analysis:")
    print("-" * 40)
    
    data_root = "/Users/elite/R-LLM/data"
    total_files = 0
    file_types = {}
    
    print("🔍 Scanning data directory...")
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith('.jsonl'):
                total_files += 1
                # Count file types
                if 'concepts' in file:
                    file_types['concepts'] = file_types.get('concepts', 0) + 1
                elif 'entities' in file:
                    file_types['entities'] = file_types.get('entities', 0) + 1
                elif 'intents' in file:
                    file_types['intents'] = file_types.get('intents', 0) + 1
                elif 'events' in file:
                    file_types['events'] = file_types.get('events', 0) + 1
                elif 'vectors' in file:
                    file_types['vectors'] = file_types.get('vectors', 0) + 1
                elif 'reasoning' in file:
                    file_types['reasoning'] = file_types.get('reasoning', 0) + 1
                elif 'summaries' in file:
                    file_types['summaries'] = file_types.get('summaries', 0) + 1
                elif 'triples' in file:
                    file_types['triples'] = file_types.get('triples', 0) + 1
                elif 'emotions' in file:
                    file_types['emotions'] = file_types.get('emotions', 0) + 1
                elif 'tags' in file:
                    file_types['tags'] = file_types.get('tags', 0) + 1
                else:
                    file_types['other'] = file_types.get('other', 0) + 1
    
    print(f"📊 Total JSONL files: {total_files:,}")
    print()
    
    # Show file type breakdown
    print("📂 File Type Breakdown:")
    for file_type, count in sorted(file_types.items()):
        percentage = (count / total_files) * 100
        print(f"   {file_type.capitalize()}: {count:,} files ({percentage:.1f}%)")
    print()
    
    # Get processed files
    processed_files = get_processed_files()
    processed_count = len(processed_files)
    remaining_count = total_files - processed_count
    
    print("📈 Processing Progress:")
    print("-" * 40)
    print(f"✅ Files processed: {processed_count:,}")
    print(f"⏳ Files remaining: {remaining_count:,}")
    
    if total_files > 0:
        progress_percentage = (processed_count / total_files) * 100
        print(f"📊 Progress: {progress_percentage:.1f}%")
        
        # Estimate remaining time
        if processed_count > 0:
            remaining_time = estimate_remaining_time(processed_count, total_files, start_time)
            print(f"⏱️ Estimated remaining time: {remaining_time}")
    
    print()
    
    # Show recent activity
    print("🔄 Recent Activity:")
    print("-" * 40)
    
    # Check log file for recent activity
    log_file = "/Volumes/MEGA/R-LLM-ultimate-trained/training.log"
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    print("📝 Recent log entries:")
                    for line in lines[-10:]:  # Last 10 lines
                        print(f"   {line.strip()}")
        except Exception as e:
            print(f"   Error reading log: {e}")
    else:
        print("   No log file found")
    
    print()
    
    # Show system resources
    print("💻 System Resources:")
    print("-" * 40)
    memory = psutil.virtual_memory()
    print(f"🖥️ System Memory: {memory.percent:.1f}% ({memory.available / 1024**3:.1f}GB free)")
    
    if os.path.exists("/Volumes/MEGA"):
        try:
            disk_usage = psutil.disk_usage("/Volumes/MEGA")
            print(f"💾 MEGA Drive: {disk_usage.free / 1024**3:.1f}GB free")
        except:
            pass
    
    print()
    print("🔄 Run this script again to see updated progress!")

def monitor_live_progress():
    """Monitor progress in real-time with detailed updates"""
    
    print("📊 RML Ultimate Training - LIVE DETAILED PROGRESS")
    print("="*60)
    print("🔄 Press Ctrl+C to stop monitoring")
    print("="*60)
    
    # Find training process
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
    
    try:
        process = psutil.Process(training_pid)
        start_time = process.create_time()
    except:
        print("❌ Error accessing process!")
        return
    
    # Count total files once
    print("🔍 Counting total files...")
    data_root = "/Users/elite/R-LLM/data"
    total_files = 0
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith('.jsonl'):
                total_files += 1
    
    print(f"📊 Total files to process: {total_files:,}")
    print()
    
    try:
        while True:
            # Clear screen
            os.system('clear')
            
            print("📊 RML Ultimate Training - LIVE DETAILED PROGRESS")
            print("="*60)
            
            # Get current process info
            try:
                runtime = time.time() - start_time
                hours = int(runtime // 3600)
                minutes = int((runtime % 3600) // 60)
                seconds = int(runtime % 60)
                
                print(f"🟢 Process: PID {training_pid}")
                print(f"⏱️ Runtime: {hours:02d}:{minutes:02d}:{seconds:02d}")
                print(f"📊 CPU: {process.cpu_percent():.1f}%")
                print(f"💾 Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")
                print()
                
            except:
                print("❌ Process stopped!")
                break
            
            # Get processed files
            processed_files = get_processed_files()
            processed_count = len(processed_files)
            remaining_count = total_files - processed_count
            
            print("📈 Processing Progress:")
            print("-" * 40)
            print(f"✅ Files processed: {processed_count:,}")
            print(f"⏳ Files remaining: {remaining_count:,}")
            
            if total_files > 0:
                progress_percentage = (processed_count / total_files) * 100
                print(f"📊 Progress: {progress_percentage:.1f}%")
                
                # Progress bar
                bar_length = 30
                filled_length = int(bar_length * progress_percentage / 100)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                print(f"📊 [{bar}] {progress_percentage:.1f}%")
                
                # Estimate remaining time
                if processed_count > 0:
                    remaining_time = estimate_remaining_time(processed_count, total_files, start_time)
                    print(f"⏱️ Estimated remaining: {remaining_time}")
            
            print()
            
            # Show recent files being processed
            print("🔄 Recent Activity:")
            print("-" * 40)
            
            # Check for recent log entries
            log_file = "/Volumes/MEGA/R-LLM-ultimate-trained/training.log"
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            print("📝 Recent log entries:")
                            for line in lines[-5:]:  # Last 5 lines
                                print(f"   {line.strip()}")
                except:
                    pass
            
            print()
            print("🔄 Monitoring... (Press Ctrl+C to stop)")
            time.sleep(3)  # Update every 3 seconds
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "live":
        monitor_live_progress()
    else:
        show_detailed_progress() 