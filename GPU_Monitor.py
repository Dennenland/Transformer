import subprocess
import time
import signal
import sys
import os
from datetime import datetime

class GPUMonitor:
    def __init__(self, update_interval=1.0):
        self.update_interval = update_interval
        self.monitoring = True
        self.max_vram_used = 0
        self.max_gpu_util = 0
        self.start_time = time.time()
        self.nvidia_smi_available = self.check_nvidia_smi()
        
    def check_nvidia_smi(self):
        """Check if nvidia-smi is available"""
        try:
            result = subprocess.run(['nvidia-smi', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception as e:
            print(f"nvidia-smi check failed: {e}")
            return False
    
    def get_gpu_info_simple(self):
        """Simplified GPU info retrieval with better error handling"""
        if not self.nvidia_smi_available:
            return None
            
        try:
            # Simple nvidia-smi query
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                print(f"nvidia-smi error: {result.stderr}")
                return None
                
            lines = result.stdout.strip().split('\n')
            gpu_data = []
            
            for i, line in enumerate(lines):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 6:
                        try:
                            data = {
                                'index': i,
                                'name': parts[0],
                                'memory_total': int(parts[1]),
                                'memory_used': int(parts[2]), 
                                'memory_free': int(parts[3]),
                                'gpu_util': int(parts[4]) if parts[4] != '[Not Supported]' else 0,
                                'temperature': int(parts[5]) if parts[5] != '[Not Supported]' else 0
                            }
                            gpu_data.append(data)
                        except ValueError as e:
                            print(f"Error parsing GPU data: {e}")
                            continue
            
            return gpu_data if gpu_data else None
            
        except subprocess.TimeoutExpired:
            print("nvidia-smi timeout - GPU may be busy")
            return None
        except Exception as e:
            print(f"Error getting GPU info: {e}")
            return None
    
    def clear_screen(self):
        """Clear screen - works on Windows and Linux"""
        try:
            os.system('cls' if os.name == 'nt' else 'clear')
        except:
            print("\n" * 50)  # Fallback if clear doesn't work
    
    def create_simple_bar(self, percentage, width=20):
        """Create a simple text progress bar"""
        filled = int(width * percentage / 100)
        return f"[{'#' * filled}{'.' * (width - filled)}] {percentage:5.1f}%"
    
    def display_status(self, gpu_data):
        """Display GPU status with robust formatting"""
        self.clear_screen()
        
        # Calculate runtime
        runtime_seconds = int(time.time() - self.start_time)
        hours, remainder = divmod(runtime_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print("=" * 70)
        print(f"GPU MONITOR - {datetime.now().strftime('%H:%M:%S')} | Runtime: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print("=" * 70)
        
        if not gpu_data:
            print("❌ No GPU data available")
            print("   - Check if NVIDIA drivers are installed")
            print("   - Verify nvidia-smi command works")
            print("   - Ensure GPU is not being used exclusively by another process")
            return
        
        for gpu in gpu_data:
            vram_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
            
            # Update maximums
            self.max_vram_used = max(self.max_vram_used, vram_percent)
            self.max_gpu_util = max(self.max_gpu_util, gpu['gpu_util'])
            
            print(f"\nGPU {gpu['index']}: {gpu['name']}")
            print("-" * 50)
            
            # VRAM Usage
            print(f"VRAM Usage: {self.create_simple_bar(vram_percent)}")
            print(f"  Used: {gpu['memory_used']:,} MB")
            print(f"  Free: {gpu['memory_free']:,} MB") 
            print(f"  Total: {gpu['memory_total']:,} MB")
            
            # GPU Utilization
            print(f"GPU Usage:  {self.create_simple_bar(gpu['gpu_util'])}")
            
            # Temperature
            if gpu['temperature'] > 0:
                temp_percent = min(gpu['temperature'], 100)  # Cap at 100 for display
                print(f"Temperature: {self.create_simple_bar(temp_percent)} ({gpu['temperature']}°C)")
            
            # Status and recommendations
            print(f"\nStatus Assessment:")
            if vram_percent < 30:
                print("  🟢 LOW USAGE - Can significantly increase batch size")
                print("  💡 Try doubling your batch size")
            elif vram_percent < 50:
                print("  🟡 MODERATE - Can increase batch size")
                print("  💡 Try increasing batch size by 50%")
            elif vram_percent < 70:
                print("  🟠 GOOD USAGE - Small increases possible")
                print("  💡 Try increasing batch size by 25%")
            elif vram_percent < 85:
                print("  🔵 OPTIMAL - Well utilized")
                print("  ✅ Current batch size is good")
            elif vram_percent < 95:
                print("  🟠 HIGH USAGE - Monitor carefully")
                print("  ⚠️ Consider reducing batch size slightly")
            else:
                print("  🔴 CRITICAL - Reduce immediately!")
                print("  🚨 Reduce batch size to avoid crashes")
            
        # Session statistics
        print(f"\n" + "=" * 70)
        print("SESSION STATISTICS")
        print("-" * 70)
        print(f"Peak VRAM Usage: {self.max_vram_used:.1f}%")
        print(f"Peak GPU Utilization: {self.max_gpu_util:.1f}%")
        print(f"Monitor Updates: Every {self.update_interval} seconds")
        print(f"\nPress Ctrl+C to stop monitoring...")
    
    def run(self):
        """Main monitoring loop"""
        print("🚀 GPU Monitor Starting...")
        print("Checking for NVIDIA GPUs...")
        
        if not self.nvidia_smi_available:
            print("❌ nvidia-smi not available!")
            print("Make sure NVIDIA drivers are properly installed.")
            input("Press Enter to exit...")
            return
        
        # Initial test
        initial_data = self.get_gpu_info_simple()
        if initial_data:
            print(f"✅ Found {len(initial_data)} GPU(s)")
            for gpu in initial_data:
                print(f"   - GPU {gpu['index']}: {gpu['name']} ({gpu['memory_total']:,}MB)")
        else:
            print("⚠️ No GPU data retrieved on first attempt")
            print("Continuing anyway - data may appear shortly...")
        
        print("\nStarting continuous monitoring...")
        time.sleep(2)
        
        try:
            while self.monitoring:
                gpu_data = self.get_gpu_info_simple()
                self.display_status(gpu_data)
                time.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            print("\n\n⏹️ Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("Monitor will exit...")
        finally:
            self.monitoring = False

def signal_handler(sig, frame):
    print("\n\n⏹️ Shutting down...")
    sys.exit(0)

def main():
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        monitor = GPUMonitor(update_interval=2.0)  # Update every 2 seconds
        monitor.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        print("Please check that:")
        print("1. NVIDIA drivers are installed")
        print("2. nvidia-smi command works from command line")
        print("3. You have an NVIDIA GPU")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    print("Starting GPU Monitor...")
    print("Debug: Python script is running")
    main()
    print("Debug: Python script finished")
