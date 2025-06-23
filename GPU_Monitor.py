import subprocess
import json
import time
import threading
import signal
import sys
from datetime import datetime

class GPUMonitor:
    def __init__(self, update_interval=2):
        self.update_interval = update_interval
        self.monitoring = False
        self.max_vram_used = 0
        self.max_gpu_util = 0
        
    def get_gpu_info(self):
        """Get GPU information using nvidia-smi"""
        try:
            # Query GPU information in JSON format
            cmd = [
                'nvidia-smi', 
                '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')
            
            gpu_info = []
            for line in lines:
                parts = [part.strip() for part in line.split(',')]
                if len(parts) >= 8:
                    gpu_data = {
                        'index': int(parts[0]),
                        'name': parts[1],
                        'memory_total': int(parts[2]),
                        'memory_used': int(parts[3]),
                        'memory_free': int(parts[4]),
                        'gpu_util': int(parts[5]) if parts[5] != '[Not Supported]' else 0,
                        'temperature': int(parts[6]) if parts[6] != '[Not Supported]' else 0,
                        'power_draw': float(parts[7]) if parts[7] != '[Not Supported]' else 0
                    }
                    gpu_info.append(gpu_data)
            
            return gpu_info
            
        except subprocess.CalledProcessError:
            print("Error: nvidia-smi not found or GPU not accessible")
            return []
        except Exception as e:
            print(f"Error getting GPU info: {e}")
            return []
    
    def display_gpu_status(self, gpu_info):
        """Display formatted GPU information"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n{'='*60}")
        print(f"GPU Status Update - {timestamp}")
        print(f"{'='*60}")
        
        for gpu in gpu_info:
            vram_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
            
            # Update maximums
            self.max_vram_used = max(self.max_vram_used, vram_percent)
            self.max_gpu_util = max(self.max_gpu_util, gpu['gpu_util'])
            
            print(f"GPU {gpu['index']}: {gpu['name']}")
            print(f"  VRAM: {gpu['memory_used']}MB / {gpu['memory_total']}MB ({vram_percent:.1f}%)")
            print(f"  GPU Utilization: {gpu['gpu_util']}%")
            print(f"  Temperature: {gpu['temperature']}°C")
            print(f"  Power Draw: {gpu['power_draw']}W")
            
            # VRAM usage warnings
            if vram_percent > 95:
                print(f"  ⚠️  WARNING: VRAM usage critical!")
            elif vram_percent > 85:
                print(f"  ⚠️  CAUTION: High VRAM usage")
            elif vram_percent > 70:
                print(f"  ℹ️  INFO: Moderate VRAM usage")
            
            print()
        
        print(f"Session Maximums:")
        print(f"  Peak VRAM Usage: {self.max_vram_used:.1f}%")
        print(f"  Peak GPU Utilization: {self.max_gpu_util}%")
    
    def monitor_loop(self):
        """Main monitoring loop"""
        print("Starting GPU monitoring... Press Ctrl+C to stop")
        print("=" * 60)
        
        while self.monitoring:
            gpu_info = self.get_gpu_info()
            if gpu_info:
                self.display_gpu_status(gpu_info)
            else:
                print("No GPU information available")
            
            time.sleep(self.update_interval)
    
    def start_monitoring(self):
        """Start monitoring in a separate thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1)
    
    def get_optimal_batch_size_suggestion(self):
        """Suggest optimal batch size based on current VRAM usage"""
        gpu_info = self.get_gpu_info()
        if not gpu_info:
            return "Cannot determine - no GPU info available"
        
        # Use first GPU for suggestion
        gpu = gpu_info[0]
        vram_used_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
        
        if vram_used_percent < 50:
            return "You can likely increase batch size for better utilization"
        elif vram_used_percent < 70:
            return "Good VRAM utilization - consider small batch size increases"
        elif vram_used_percent < 85:
            return "Optimal VRAM usage - current batch size seems good"
        elif vram_used_percent < 95:
            return "High VRAM usage - consider reducing batch size slightly"
        else:
            return "Critical VRAM usage - reduce batch size to avoid OOM errors"

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\nStopping GPU monitor...")
    sys.exit(0)

def main():
    # Set up signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    
    print("GPU Performance Monitor")
    print("=" * 30)
    
    monitor = GPUMonitor(update_interval=3)  # Update every 3 seconds
    
    # Initial GPU check
    gpu_info = monitor.get_gpu_info()
    if not gpu_info:
        print("No NVIDIA GPUs detected or nvidia-smi not available")
        print("Make sure you have NVIDIA drivers installed and GPU is accessible")
        input("Press Enter to continue anyway...")
        return
    
    print("Available GPUs:")
    for gpu in gpu_info:
        print(f"  GPU {gpu['index']}: {gpu['name']} ({gpu['memory_total']}MB VRAM)")
    
    print(f"\nBatch size suggestion: {monitor.get_optimal_batch_size_suggestion()}")
    
    try:
        monitor.monitor_loop()
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    finally:
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()
