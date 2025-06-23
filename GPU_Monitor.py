import subprocess
import time
import threading
import signal
import sys
import os
from datetime import datetime
from collections import deque

class GPUMonitor:
    def __init__(self, update_interval=0.5):  # Much faster updates
        self.update_interval = update_interval
        self.monitoring = False
        self.max_vram_used = 0
        self.max_gpu_util = 0
        self.vram_history = deque(maxlen=120)  # 1 minute of history at 0.5s intervals
        self.gpu_util_history = deque(maxlen=120)
        self.temp_history = deque(maxlen=120)
        self.power_history = deque(maxlen=120)
        self.start_time = None
        
    def clear_screen(self):
        """Clear the console screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def get_gpu_info(self):
        """Get GPU information using nvidia-smi"""
        try:
            cmd = [
                'nvidia-smi', 
                '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit',
                '--format=csv,noheader,nounits'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
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
                        'memory_util': int(parts[6]) if len(parts) > 6 and parts[6] != '[Not Supported]' else 0,
                        'temperature': int(parts[7]) if parts[7] != '[Not Supported]' else 0,
                        'power_draw': float(parts[8]) if len(parts) > 8 and parts[8] != '[Not Supported]' else 0,
                        'power_limit': float(parts[9]) if len(parts) > 9 and parts[9] != '[Not Supported]' else 0
                    }
                    gpu_info.append(gpu_data)
            
            return gpu_info
            
        except Exception as e:
            return []
    
    def create_bar(self, percentage, width=30):
        """Create a visual progress bar"""
        filled = int(width * percentage / 100)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}] {percentage:5.1f}%"
    
    def get_trend_indicator(self, history):
        """Get trend indicator based on recent history"""
        if len(history) < 10:
            return "─"
        
        recent = list(history)[-10:]
        older = list(history)[-20:-10] if len(history) >= 20 else recent[:5]
        
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        
        if recent_avg > older_avg + 2:
            return "↗"
        elif recent_avg < older_avg - 2:
            return "↘"
        else:
            return "→"
    
    def get_utilization_status(self, vram_percent, gpu_util):
        """Get optimization recommendations"""
        if vram_percent < 30:
            return "🟢 LOW - Increase batch size significantly", "Can increase batch size by 50-100%"
        elif vram_percent < 50:
            return "🟡 MODERATE - Can increase batch size", "Try increasing batch size by 25-50%"
        elif vram_percent < 70:
            return "🟠 GOOD - Reasonable utilization", "Small batch size increases possible"
        elif vram_percent < 85:
            return "🔵 OPTIMAL - Well utilized", "Current settings are good"
        elif vram_percent < 95:
            return "🟠 HIGH - Monitor closely", "Consider reducing batch size slightly"
        else:
            return "🔴 CRITICAL - Reduce immediately", "Reduce batch size to avoid OOM errors"
    
    def display_gpu_status(self, gpu_info):
        """Display comprehensive GPU information with live updates"""
        if not gpu_info:
            return
            
        self.clear_screen()
        
        # Header
        runtime = ""
        if self.start_time:
            elapsed = time.time() - self.start_time
            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)
            runtime = f" | Runtime: {hours:02d}:{minutes:02d}:{seconds:02d}"
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{'='*80}")
        print(f"🚀 REAL-TIME GPU MONITOR - {timestamp}{runtime}")
        print(f"{'='*80}")
        
        for i, gpu in enumerate(gpu_info):
            vram_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
            power_percent = (gpu['power_draw'] / gpu['power_limit']) * 100 if gpu['power_limit'] > 0 else 0
            
            # Update history
            self.vram_history.append(vram_percent)
            self.gpu_util_history.append(gpu['gpu_util'])
            self.temp_history.append(gpu['temperature'])
            self.power_history.append(power_percent)
            
            # Update maximums
            self.max_vram_used = max(self.max_vram_used, vram_percent)
            self.max_gpu_util = max(self.max_gpu_util, gpu['gpu_util'])
            
            status, recommendation = self.get_utilization_status(vram_percent, gpu['gpu_util'])
            
            print(f"\n🎯 GPU {gpu['index']}: {gpu['name']}")
            print(f"{'─'*80}")
            
            # VRAM Usage
            vram_trend = self.get_trend_indicator(self.vram_history)
            print(f"💾 VRAM Usage: {self.create_bar(vram_percent)} {vram_trend}")
            print(f"   Used: {gpu['memory_used']:,}MB / {gpu['memory_total']:,}MB")
            print(f"   Free: {gpu['memory_free']:,}MB")
            
            # GPU Utilization
            gpu_trend = self.get_trend_indicator(self.gpu_util_history)
            print(f"⚡ GPU Compute: {self.create_bar(gpu['gpu_util'])} {gpu_trend}")
            
            # Temperature
            temp_trend = self.get_trend_indicator(self.temp_history)
            temp_color = "🔥" if gpu['temperature'] > 80 else "🌡️"
            print(f"{temp_color} Temperature: {self.create_bar(gpu['temperature'], width=20)} {temp_trend}")
            print(f"   Current: {gpu['temperature']}°C")
            
            # Power Usage
            if gpu['power_draw'] > 0:
                power_trend = self.get_trend_indicator(self.power_history)
                print(f"⚡ Power Draw: {self.create_bar(power_percent, width=20)} {power_trend}")
                print(f"   Current: {gpu['power_draw']:.1f}W / {gpu['power_limit']:.1f}W")
            
            print(f"\n📊 Status: {status}")
            print(f"💡 Recommendation: {recommendation}")
            
        # Session Statistics
        print(f"\n{'='*80}")
        print(f"📈 SESSION STATISTICS")
        print(f"{'─'*80}")
        print(f"Peak VRAM Usage: {self.max_vram_used:.1f}%")
        print(f"Peak GPU Utilization: {self.max_gpu_util:.1f}%")
        
        if len(self.vram_history) > 10:
            avg_vram = sum(list(self.vram_history)[-60:]) / min(60, len(self.vram_history))
            avg_gpu = sum(list(self.gpu_util_history)[-60:]) / min(60, len(self.gpu_util_history))
            print(f"Average VRAM (last min): {avg_vram:.1f}%")
            print(f"Average GPU Util (last min): {avg_gpu:.1f}%")
