"""Resource telemetry sampled separately from model wall-clock measurements."""
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import psutil


class ResourceMonitor:
    def __init__(self, path, interval=5):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self.stop_event = threading.Event()
        self.process = psutil.Process(os.getpid())
        self.start_time = time.monotonic()
        self.peak_rss = 0

    def sample(self):
        rss = self.process.memory_info().rss
        self.peak_rss = max(self.peak_rss, rss)
        record = {'elapsed_s': time.monotonic()-self.start_time, 'pid': os.getpid(),
                  'rss_bytes': rss, 'cpu_percent': self.process.cpu_percent(),
                  'system_memory_percent': psutil.virtual_memory().percent}
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                     '--format=csv,noheader,nounits'], stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, timeout=3, text=True)
            record['gpu_device_totals_not_process_usage'] = result.stdout.strip().splitlines()
        except (OSError, subprocess.TimeoutExpired):
            record['gpu_device_totals_not_process_usage'] = []
        with self.path.open('a') as f:
            f.write(json.dumps(record) + '\n')

    def _run(self):
        while not self.stop_event.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join(timeout=5)
        self.sample()
