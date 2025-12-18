import os, subprocess, signal, time
from typing import List, Dict, Optional
import requests
import tkinter as tk
from tkinter import filedialog

from helpers_ import is_windows, safe_kill_process, server_base, check_health, now_ms, approx_token_count, fetch_metrics_text, parse_prometheus_sample, pick_file


def is_windows() -> bool:
    return os.name == "nt"


def safe_kill_process(p: subprocess.Popen):
    if p is None:
        return
    try:
        if is_windows():
            # terminate() is usually enough for llama-server.exe; kill() if needed
            p.terminate()
        else:
            p.send_signal(signal.SIGTERM)
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    except Exception:
        pass


def server_base(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def check_health(host: str, port: int, timeout=0.6) -> bool:
    # Many llama-server builds serve a root page; fallback to /v1/models
    base = server_base(host, port)
    try:
        r = requests.get(base, timeout=timeout)
        if r.status_code < 500:
            return True
    except Exception:
        pass
    try:
        r = requests.get(f"{base}/v1/models", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def approx_token_count(text: str) -> int:
    # Rough heuristic: ~4 chars/token in English-ish text.
    # For analysis UI it’s enough; you can swap this with a real tokenizer later.
    if not text:
        return 0
    return max(1, len(text) // 4)


def fetch_metrics_text(host: str, port: int) -> Optional[str]:
    try:
        r = requests.get(f"{server_base(host, port)}/metrics", timeout=1.0)
        if r.status_code == 200 and r.text.strip():
            return r.text
    except Exception:
        return None
    return None


def parse_prometheus_sample(metrics_text: str, keys: List[str]) -> Dict[str, float]:
    """
    Super-light Prometheus text parser:
    returns last seen numeric value for each key (exact metric name match).
    """
    out: Dict[str, float] = {}
    if not metrics_text:
        return out
    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # line format: metric_name{labels...} value
        parts = line.split()
        if len(parts) < 2:
            continue
        name_and_labels = parts[0]
        value_str = parts[1]
        # metric name ends before '{'
        metric_name = name_and_labels.split("{", 1)[0]
        if metric_name in keys:
            try:
                out[metric_name] = float(value_str)
            except Exception:
                pass
    return out

def pick_file(initial_dir=".", filetypes=(("All files", "*.*"),)):
    root = tk.Tk()
    root.withdraw()           # hide main window
    root.attributes("-topmost", True)  # bring dialog to front
    path = filedialog.askopenfilename(
        initialdir=initial_dir,
        filetypes=filetypes
    )
    root.destroy()
    return path