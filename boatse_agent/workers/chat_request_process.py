# at top-level imports
import json, subprocess, sys, os
from pathlib import Path

WORKER_FILE = (Path(__file__).parent / "inference_worker.py").resolve()

class CallTimeout(RuntimeError):
    pass

def run_with_timeout(messages, model_kwargs, timeout_s=300):
    cmd = [sys.executable, "-u", str(WORKER_FILE)]  # run as a module
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # we can read if needed
        text=True,
    )
    try:
        req = json.dumps({"messages": messages, "model_kwargs": model_kwargs})
        out, err = proc.communicate(req, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise CallTimeout(f"Model call exceeded {timeout_s}s")

    if proc.returncode != 0:
        raise RuntimeError(f"worker exited {proc.returncode}: {err.strip()}")

    data = json.loads(out)
    if not data.get("ok"):
        raise RuntimeError(f"worker error: {data}")
    return data["content"]