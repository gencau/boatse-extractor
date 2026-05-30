import os, time, uuid
import json
from contextlib import contextmanager
from datetime import datetime, timezone

TRUNCATE_CHARS = 10000  # adjust to your needs

def _truncate(s, n=TRUNCATE_CHARS):
    if s is None:
        return None
    s = str(s)
    return s if len(s) <= n else (s[:n] + f"... <truncated {len(s)-n} chars>")

def _fmt_messages(msgs):
    try:
        out = []
        for m in msgs:
            out.append({"role": m.get("role"), "content": _truncate(m.get("content"))})
        return out
    except Exception:
        return _truncate(msgs)

class RunLogger:
    def __init__(self, log_dir="logs", run_id=None, model_name=None):
        os.makedirs(log_dir, exist_ok=True)
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        self.path = os.path.join(log_dir, f"{self.run_id}.jsonl")
        self.summary = {
            "run_id": self.run_id,
            "model": model_name,
            "reprompts": 0,
            "events": 0,
            "started_at": datetime.now(timezone.utc).isoformat()
        }

    def _write(self, record: dict):
        record["run_id"] = self.run_id
        record["ts"] = datetime.now(timezone.utc).isoformat()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.summary["events"] += 1

    def event(self, event_type, step, **kwargs):
        self._write({"type": event_type, "step": step, **kwargs})

    def inc_reprompt(self, step, attempt, last_msg=None, model_reply=None):
        self.summary["reprompts"] += 1
        self._write({
            "type": "reprompt",
            "step": step,
            "attempt": attempt,
            "last_msg": _truncate(last_msg),
            "model_reply": _truncate(model_reply),
        })

    @contextmanager
    def timeit(self, step, label="duration"):
        start = time.perf_counter()
        try:
            yield
        finally:
            dur = time.perf_counter() - start
            self._write({"type": "timing", "step": step, label: dur})

    def close(self):
        self.summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        # write summary as a final line for convenience
        self._write({"type": "summary", **self.summary})
