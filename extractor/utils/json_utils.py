import ast
import json
import re
from typing import Dict, Any, Union

def parse_json_safe(result: Any, logger=None, step: str = "") -> Union[Dict, list, str]:
    """
    Robustly parse JSON- or Python-dict-like text.
    - If already a dict/list, return as-is.
    - Else: extract candidates from ```json ...```, any ```...``` block,
      after </think>, and the last {...} block in the text.
    - Try json.loads first, then ast.literal_eval.
    Return dict/list on success, or "" on failure.
    """
    def _candidates(s: str):
        c = []
        # fenced json block
        c += re.findall(r"```json\s*([\s\S]*?)```", s)
        # any fenced block
        c += re.findall(r"```[\w]*\s*([\s\S]*?)```", s)
        # after </think>
        c += re.findall(r"</think>\s*({[\s\S]*})", s)
        # last {...} fallback
        m = re.findall(r"({[\s\S]*})", s)
        if m:
            c.append(m[-1])
        # whole string as last resort
        c.append(s)
        # strip empties
        return [x.strip() for x in c if x and x.strip()]

    # pass-through
    if isinstance(result, (dict, list)):
        return result

    s = result if isinstance(result, str) else str(result)

    for t in _candidates(s):
        # JSON first
        try:
            obj = json.loads(t)
            if logger: logger.event("parse", step=step, method="json", ok=True)
            return obj
        except Exception:
            pass
        # Python literal fallback (handles single quotes, trailing commas in tuples, etc.)
        try:
            obj = ast.literal_eval(t)
            if isinstance(obj, (dict, list)):
                if logger: logger.event("parse", step=step, method="ast", ok=True)
                return obj
        except Exception:
            pass

    if logger: logger.event("parse", step=step, method="fail", ok=False)
    return ""
