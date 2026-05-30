import json
import ast
from typing import List, Optional, Any

def extract_ranked_files_from_any(parsed: Any) -> Optional[List[str]]:
    """
    Accepts:
      - {"ranked_files": [...]}
      - {"function_call": "exit", "args": {"ranked_files": [...]} }
      - {"function_call": {"name":"exit","arguments":"{...json...}"}}  # OpenAI style
    Returns the list or None.
    """
    if not isinstance(parsed, dict):
        return None

    # Case A: top-level
    rf = parsed.get("ranked_files")
    if isinstance(rf, list):
        return rf

    # Case B: flat function_call form
    fc = parsed.get("function_call")
    args = parsed.get("args")

    if isinstance(fc, str) and fc.lower() == "exit":
        if isinstance(args, dict) and isinstance(args.get("ranked_files"), list):
            return args["ranked_files"]

    # Case C: OpenAI style function_call object
    if isinstance(fc, dict):
        name = fc.get("name")
        if isinstance(name, str) and name.lower() == "exit":
            arguments = fc.get("arguments")
            if isinstance(arguments, dict):
                rf = arguments.get("ranked_files")
                if isinstance(rf, list):
                    return rf
            elif isinstance(arguments, str):
                # arguments may be a JSON string
                try:
                    aobj = json.loads(arguments)
                except Exception:
                    try:
                        aobj = ast.literal_eval(arguments)
                    except Exception:
                        aobj = None
                if isinstance(aobj, dict):
                    rf = aobj.get("ranked_files")
                    if isinstance(rf, list):
                        return rf

    return None
