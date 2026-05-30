from typing import Any, Dict, List
import json

def to_assistant_content(obj: Any) -> str:
    """Ensure content appended to messages is a string."""
    if isinstance(obj, (dict, list)):
        return json.dumps(obj, ensure_ascii=False)
    return str(obj)

def append_user_prompt(messages: List[Dict[str, str]], template: str, **kwargs) -> None:
    messages.append({"role": "user", "content": template.format(**kwargs)})
