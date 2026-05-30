import os
from typing import Optional, Tuple, Dict, Any, List


def _within_repo(repo_dir: str, path: str) -> bool:
    repo_abs = os.path.abspath(repo_dir)
    p_abs = os.path.abspath(os.path.join(repo_abs, str(path).lstrip("/\\")))
    try:
        return os.path.commonpath([p_abs, repo_abs]) == repo_abs
    except Exception:
        return False
    
def safe_repo_path(repo_dir: str, file_path: str) -> Optional[str]:
    """
    Concatenate the absolute repo root and the rest of the path, while:
      - de-duplicating if file_path already includes the repo root
      - blocking path traversal outside of repo_dir
    """
    if not file_path:
        return None

    repo_abs = os.path.abspath(repo_dir)
    cand = str(file_path)


    # Force relative join (remove any leading separators to avoid os.path.join swallowing repo_abs)
    full = os.path.join(repo_abs, cand.lstrip("/\\"))
    full_abs = os.path.abspath(full)
    print(f"Resolved path: {full_abs}")
    return full_abs

def is_missing_path(tool_name: str, tool_args: Dict[str, Any], result: Any, repo_dir: str) -> Tuple[bool, str]:
    """
    True if the target path is invalid/missing in repo_dir, plus a clear error message.
    Works for view_file/view_readme; falls back to string check otherwise.
    """
    if tool_name == "view_file":
        file_path = tool_args.get("file_path", "")
        safe = safe_repo_path(repo_dir, file_path)
        if not safe:
            return True, f"Invalid path '{file_path}'. Paths must be inside the repository root."
        if not os.path.isfile(safe):
            return True, f"File '{file_path}' does not exist in the repository."
        return False, ""

    if tool_name == "view_readme":
        safe = safe_repo_path(repo_dir, "README.md")
        if not safe or not os.path.isfile(safe):
            return True, "README.md does not exist in the repository."
        return False, ""

    # Non path-based tools: be tolerant and look for a textual error hint
    if isinstance(result, str) and "not found" in result.lower():
        return True, str(result)
    return False, ""

def validate_ranked_files(ranked_files: List[str], repo_dir: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Returns (valid_files, invalid_entries).
    invalid_entries is a list of (path, reason).
    """
    valid: List[str] = []
    invalid: List[Tuple[str, str]] = []

    for p in ranked_files or []:
        if not p or not isinstance(p, str):
            invalid.append((str(p), "not a string path"))
            continue
        if not _within_repo(repo_dir, p):
            invalid.append((p, "outside repository root"))
            continue
        abs_p = os.path.abspath(os.path.join(repo_dir, p.lstrip("/\\")))
        if not os.path.isfile(abs_p):
            invalid.append((p, "file does not exist"))
            continue
        valid.append(p)

    return valid, invalid