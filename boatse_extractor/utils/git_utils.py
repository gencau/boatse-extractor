from typing import List, Literal
import re
import unidiff

Side = Literal["a", "b", "both", "new", "old"]

def _changed_files_from_git_diff(diff_str: str, side: Side = "b") -> List[str]:
    """
    Parse change file names from diff
    :param diff_str: diff in string format gather from `get_git_diff_between_commits`
    :return: list of changed files according to diff
    """
    files, seen = [], set()
    for m in re.finditer(r'(?m)^diff --git a/(.+?) b/(.+)$', diff_str):
        a, b = m.group(1).strip(), m.group(2).strip()
        picks = {"b":[b], "new":[b], "a":[a], "old":[a], "both":[a, b]}[side]
        for p in picks:
            if p and p not in seen:
                files.append(p)
                seen.add(p)

    return list(files)

def parse_changed_files_from_diff(diff_str: str) -> List[str]:
    """
    Parse change file names from diff
    :param diff_str: diff in string format gather from `get_git_diff_between_commits`
    :return: list of changed files according to diff
    """
    try:
        source_files = {   
            patched_file.source_file.split("a/", 1)[-1]
            for patched_file in unidiff.PatchSet.from_string(diff_str)
        }
    except Exception as e:
        # Fallback: parse lines starting with "+++ b/"
        return _changed_files_from_git_diff(diff_str)

    return list(source_files)