import shutil
import os
import re
import json
import ast
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from pathlib import Path


def is_test_file(file_path: str):
    test_phrases = ["test", "tests", "testing"]
    words = set(re.split(r" |_|\/|\.", file_path.lower()))
    return any(word in words for word in test_phrases)

def build_json_files(source_dir:Path, index_dir: str, exts=('.py',), skip_tests=True):
    
    # clear up any existing index at that location
    shutil.rmtree(index_dir, ignore_errors=True)
    os.makedirs(index_dir)

    all_py = [
        p
        for p in source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    ]
    print(f"Found {len(all_py)} total file(s) with extension(s) {exts}.")

    # Filter to only keep python files
    if skip_tests:
        filtered = []
        for p in all_py:
            # e.g. "astropy/modeling/tests/test_core.py"
            rel = p.relative_to(source_dir).as_posix()
            if is_test_file(rel):
                # skip it
                continue
            filtered.append(p)
        code_files = filtered
    else:
        code_files = all_py

    print(f"After skip_tests={skip_tests}, keeping {len(code_files)} file(s) to index.")

    
    for doc_id, path in enumerate(code_files):
        try:
            content = path.read_text(encoding='utf-8', errors='ignore')
            doc = {
                'id': str(doc_id),
                'contents': content,
                'path': str(path.relative_to(source_dir).as_posix())
            }
            outpath = Path(index_dir) / f'{doc_id}.json'
            with open(outpath, 'w', encoding='utf-8') as f:
                json.dump(doc, f)
        except Exception as e:
            print(f"Could not process file {path}, error: {str(e)}")

    print(f"Indexed {len(code_files)} JSON files to {index_dir}")

def extract_from_dict(json_str: str) -> str:
    try:
        obj = ast.literal_eval(json_str)
    except:
        print(f"Could not extract relevant information from dictionary")
        return ""

    keep = ["explanation", "identifiers", "code_snippet", "code snippet"]
    extracted = {k: obj[k] for k in keep if k in obj}
    print(f'Extracted: {extracted}')
    return extracted

def _parse_structured(s: Any) -> Optional[Union[Dict, List]]:
    """Parse JSON or Python-literal dict/list from a cell. Return dict/list or None."""
    if isinstance(s, (dict, list)):
        return s
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    text = str(s).strip()
    if not text:
        return None

    # 1) Try strict JSON first
    try:
        obj = json.loads(text)
        if isinstance(obj, (dict, list)):
            return obj
    except Exception:
        pass

    # 2) Try Python literal (handles single quotes, etc.)
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, (dict, list)):
            return obj
    except Exception:
        pass

    # 3) Try to extract the last {...} or [...] block and parse that
    for pat in (r"({[\s\S]*})", r"(\[[\s\S]*\])"):
        matches = re.findall(pat, text)
        if matches:
            chunk = matches[-1].strip()
            # JSON
            try:
                obj = json.loads(chunk)
                if isinstance(obj, (dict, list)):
                    return obj
            except Exception:
                pass
            # Python literal
            try:
                obj = ast.literal_eval(chunk)
                if isinstance(obj, (dict, list)):
                    return obj
            except Exception:
                pass

    return None

def _build_query_from_dict(d: Dict[str, Any]) -> str:
    """Compose a BM25 query string from a structured dict."""
    code_snippet = d.get("code_snippet") or d.get("code snippet") or ""
    identifiers = d.get("identifiers")
    if isinstance(identifiers, str):
        identifiers = [identifiers]
    if identifiers is None:
        identifiers = []

    parts = [
        d.get("explanation", ""),
        d.get("path", ""),
        d.get("filename", ""),
        " ".join(map(str, identifiers)) if identifiers else "",
        d.get("error_message", ""),
        d.get("stack_trace", ""),
        code_snippet,
    ]
    text = " ".join(p for p in parts if p)
    return re.sub(r"\s+", " ", text).strip()

def _coerce_issue_description(cell: Any) -> str:
    """
    Always return a string for BM25:
      - dict  -> normalized text
      - list  -> join each item's normalized text
      - else  -> string of the cell
    """
    obj = _parse_structured(cell)
    if isinstance(obj, dict):
        return _build_query_from_dict(obj)
    if isinstance(obj, list):
        chunks: List[str] = []
        for item in obj:
            if isinstance(item, dict):
                chunks.append(_build_query_from_dict(item))
            else:
                chunks.append(str(item))
        text = " ".join(chunks)
        return re.sub(r"\s+", " ", text).strip()
    # fallback: raw string
    return re.sub(r"\s+", " ", str(cell)).strip()