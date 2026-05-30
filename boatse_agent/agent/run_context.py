from dataclasses import dataclass
from typing import Optional, Dict, Any
import src.utils.tokenization_utils as tk

# =========================
# Data structures
# =========================
@dataclass
class RunContext:
    repo_name: str
    repo_dir: str
    dataset: str
    issue_index: int
    issue_description: str
    model_name: str
    num_files: str
    tk: tk.TokenizationUtils
    extracted_option: str = "all"
    viewed_files: Optional[list[str]] = None  # file_paths
    viewed_files_full: Optional[Dict[str, str]] = None  # file_path -> full content
    bm25_results: Optional[Dict[str, Any]] = None
    extracted_info: Optional[Dict[str, Any]] = None