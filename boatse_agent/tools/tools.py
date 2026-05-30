from typing import Any, Dict, List, Optional, Tuple
import os
import json
import pandas as pd
from utils.path_utils import safe_repo_path
from utils.json_utils import parse_json_safe
from utils.view_file_utils import filter_file_preview, read_file_safe
from utils.bm25_utils import extract_from_dict
from agent.run_context import RunContext
from logging.logger import RunLogger

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

_LANG_BY_EXT = {
    ".py": Language.PYTHON,
    ".java": Language.JAVA,
    ".kt": Language.KOTLIN,
    ".kts": Language.KOTLIN,
}


def _guess_language(file_path: str):
    _, ext = os.path.splitext(file_path.lower())
    return _LANG_BY_EXT.get(ext)

def make_code_splitter(lang_enum, token_len_fn, chunk_tokens=512, overlap_tokens=64):
    """
    lang_enum: a langchain_text_splitters.Language or None
    token_len_fn: callable(str)->int using *your* tokenizer
    """
    if lang_enum:
        return RecursiveCharacterTextSplitter.from_language(
            language=lang_enum,
            chunk_size=chunk_tokens,
            chunk_overlap=overlap_tokens,
            length_function=token_len_fn,
            add_start_index=True,  # we’ll compute line numbers
        )
    else:
        # generic fallback: split on paragraphs/lines/spaces, still token-aware
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_tokens,
            chunk_overlap=overlap_tokens,
            length_function=token_len_fn,
            separators=["\n\n", "\n", " ", ""],
            add_start_index=True,
        )

def first_chunk_approx_n_tokens(
    ctx: RunContext,
    text: str,
    file_path: str,
    max_tokens: int = 512,
    overlap_tokens: int = 0,
) -> str:
    """
    Use a token-aware LangChain splitter to get a clean first chunk of ~n tokens,
    preferring to cut on natural boundaries (paragraphs/lines/functions).
    """
    lang_enum = _guess_language(file_path)
    print(f"Guessed language for {file_path}: {lang_enum}")
    splitter = make_code_splitter(lang_enum, ctx.tk.count_text_tokens, max_tokens, overlap_tokens)

    docs = splitter.create_documents(
        [text],
        metadatas=[{"source": file_path}]
    )
    if not docs:
        return "", (1, 1)
    
    chunk = docs[0].page_content
    # Add markers that make the code look intentionally incomplete/broken
    if ctx.tk.count_text_tokens(text) > max_tokens:
        if _guess_language(file_path) == Language.PYTHON:
            chunk += "\n\n# ... [FILE TRUNCATED - REMAINING CODE OMITTED] ..."
        else:
            chunk += "\n\n// ... [FILE TRUNCATED - REMAINING CODE OMITTED] ..."
    
    return chunk

# =========================
# Tools (BM25 / extract / FS)
# =========================

def search_index(query: str, repo_dir: str, top_k: int = 10) -> Tuple[List[str], List[float]]:
    from pyserini.search import LuceneSearcher
    from pyserini.index.lucene import IndexReader

    index_dir = os.path.join(repo_dir, "indexes")
    if top_k == 0:
        reader = IndexReader(index_dir)
        stats = reader.stats()
        print("Documents in index:", stats['documents'])

    searcher = LuceneSearcher(index_dir)
    searcher.set_bm25(k1=0.9, b=0.4)
    hits = searcher.search(query, k=top_k)

    files, scores = [], []
    print(f"Top {top_k} results for '{query}':")
    for i, hit in enumerate(hits):
        doc_raw = hit.lucene_document.get('raw')
        doc_dict = json.loads(doc_raw)
        files.append(doc_dict.get('path'))
        scores.append(hit.score)

    print(f"Final files contains: {files} with {len(files)} hits.")
    return files, scores

def get_bm25_topk(top_k: int, query: str, repo_dir: str) -> Tuple[List[str], List[float]]:
    if 1 <= top_k <= 30:
        print(f"Retrieving top-{top_k} results with BM25 search...")
        return search_index(query, repo_dir, top_k)
    raise ValueError("Invalid top_k value. Please provide an integer between 1 and 30.")

def extract_relevant( ctx: RunContext) -> Any:
    # For demo: return what's been provided in the context
    return ctx.extracted_info

def build_bm25_query(ctx: RunContext) -> str:
    """
    Prefer the structured dict produced by extract_relevant; otherwise
    fall back to the raw issue description string.
    """
    info = ctx.extracted_info
    if isinstance(info, dict):
        parts = []
        # handle both "code_snippet" and "code snippet"
        code_snippet = info.get("code_snippet") or info.get("code snippet") or ""
        identifiers = info.get("identifiers") or []
        if isinstance(identifiers, str):
            identifiers = [identifiers]

        if ctx.extracted_option == "all":
            parts.extend([
                info.get("explanation", ""),
                info.get("path", ""),
                info.get("filename", ""),
                " ".join(map(str, identifiers)),
                info.get("error_message", ""),
                code_snippet,
            ])
        else:
            parts.extend([
                info.get("explanation", ""),
                " ".join(map(str, identifiers)),
                code_snippet,
            ])
        # collapse to a single string
        return " ".join(str(p) for p in parts if p)
    # fallback
    return ctx.issue_description

# =========================
# Tool execution dispatcher
# =========================

def exec_tool(tool_name: str, tool_args: Dict[str, Any], ctx: RunContext, logger: Optional[RunLogger], step: str, CONTENT_LIMIT: int = 2000) -> Tuple[str, Dict[str, Any], Any]:
    """Unified tool executor that returns (tool_name, args, tool_result)."""
    if logger:
        logger.event("tool_decision", step=step, tool_name=tool_name, tool_args=tool_args)

    if tool_name == "get_bm25_topk" or tool_name == "get_bm25_top20" or tool_name == "get_bm25_top10" or tool_name == "get_bm25_top30":
        top_k = 20 if tool_name == "get_bm25_top20" else 10 if tool_name == "get_bm25_top10" else tool_args.get("top_k") if tool_name == "get_bm25_topk" else 30
        # Build query (allow dict-string input)
        concat_info = build_bm25_query(ctx)
        print(f"BM25 query: {concat_info}")

        if logger:
            logger.event("bm25_query", step=step, query_preview=concat_info, used_extracted=bool(ctx.extracted_info))
        
        files, scores = get_bm25_topk(top_k, str(concat_info), ctx.repo_dir)
        if logger:
            logger.event("tool_result", step=step, bm25_top_k=top_k, files=files, scores=scores)
        ctx.bm25_results = {"files": files}
        return tool_name, tool_args, f"bm25 results: {{\"files\": {files}, \"scores\": {scores}}}"

    if tool_name == "extract_relevant":
        # TODO: use info passed as parameter here
        result = extract_relevant(ctx.issue_index, ctx.dataset, ctx.extracted_option)

        if isinstance(result, dict):
            ctx.extracted_info = result
        else:
            parsed = parse_json_safe(result)
            if isinstance(parsed, dict):
                ctx.extracted_info = parsed

        if logger:
            logger.event("tool_result", step=step, extract_relevant=result)
        return tool_name, tool_args, f"extracted info: {result}"

    if tool_name == "view_file":
        file_path = tool_args.get("file_path", "")
        safe = safe_repo_path(ctx.repo_dir, file_path)

        if not safe or not os.path.isfile(safe):
            result = f"File {file_path} not found."
            if logger:
                logger.event("tool_result", step=step, view_file=file_path, error="not found or invalid path")
            return tool_name, tool_args, result
        
        # Remember we requested to view this file
        ctx.viewed_files.append(file_path) 

        view_count = ctx.viewed_files.count(file_path) if ctx.viewed_files else 0
        if view_count > 0 and view_count < 2:
            note = f"[Note: you already viewed this file. Consider choosing another file.]"
        elif view_count >= 2 and view_count < 5:
            note = f"[Note: you have viewed this file {view_count} times. Please select another file.]"
            return tool_name, tool_args, note
        elif view_count >= 5:
            return "max_views_reached", tool_args, ""
        else:
            note = ""

        content, err = read_file_safe(safe)
        if err:
            result = f"File {file_path} not found."
            if logger:
                logger.event("tool_result", step=step, view_file=file_path, error=err)
            return tool_name, tool_args, result

        text = content or ""
        filtered = filter_file_preview(file_path, text) ## this filters out too much
        ctx.viewed_files_full[file_path] = filtered

        print(f"Viewing file: {safe}")

        if logger:
            removed = max(0, len(text.splitlines()) - len(filtered.splitlines()))
            logger.event("preview_filter", step=step, file=file_path, removed_lines=removed)

        clipped = first_chunk_approx_n_tokens(
                    ctx=ctx, text=filtered, file_path=file_path,
                    max_tokens=512,
                    overlap_tokens=0,
                )
        if logger:
            logger.event("tool_result", step=step, view_file=file_path, content=clipped)
        return tool_name, tool_args, f"{note}\nfile content: {file_path}: {clipped}"

    if tool_name == "view_readme":
        safe = safe_repo_path(ctx.repo_dir, "README.md")
        if not safe or not os.path.isfile(safe):
            result = "README.md file not found."
            if logger:
                logger.event("tool_result", step=step, view_readme=True, error="not found or invalid path")
            return tool_name, tool_args, result
        content, err = read_file_safe(safe)
        if err:
            result = "README.md file not found."
            if logger:
                logger.event("tool_result", step=step, view_readme=True, error=err)
            return tool_name, tool_args, result
        clipped = ctx.tk._truncate(content, CONTENT_LIMIT)
        if logger:
            logger.event("tool_result", step=step, view_readme=True, content=clipped)
        ctx.viewed_files.append("README.md")
        return tool_name, tool_args, f"readme: {clipped}"
        
    elif tool_name == "exit":
        rf = tool_args.get("ranked_files")
        if isinstance(rf, list):
            if logger:
                logger.event("final_answer", step=step, ranked_files=rf)
            return "exit", tool_args, {"ranked_files": rf}
        if logger:
            logger.event("tool_result", step=step, exit_called=True)
        return "exit", tool_args, "Exiting function calling to give final answer."

    # Fallback: maybe the model directly returned final ranked files
    ranked_files = tool_args.get("ranked_files", [])
    if ranked_files:
        if logger:
            logger.event("final_answer", step=step, ranked_files=ranked_files)
        return "exit", tool_args, {"ranked_files": ranked_files}

    if logger:
        logger.event("warning", step=step, message="Unknown tool name")
    return tool_name, tool_args, "Unknown tool name. Please check the response."