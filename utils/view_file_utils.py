import os
import re
from typing import Literal, Optional, Tuple


# ---------- Python ----------

# Top-of-file block of '#' comments only
_TOP_HASH_COMMENT_BLOCK = re.compile(r'^\s*(?:#.*\n)+')

# Python import lines - more precise to avoid removing too much
_PY_IMPORT_LINE = re.compile(
    r'^\s*(?:from\s+[\w.]+\s+import\s+[\w., *]+|import\s+[\w., ]+)$'
)

def filter_python_minimal(src: str) -> str:
    original = src

    # 1) Remove ONLY the first leading block of '#' lines that contains copyright
    m = _TOP_HASH_COMMENT_BLOCK.match(src)
    if m:
        comment_block = src[m.start():m.end()]
        if 'copyright' in comment_block.lower():
            # Remove just this one copyright block and stop
            src = src[m.end():]

    # 2) Process lines to remove imports but preserve docstrings and code
    lines = src.split('\n')
    filtered_lines = []
    in_triple_quote = False
    in_single_quote = False
    quote_char = None
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Track triple quote state for docstrings
        if not in_triple_quote and not in_single_quote:
            # Check for start of triple quoted string
            if '"""' in line or "'''" in line:
                # Count quotes to determine if we're entering a string
                triple_double = line.count('"""')
                triple_single = line.count("'''")
                
                if triple_double % 2 == 1 or triple_single % 2 == 1:
                    in_triple_quote = True
                    if triple_double % 2 == 1:
                        quote_char = '"""'
                    else:
                        quote_char = "'''"
                    filtered_lines.append(line)
                    continue
        
        # Handle triple quote state
        if in_triple_quote:
            filtered_lines.append(line)
            if quote_char in line:
                # Check if this line ends the triple quote
                # Simple heuristic: if the quote char appears again, we might be ending
                parts = line.split(quote_char)
                if len(parts) > 1 and parts[-1].count(quote_char) % 2 == 0:
                    in_triple_quote = False
                    quote_char = None
            continue
        
        # Only remove import lines when NOT in any kind of string/docstring
        if (_PY_IMPORT_LINE.match(line) and 
            not stripped.startswith('#') and  # Don't remove commented imports
            not any(word in stripped.lower() for word in ['def ', 'class ', '=', 'if ', 'for ', 'return '])):  # Don't remove if it contains other constructs
            continue
        
        filtered_lines.append(line)
    
    src = '\n'.join(filtered_lines)

    # 3) Tidy up blank lines but be less aggressive
    src = re.sub(r'\n{3,}', '\n\n', src).lstrip('\n')

    # Fallback if we stripped everything
    if not src.strip():
        return original.strip()
    return src


# ---------- Java / Kotlin ----------

# Multiple top-of-file block comments (more flexible)
_TOP_BLOCK_COMMENTS = re.compile(r'^\s*(?:/\*.*?\*/\s*)+', re.DOTALL)

# More precise package/import pattern
_JAVA_KOTLIN_PKG_IMPORT = re.compile(
    r'^\s*(?:package|import)\s+[\w.]+(?:\s*\.\s*[\w.]+)*(?:\s*;\s*)?$'
)

def filter_java_kotlin_minimal(src: str) -> str:
    original = src

    # 1) Check ONLY the first leading block comment for copyright, remove only if it contains copyright
    m = _TOP_BLOCK_COMMENTS.match(src)
    if m:
        first_comment_block = m.group(0)
        if 'copyright' in first_comment_block.lower():
            # Remove just this one copyright block and stop
            src = src[m.end():]

    # 2) Process lines to remove imports but preserve javadocs and code
    lines = src.split('\n')
    filtered_lines = []
    in_javadoc = False
    in_string = False
    string_char = None
    escape_next = False
    
    for line in lines:
        stripped = line.strip()
        char_index = 0
        
        # Skip if we're in a javadoc
        if in_javadoc:
            filtered_lines.append(line)
            if '*/' in line:
                in_javadoc = False
            continue
        
        # Check for javadoc start
        if not in_javadoc and '/**' in line:
            filtered_lines.append(line)
            in_javadoc = True
            continue
        
        # Only remove package/import lines when NOT in comments or strings
        if (_JAVA_KOTLIN_PKG_IMPORT.match(line) and 
            not stripped.startswith('//') and  # Don't remove commented imports
            not any(word in stripped.lower() for word in ['class ', 'interface ', 'enum ', '=', '{', '}'])):  # Don't remove if it contains other constructs
            continue
        
        filtered_lines.append(line)
    
    src = '\n'.join(filtered_lines)

    # 3) Tidy up blank lines but be less aggressive
    src = re.sub(r'\n{3,}', '\n\n', src).lstrip('\n')

    # Fallback if we stripped everything
    if not src.strip():
        return original.strip()
    return src


def filter_file_preview(file_path: str, src: str) -> str:
    ext = os.path.splitext(file_path.lower())[1]
    if ext == '.py':
        return filter_python_minimal(src)
    if ext in ('.java', '.kt', '.kts'):
        return filter_java_kotlin_minimal(src)
    return src


def read_file_safe(path: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content, None
    except FileNotFoundError:
        return None, "not found"
    except UnicodeDecodeError:
        # Fallback for different encodings
        try:
            with open(path, 'r', encoding='latin-1') as f:
                content = f.read()
            return content, None
        except Exception as e:
            return None, f"read error: {str(e)}"