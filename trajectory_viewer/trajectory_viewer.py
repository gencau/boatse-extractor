"""
Agent Trajectory Viewer for Google Colab
-----------------------------------------
Renders an interactive HTML timeline from a bug-localization agent trajectory
log (.jsonl). Each line in the file is a JSON event emitted by the pipeline.

Usage in Colab:
    render_trajectory("my_trajectory.jsonl")
    render_trajectory("https://example.com/trajectory.jsonl")
    render_trajectory(events=my_list_of_dicts)
"""

import json
import textwrap
from datetime import datetime
from pathlib import Path
from IPython.display import HTML, display


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_events(source) -> list[dict]:
    if isinstance(source, (list, tuple)):
        return list(source)
    source = str(source)
    if source.startswith("http://") or source.startswith("https://"):
        import urllib.request
        with urllib.request.urlopen(source) as r:
            raw = r.read().decode()
    else:
        raw = Path(source).read_text()
    events = []
    for i, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  [skip] line {i}: {e}")
    return events


# ─────────────────────────────────────────────────────────────────────────────
# Grouping — merge file-view triplets into single logical events
# ─────────────────────────────────────────────────────────────────────────────

def _group_events(events: list[dict]) -> list[dict]:
    """
    Merge consecutive (tool_decision → preview_filter? → tool_result) triplets
    that represent a single file view into one synthetic event so they render
    as a single card instead of three disconnected rows.

    All other events pass through unchanged.
    """
    out = []
    i   = 0
    while i < len(events):
        ev = events[i]

        # Detect a file-view tool_decision
        if (ev.get("type") == "tool_decision"
                and ev.get("tool_name") == "view_file"):

            merged = {
                "type":      "file_view_group",
                "step":      ev.get("step", ""),
                "ts":        ev.get("ts", ""),
                "file_path": ev.get("tool_args", {}).get("file_path", "?"),
                "removed_lines": None,
                "content":   None,
                "run_id":    ev.get("run_id", ""),
            }

            j = i + 1
            # consume optional preview_filter
            if j < len(events) and events[j].get("type") == "preview_filter":
                merged["removed_lines"] = events[j].get("removed_lines", 0)
                j += 1
            # consume the tool_result that carries the content
            if j < len(events) and events[j].get("type") == "tool_result":
                merged["content"] = events[j].get("content", "")
                j += 1

            out.append(merged)
            i = j
            continue

        out.append(ev)
        i += 1

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Categorisation & labels
# ─────────────────────────────────────────────────────────────────────────────

def _categorise(ev: dict) -> str:
    mapping = {
        "start":             "meta",
        "end":               "meta",
        "summary":           "meta",
        "token_count":       "meta",
        "reprompt":          "error",
        "reprompt_success":  "meta",
        "parse":             "meta",
        "preview_filter":    "meta",
        "bm25_query":        "meta",
        "prompt":            "prompt",
        "response":          "response",
        "tool_decision":     "tool",
        "tool_result":       "result",
        "file_view_group":   "tool",
        "self_eval":         "eval",
        "self_eval_applied": "eval",
        "final_answer":      "eval",
    }
    return mapping.get(ev.get("type", ""), "meta")


def _make_label(ev: dict) -> str:
    t    = ev.get("type", "?")
    step = ev.get("step", "")

    if t == "file_view_group":
        path     = ev.get("file_path", "?")
        removed  = ev.get("removed_lines")
        truncated = f"  ·  {removed} lines removed by pipeline" if removed else ""
        return f"File view — {path.split('/')[-1]}{truncated}"
    if t == "start":
        return f"Pipeline started — {ev.get('dataset','?')} / {ev.get('repo_name','?')[:40]}"
    if t == "end":
        return f"Pipeline complete — reprompts: {ev.get('reprompts', 0)}"
    if t == "summary":
        return f"Summary — model: {ev.get('model','?')}  events: {ev.get('events','?')}"
    if t == "token_count":
        n, ctx = ev.get("token_count", 0), ev.get("context_size", 0)
        pct = round(n / ctx * 100) if ctx else 0
        return f"Token count: {n:,} / {ctx:,}  ({pct}%)"
    if t == "prompt":
        msgs  = ev.get("messages", [])
        roles = [m["role"] for m in msgs]
        return f"Prompt — {len(msgs)} message(s)  [{', '.join(roles)}]"
    if t == "response":
        r = ev.get("response", "")
        try:
            obj = json.loads(r)
            if "function_call" in obj:
                return f"Response — calls {obj['function_call']}()"
            if "ranked_files" in obj:
                return f"Response — ranked_files ({len(obj['ranked_files'])} files)"
        except Exception:
            pass
        return f"Response — {str(r)[:80]}"
    if t == "tool_decision":
        name    = ev.get("tool_name", "?")
        args    = ev.get("tool_args", {})
        arg_str = ", ".join(f"{k}={str(v)[:30]}" for k, v in args.items())
        return f"Tool call — {name}({arg_str})"
    if t == "tool_result":
        # detect extract_relevant result (dynamic key)
        er = _extract_relevant_payload(ev)
        if er:
            return f"Tool result — extract_relevant · {er.get('filename','?')}"
        keys = [k for k in ev.keys() if k not in ("type","step","run_id","ts")]
        return f"Tool result — {', '.join(keys[:4])}"
    if t == "bm25_query":
        return f"BM25 query — {ev.get('query_preview','')[:70]}…"
    if t == "parse":
        return f"Parse {'✓' if ev.get('ok') else '✗'} — {step}"
    if t == "reprompt":
        return f"Re-prompt (attempt {ev.get('attempt','?')}) — format failure"
    if t == "reprompt_success":
        files = ev.get("ranked_files", [])
        top   = files[0].split("/")[-1] if files else "?"
        return f"Re-prompt succeeded — {len(files)} files · top: {top}"
    if t in ("self_eval", "self_eval_applied", "final_answer"):
        files = ev.get("ranked_files", [])
        lbl   = {"self_eval":"Self-eval prompt",
                 "self_eval_applied":"Self-eval applied",
                 "final_answer":"Final answer"}.get(t, t)
        return f"{lbl} — top file: {files[0].split('/')[-1] if files else '?'}"
    if t == "preview_filter":
        return f"File preview filtered — removed {ev.get('removed_lines',0)} lines from {ev.get('file','?').split('/')[-1]}"
    return f"{t} — {step}"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_ts(ts_str: str) -> str:
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.strftime("%H:%M:%S")
    except Exception:
        return ts_str[:19]


def _esc(s: str) -> str:
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def _try_parse_files(response_str: str):
    try:
        return json.loads(response_str).get("ranked_files")
    except Exception:
        return None


def _extract_relevant_payload(ev: dict):
    """
    The extract_relevant result is stored under the key 'extract_relevant'
    (same name as the tool), not under a fixed key like 'content'.
    """
    return ev.get("extract_relevant") or ev.get("tool_result")


# ─────────────────────────────────────────────────────────────────────────────
# Expandable code blocks
# ─────────────────────────────────────────────────────────────────────────────

_EXPAND_THRESHOLD = 400
_block_counter    = 0


def _expandable_code(text: str, label: str = "") -> str:
    """
    Short text  → plain code block.
    Long text   → preview + 'show full ▾' / 'show less ▴' toggle.
    Optional label renders as a small header above the block.
    """
    global _block_counter
    escaped = _esc(text)
    header  = f'<div class="traj-code-label">{_esc(label)}</div>' if label else ""

    if len(text) <= _EXPAND_THRESHOLD:
        return f'{header}<div class="traj-code">{escaped}</div>'

    _block_counter += 1
    bid = f"tbl-{_block_counter}"
    preview = _esc(text[:_EXPAND_THRESHOLD])

    return (
        f'{header}'
        f'<div class="traj-code" id="{bid}-pre">{preview}'
        f'<span class="traj-ellipsis"> …</span></div>'
        f'<div class="traj-code traj-hidden" id="{bid}-full">{escaped}</div>'
        f'<button class="traj-expand-btn" onclick="trajToggle(\'{bid}\')" '
        f'id="{bid}-btn">show full ▾</button>'
    )


def _files_html(files: list, scores: list = None) -> str:
    scores = scores or []
    parts  = ['<div class="traj-files">']
    for i, f in enumerate(files):
        sc = f"{scores[i]:.2f}" if i < len(scores) else ""
        parts.append(
            f'<div class="traj-chip">'
            f'<span class="traj-rank">{i+1}</span>'
            f'<span>{_esc(f)}</span>'
            f'{"<span class=traj-score>" + sc + "</span>" if sc else ""}'
            f'</div>'
        )
    parts.append('</div>')
    return "".join(parts)


def _kv(key: str, val: str) -> str:
    return (f'<div class="traj-kv">'
            f'<span class="traj-kv-k">{_esc(key)}</span>'
            f'<span class="traj-kv-v">{_esc(val)}</span>'
            f'</div>')


def _kv_header(key: str) -> str:
    return (f'<div class="traj-kv">'
            f'<span class="traj-kv-k">{_esc(key)}</span>'
            f'</div>')


# ─────────────────────────────────────────────────────────────────────────────
# Body builder — one function per event type
# ─────────────────────────────────────────────────────────────────────────────

def _body(ev: dict) -> str:
    t     = ev.get("type", "")
    parts = []

    # ── file view group (merged triplet) ──────────────────────────────────────
    if t == "file_view_group":
        parts.append(_kv("file", ev.get("file_path", "?")))
        if ev.get("removed_lines"):
            parts.append(_kv("pipeline truncation",
                             f"{ev['removed_lines']} lines removed before agent saw this"))
        content = ev.get("content") or "(no content captured)"
        parts.append(_expandable_code(content, label="file content"))
        return "".join(parts)

    # ── token count ───────────────────────────────────────────────────────────
    if t == "token_count":
        n, ctx = ev.get("token_count", 0), ev.get("context_size", 1)
        pct = round(n / ctx * 100)
        parts.append(
            f'<div class="traj-token-wrap">'
            f'<div class="traj-token-lbl">{n:,} / {ctx:,} tokens ({pct}%)</div>'
            f'<div class="traj-token-bg">'
            f'<div class="traj-token-fill" style="width:{pct}%"></div>'
            f'</div></div>'
        )
        return "".join(parts)

    # ── BM25 query ────────────────────────────────────────────────────────────
    if t == "bm25_query":
        query = ev.get("query_preview", "")
        parts.append(_kv("used extracted info", str(ev.get("used_extracted", ""))))
        parts.append(_expandable_code(query, label="query text"))
        return "".join(parts)

    # ── prompt ────────────────────────────────────────────────────────────────
    if t == "prompt":
        for msg in ev.get("messages", []):
            role    = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(c.get("text", "") for c in content
                                   if isinstance(c, dict))
            parts.append(_expandable_code(content, label=f"[{role}]"))
        return "".join(parts)

    # ── response ──────────────────────────────────────────────────────────────
    if t == "response":
        raw = ev.get("response", "")
        # ranked files embedded in the response JSON
        ranked = _try_parse_files(raw)
        if ranked:
            scores = []
            parts.append(_files_html(ranked, scores))
        else:
            parts.append(_expandable_code(raw, label="raw response"))
        return "".join(parts)

    # ── tool_decision (non-file-view) ─────────────────────────────────────────
    if t == "tool_decision":
        parts.append(_kv("tool", ev.get("tool_name", "?")))
        for k, v in ev.get("tool_args", {}).items():
            val = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            parts.append(_kv(k, val))
        return "".join(parts)

    # ── tool_result ───────────────────────────────────────────────────────────
    if t == "tool_result":
        # extract_relevant result
        er = _extract_relevant_payload(ev)
        if er and isinstance(er, dict):
            for k, v in er.items():
                val = json.dumps(v, indent=2) if isinstance(v, (dict, list)) else str(v)
                if "\n" in val or len(val) > 100:
                    parts.append(_kv_header(k))
                    parts.append(_expandable_code(val))
                else:
                    parts.append(_kv(k, val))
            return "".join(parts)

        # BM25 file list result
        if ev.get("files"):
            parts.append(_files_html(ev["files"], ev.get("scores", [])))
            return "".join(parts)

        # file content result (ungrouped fallback)
        if ev.get("view_file"):
            parts.append(_kv("file", ev["view_file"]))
            parts.append(_expandable_code(ev.get("content", ""), label="file content"))
            return "".join(parts)

        # generic fallback
        skip = {"type","step","run_id","ts"}
        for k, v in ev.items():
            if k in skip:
                continue
            val = json.dumps(v, indent=2) if isinstance(v, (dict, list)) else str(v)
            if "\n" in val or len(val) > 100:
                parts.append(_kv_header(k))
                parts.append(_expandable_code(val))
            else:
                parts.append(_kv(k, val))
        return "".join(parts)

    # ── reprompt ──────────────────────────────────────────────────────────────
    if t == "reprompt":
        parts.append(_kv("attempt", str(ev.get("attempt", "?"))))
        parts.append(_expandable_code(ev.get("last_msg", ""), label="system message"))
        return "".join(parts)

    # ── reprompt_success ──────────────────────────────────────────────────────
    if t == "reprompt_success":
        ranked = ev.get("ranked_files", [])
        if ranked:
            parts.append(_files_html(ranked))
        return "".join(parts)

    # ── ranked file lists (eval / final_answer / end) ─────────────────────────
    if ev.get("ranked_files"):
        ranked = ev["ranked_files"]
        scores = ev.get("confidence_scores", [])
        parts.append(_files_html(ranked, scores))

    # ── generic key-value fallback ────────────────────────────────────────────
    skip = {"type","step","run_id","ts","ranked_files","confidence_scores",
            "files","scores","messages","response","content","view_file",
            "extract_relevant","tool_result"}
    for k, v in ev.items():
        if k in skip:
            continue
        val = json.dumps(v, indent=2) if isinstance(v, (dict, list)) else str(v)
        if "\n" in val or len(val) > 100:
            parts.append(_kv_header(k))
            parts.append(_expandable_code(val))
        else:
            parts.append(_kv(k, val))

    return "".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# HTML assembly
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
<style>
#traj-viewer * { box-sizing:border-box; margin:0; padding:0;
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }
#traj-viewer { padding:12px 0; max-width:900px; }
.traj-controls { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:14px; align-items:center; }
.traj-controls span { font-size:12px; color:#888; margin-right:4px; }
.traj-btn { font-size:12px; padding:4px 11px; border-radius:99px;
  border:1px solid #d0d0d0; background:transparent; color:#555; cursor:pointer; }
.traj-btn:hover { background:#f5f5f5; }
.traj-btn.active { background:#e8f0fe; color:#1a56db; border-color:#a8c2f8; }
.traj-divider { font-size:10px; text-transform:uppercase; letter-spacing:.08em;
  color:#aaa; padding:14px 0 5px; }
.traj-event { border-radius:8px; margin-bottom:2px; overflow:hidden; }
.traj-header { display:flex; align-items:center; gap:9px; padding:9px 12px;
  cursor:pointer; border-radius:8px; }
.traj-header:hover { background:#f7f7f7; }
.traj-badge { font-size:11px; font-weight:600; padding:2px 8px;
  border-radius:99px; flex-shrink:0; }
.cat-prompt   .traj-badge { background:#ede9fe; color:#5b21b6; }
.cat-response .traj-badge { background:#d1fae5; color:#065f46; }
.cat-tool     .traj-badge { background:#fef3c7; color:#92400e; }
.cat-result   .traj-badge { background:#dbeafe; color:#1e40af; }
.cat-error    .traj-badge { background:#fee2e2; color:#991b1b; }
.cat-meta     .traj-badge { background:#f3f4f6; color:#374151; }
.cat-eval     .traj-badge { background:#fce7f3; color:#9d174d; }
.traj-label { font-size:13px; font-weight:500; color:#111; flex:1; min-width:0;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.traj-ts    { font-size:11px; color:#aaa; white-space:nowrap; }
.traj-chev  { font-size:13px; color:#bbb; transition:transform .2s; flex-shrink:0; }
.traj-event.open .traj-chev { transform:rotate(180deg); }
.traj-body  { display:none; padding:4px 12px 14px 12px; }
.traj-event.open .traj-body { display:block; }
.traj-kv    { display:flex; gap:10px; font-size:12px; padding:3px 0;
  border-bottom:1px solid #f0f0f0; }
.traj-kv:last-of-type { border-bottom:none; }
.traj-kv-k  { color:#888; flex-shrink:0; width:130px; }
.traj-kv-v  { color:#222; word-break:break-all; }
.traj-code  { font-family:'SFMono-Regular',Consolas,monospace; font-size:11.5px;
  background:#f8f8f8; border:1px solid #eee; border-radius:6px;
  padding:8px 10px; overflow-x:auto; white-space:pre-wrap;
  word-break:break-all; color:#333; line-height:1.55; margin-top:4px; }
.traj-code-label { font-size:11px; color:#888; margin-top:8px; margin-bottom:2px; }
.traj-hidden { display:none; }
.traj-ellipsis { color:#bbb; }
.traj-expand-btn { margin-top:3px; font-size:11px; color:#1a56db;
  background:none; border:none; cursor:pointer; padding:0; }
.traj-expand-btn:hover { text-decoration:underline; }
.traj-files { display:flex; flex-direction:column; gap:3px; margin-top:6px; }
.traj-chip  { font-size:11.5px; font-family:monospace; padding:3px 9px;
  background:#f3f4f6; border-radius:6px; color:#555;
  display:flex; align-items:center; gap:8px; }
.traj-rank  { font-size:11px; font-weight:600; width:16px;
  text-align:center; color:#aaa; }
.traj-score { margin-left:auto; font-size:11px; color:#aaa; }
.traj-token-wrap { margin-top:8px; }
.traj-token-lbl  { font-size:11px; color:#888; margin-bottom:3px; }
.traj-token-bg   { height:5px; border-radius:99px; background:#eee; overflow:hidden; }
.traj-token-fill { height:100%; border-radius:99px; background:#7c3aed; }
</style>
"""

_JS = """
<script>
function trajToggle(bid) {
  var pre  = document.getElementById(bid+'-pre');
  var full = document.getElementById(bid+'-full');
  var btn  = document.getElementById(bid+'-btn');
  var isExpanded = !full.classList.contains('traj-hidden');
  if (isExpanded) {
    full.classList.add('traj-hidden');
    pre.classList.remove('traj-hidden');
    btn.textContent = 'show full \u25be';
  } else {
    full.classList.remove('traj-hidden');
    pre.classList.add('traj-hidden');
    btn.textContent = 'show less \u25b4';
  }
}
(function(){
  var viewer = document.getElementById('traj-viewer');
  var active = 'all';
  viewer.querySelector('.traj-controls').addEventListener('click', function(e){
    var btn = e.target.closest('.traj-btn');
    if (!btn) return;
    active = btn.dataset.cat;
    viewer.querySelectorAll('.traj-btn').forEach(function(b){
      b.classList.toggle('active', b===btn);
    });
    viewer.querySelectorAll('.traj-event,.traj-divider').forEach(function(el){
      if (el.classList.contains('traj-divider')) { el.style.display=''; return; }
      el.style.display = (active==='all' || el.dataset.cat===active) ? '' : 'none';
    });
  });
  viewer.querySelector('.traj-timeline').addEventListener('click', function(e){
    if (e.target.classList.contains('traj-expand-btn')) return;
    var hdr = e.target.closest('.traj-header');
    if (!hdr) return;
    hdr.closest('.traj-event').classList.toggle('open');
  });
})();
</script>
"""


def _build_html(events: list[dict]) -> str:
    global _block_counter
    _block_counter = 0

    grouped = _group_events(events)

    cats       = ["all","prompt","response","tool","result","eval","error","meta"]
    cat_labels = {"all":"all","prompt":"prompts","response":"responses",
                  "tool":"tool calls","result":"results","eval":"eval",
                  "error":"errors","meta":"meta"}

    btns = '<span>show:</span>' + "".join(
        f'<button class="traj-btn{" active" if c=="all" else ""}" '
        f'data-cat="{c}">{cat_labels[c]}</button>'
        for c in cats
    )

    rows      = []
    last_step = None

    for ev in grouped:
        cat   = _categorise(ev)
        label = _make_label(ev)
        ts    = _fmt_ts(ev.get("ts", ""))
        step  = ev.get("step", "")

        if step != last_step:
            rows.append(f'<div class="traj-divider">{_esc(step)}</div>')
            last_step = step

        body_html  = _body(ev)
        badge_text = cat_labels.get(cat, cat)

        rows.append(
            f'<div class="traj-event cat-{cat}" data-cat="{cat}">'
            f'<div class="traj-header">'
            f'<span class="traj-badge">{badge_text}</span>'
            f'<span class="traj-label">{_esc(label)}</span>'
            f'<span class="traj-ts">{ts}</span>'
            f'<span class="traj-chev">▾</span>'
            f'</div>'
            f'<div class="traj-body">{body_html}</div>'
            f'</div>'
        )

    return (
        _CSS
        + f'<div id="traj-viewer">'
          f'<div class="traj-controls">{btns}</div>'
          f'<div class="traj-timeline">{"".join(rows)}</div>'
          f'</div>'
        + _JS
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def render_trajectory(source=None, *, events=None, height: int = 700):
    """
    Render an agent trajectory log as an interactive HTML timeline in Colab.

    Parameters
    ----------
    source : str or Path, optional
        Path to a .jsonl file, or a URL.
    events : list[dict], optional
        Already-loaded list of event dicts (alternative to source).
    height : int
        Height of the scrollable iframe in pixels (default 700).
    """
    if events is None and source is None:
        raise ValueError("Provide source= (path/URL) or events= (list of dicts).")
    if events is None:
        events = _load_events(source)

    print(f"Loaded {len(events)} raw events.")
    html = _build_html(events)

    wrapped = (
        f'<div style="border:1px solid #e0e0e0; border-radius:8px; overflow:auto;'
        f'height:{height}px; padding:8px 12px; background:#fff;">'
        f'{html}</div>'
    )
    display(HTML(wrapped))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        render_trajectory(sys.argv[1])
    else:
        print(textwrap.dedent("""
            Usage:
              python trajectory_viewer.py path/to/trajectory.jsonl

            Or in a Colab cell:
              from trajectory_viewer import render_trajectory
              render_trajectory("trajectory.jsonl")
              render_trajectory("https://example.com/trajectory.jsonl")
              render_trajectory(events=my_list_of_dicts)
        """))
