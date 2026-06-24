"""
Agent Trajectory Viewer for Google Colab
-----------------------------------------
Renders an interactive HTML timeline from a bug-localization agent trajectory
log (.jsonl). Each line in the file is a JSON event emitted by the pipeline.

Usage in Colab:
    # Option A — load from a file path
    render_trajectory("my_trajectory.jsonl")

    # Option B — load from a string / already-parsed list
    render_trajectory(events=my_list_of_dicts)

    # Option C — load from a URL
    render_trajectory("https://example.com/trajectory.jsonl")
"""

import json
import textwrap
from datetime import datetime
from pathlib import Path
from IPython.display import HTML, display


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_events(source) -> list[dict]:
    """Accept a file path (str/Path), a URL, or a list of dicts."""
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


def _categorise(ev: dict) -> str:
    """Map raw event type → display category."""
    t = ev.get("type", "")
    mapping = {
        "start":            "meta",
        "end":              "meta",
        "summary":          "meta",
        "token_count":      "meta",
        "reprompt":         "error",
        "reprompt_success": "meta",
        "parse":            "meta",
        "preview_filter":   "meta",
        "bm25_query":       "meta",
        "prompt":           "prompt",
        "response":         "response",
        "tool_decision":    "tool",
        "tool_result":      "result",
        "self_eval":        "eval",
        "self_eval_applied":"eval",
        "final_answer":     "eval",
    }
    return mapping.get(t, "meta")


def _make_label(ev: dict) -> str:
    """Generate a human-readable one-line label for an event."""
    t    = ev.get("type", "?")
    step = ev.get("step", "")

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
        keys = [k for k in ev.keys() if k not in ("type","step","run_id","ts")]
        return f"Tool result — {', '.join(keys[:4])}"
    if t == "bm25_query":
        return f"BM25 query — {ev.get('query_preview','')[:60]}…"
    if t == "parse":
        return f"Parse {'✓' if ev.get('ok') else '✗'} — {step}"
    if t == "reprompt":
        return f"Re-prompt (attempt {ev.get('attempt','?')}) — format failure"
    if t == "reprompt_success":
        return f"Re-prompt succeeded — {len(ev.get('ranked_files',[]))} files returned"
    if t in ("self_eval", "self_eval_applied", "final_answer"):
        files = ev.get("ranked_files", [])
        lbl   = {"self_eval":"Self-eval prompt","self_eval_applied":"Self-eval applied","final_answer":"Final answer"}.get(t, t)
        return f"{lbl} — top file: {files[0].split('/')[-1] if files else '?'}"
    if t == "preview_filter":
        return f"File preview filtered — removed {ev.get('removed_lines',0)} lines"
    return f"{t} — {step}"


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


# ── expandable code block ─────────────────────────────────────────────────────
# Each truncated block gets a unique id so the toggle button can find it.

_EXPAND_THRESHOLD_CHARS = 400   # show this many chars before offering "show more"
_block_counter = 0

def _expandable_code(text: str) -> str:
    """
    Return an HTML snippet with the first _EXPAND_THRESHOLD_CHARS visible and
    the rest hidden behind a 'show full ▾' / 'show less ▴' toggle button.
    If the text is short enough, just return a plain code block.
    """
    global _block_counter
    escaped = _esc(text)

    if len(text) <= _EXPAND_THRESHOLD_CHARS:
        return f'<div class="traj-code">{escaped}</div>'

    _block_counter += 1
    bid = f"traj-block-{_block_counter}"
    preview  = _esc(text[:_EXPAND_THRESHOLD_CHARS])
    full     = escaped   # full already escaped above

    return (
        f'<div class="traj-code" id="{bid}-pre">{preview}'
        f'<span class="traj-ellipsis">…</span></div>'
        f'<div class="traj-code traj-hidden" id="{bid}-full">{full}</div>'
        f'<button class="traj-expand-btn" '
        f'  onclick="trajToggle(\'{bid}\')" '
        f'  id="{bid}-btn">show full ▾</button>'
    )


# ── HTML builder ─────────────────────────────────────────────────────────────

_CSS = """
<style>
#traj-viewer * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
#traj-viewer { padding: 12px 0; max-width: 900px; }
.traj-controls { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 14px; align-items: center; }
.traj-controls span { font-size: 12px; color: #888; margin-right: 4px; }
.traj-btn { font-size: 12px; padding: 4px 11px; border-radius: 99px; border: 1px solid #d0d0d0; background: transparent; color: #555; cursor: pointer; }
.traj-btn:hover { background: #f5f5f5; }
.traj-btn.active { background: #e8f0fe; color: #1a56db; border-color: #a8c2f8; }
.traj-divider { font-size: 10px; text-transform: uppercase; letter-spacing: .08em; color: #aaa; padding: 14px 0 5px; }
.traj-event { border-radius: 8px; margin-bottom: 2px; overflow: hidden; }
.traj-header { display: flex; align-items: center; gap: 9px; padding: 9px 12px; cursor: pointer; border-radius: 8px; }
.traj-header:hover { background: #f7f7f7; }
.traj-badge { font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 99px; flex-shrink: 0; }
.cat-prompt   .traj-badge { background:#ede9fe; color:#5b21b6; }
.cat-response .traj-badge { background:#d1fae5; color:#065f46; }
.cat-tool     .traj-badge { background:#fef3c7; color:#92400e; }
.cat-result   .traj-badge { background:#dbeafe; color:#1e40af; }
.cat-error    .traj-badge { background:#fee2e2; color:#991b1b; }
.cat-meta     .traj-badge { background:#f3f4f6; color:#374151; }
.cat-eval     .traj-badge { background:#fce7f3; color:#9d174d; }
.traj-label { font-size: 13px; font-weight: 500; color: #111; flex: 1; min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.traj-ts    { font-size: 11px; color: #aaa; white-space: nowrap; }
.traj-chev  { font-size: 13px; color: #bbb; transition: transform .2s; }
.traj-event.open .traj-chev { transform: rotate(180deg); }
.traj-body  { display: none; padding: 0 12px 12px 12px; }
.traj-event.open .traj-body { display: block; }
.traj-kv    { display: flex; gap: 10px; font-size: 12px; padding: 3px 0; border-bottom: 1px solid #f0f0f0; }
.traj-kv:last-child { border-bottom: none; }
.traj-kv-k  { color: #888; flex-shrink: 0; width: 110px; }
.traj-kv-v  { color: #222; word-break: break-all; }
.traj-code  { font-family: 'SFMono-Regular', Consolas, monospace; font-size: 11.5px; background: #f8f8f8; border: 1px solid #eee; border-radius: 6px; padding: 8px 10px; overflow-x: auto; white-space: pre-wrap; word-break: break-all; color: #333; line-height: 1.55; margin-top: 6px; }
.traj-hidden { display: none; }
.traj-ellipsis { color: #aaa; }
.traj-expand-btn { margin-top: 4px; font-size: 11px; color: #1a56db; background: none; border: none; cursor: pointer; padding: 0; }
.traj-expand-btn:hover { text-decoration: underline; }
.traj-files { display: flex; flex-direction: column; gap: 3px; margin-top: 6px; }
.traj-chip  { font-size: 11.5px; font-family: monospace; padding: 3px 9px; background: #f3f4f6; border-radius: 6px; color: #555; display: flex; align-items: center; gap: 8px; }
.traj-rank  { font-size: 11px; font-weight: 600; width: 16px; text-align: center; color: #aaa; }
.traj-score { margin-left: auto; font-size: 11px; color: #aaa; }
.traj-token-wrap { margin-top: 8px; }
.traj-token-lbl  { font-size: 11px; color: #888; margin-bottom: 3px; }
.traj-token-bg   { height: 5px; border-radius: 99px; background: #eee; overflow: hidden; }
.traj-token-fill { height: 100%; border-radius: 99px; background: #7c3aed; }
</style>
"""

_JS = """
<script>
function trajToggle(bid) {
  var pre  = document.getElementById(bid + '-pre');
  var full = document.getElementById(bid + '-full');
  var btn  = document.getElementById(bid + '-btn');
  var expanded = !full.classList.contains('traj-hidden');
  if (expanded) {
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
      b.classList.toggle('active', b === btn);
    });
    viewer.querySelectorAll('.traj-event, .traj-divider').forEach(function(el){
      if (el.classList.contains('traj-divider')) { el.style.display = ''; return; }
      el.style.display = (active === 'all' || el.dataset.cat === active) ? '' : 'none';
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
    _block_counter = 0          # reset per render so ids are stable

    cats       = ["all", "prompt", "response", "tool", "result", "eval", "error", "meta"]
    cat_labels = {"all":"all","prompt":"prompts","response":"responses",
                  "tool":"tool calls","result":"results","eval":"eval",
                  "error":"errors","meta":"meta"}

    btns = '<span>show:</span>' + "".join(
        f'<button class="traj-btn{" active" if c=="all" else ""}" data-cat="{c}">{cat_labels[c]}</button>'
        for c in cats
    )

    rows      = []
    last_step = None

    for ev in events:
        cat   = _categorise(ev)
        label = _make_label(ev)
        ts    = _fmt_ts(ev.get("ts", ""))
        step  = ev.get("step", "")

        if step != last_step:
            rows.append(f'<div class="traj-divider">{_esc(step)}</div>')
            last_step = step

        body_parts = []

        # ── token bar ────────────────────────────────────────────────────────
        if ev.get("type") == "token_count":
            n, ctx = ev.get("token_count", 0), ev.get("context_size", 1)
            pct = round(n / ctx * 100)
            body_parts.append(
                f'<div class="traj-token-wrap">'
                f'<div class="traj-token-lbl">{n:,} / {ctx:,} tokens ({pct}%)</div>'
                f'<div class="traj-token-bg"><div class="traj-token-fill" style="width:{pct}%"></div></div>'
                f'</div>'
            )

        # ── ranked files ─────────────────────────────────────────────────────
        ranked = (ev.get("ranked_files") or
                  (ev.get("response") and _try_parse_files(ev.get("response", ""))))
        if ranked and isinstance(ranked, list) and ranked and isinstance(ranked[0], str):
            scores = ev.get("confidence_scores", [])
            body_parts.append('<div class="traj-files">')
            for i, f in enumerate(ranked):
                score = scores[i] if i < len(scores) else ""
                body_parts.append(
                    f'<div class="traj-chip">'
                    f'<span class="traj-rank">{i+1}</span>'
                    f'<span>{_esc(f)}</span>'
                    f'{"<span class=traj-score>score: " + _esc(str(score)) + "</span>" if score else ""}'
                    f'</div>'
                )
            body_parts.append('</div>')

        # ── BM25 file list ────────────────────────────────────────────────────
        if ev.get("type") == "tool_result" and ev.get("files"):
            scores = ev.get("scores", [])
            body_parts.append('<div class="traj-files">')
            for i, f in enumerate(ev["files"]):
                sc = f"{scores[i]:.2f}" if i < len(scores) else ""
                body_parts.append(
                    f'<div class="traj-chip">'
                    f'<span class="traj-rank">{i+1}</span>'
                    f'<span>{_esc(f)}</span>'
                    f'{"<span class=traj-score>" + sc + "</span>" if sc else ""}'
                    f'</div>'
                )
            body_parts.append('</div>')

        # ── file content view ─────────────────────────────────────────────────
        if ev.get("type") == "tool_result" and ev.get("view_file"):
            content = ev.get("content", "")
            body_parts.append(
                f'<div class="traj-kv">'
                f'<span class="traj-kv-k">file</span>'
                f'<span class="traj-kv-v">{_esc(ev["view_file"])}</span>'
                f'</div>'
            )
            body_parts.append(_expandable_code(content))

        # ── messages (prompts) ────────────────────────────────────────────────
        if ev.get("messages"):
            for msg in ev["messages"]:
                role    = msg.get("role", "?")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        c.get("text", "") for c in content if isinstance(c, dict)
                    )
                body_parts.append(
                    f'<div class="traj-kv">'
                    f'<span class="traj-kv-k">[{_esc(role)}]</span>'
                    f'</div>'
                )
                body_parts.append(_expandable_code(content))

        # ── general key-value rows ────────────────────────────────────────────
        skip = {"type","step","run_id","ts","ranked_files","confidence_scores",
                "files","scores","view_file","content","messages"}
        for k, v in ev.items():
            if k in skip:
                continue
            val = json.dumps(v, indent=2) if isinstance(v, (dict, list)) else str(v)
            if "\n" in val or len(val) > 100:
                body_parts.append(
                    f'<div class="traj-kv"><span class="traj-kv-k">{_esc(k)}</span></div>'
                )
                body_parts.append(_expandable_code(val))
            else:
                body_parts.append(
                    f'<div class="traj-kv">'
                    f'<span class="traj-kv-k">{_esc(k)}</span>'
                    f'<span class="traj-kv-v">{_esc(val)}</span>'
                    f'</div>'
                )

        body_html  = "".join(body_parts)
        badge_text = cat_labels.get(cat, cat)

        rows.append(
            f'<div class="traj-event cat-{cat}" data-cat="{cat}">'
            f'  <div class="traj-header">'
            f'    <span class="traj-badge">{badge_text}</span>'
            f'    <span class="traj-label">{_esc(label)}</span>'
            f'    <span class="traj-ts">{ts}</span>'
            f'    <span class="traj-chev">▾</span>'
            f'  </div>'
            f'  <div class="traj-body">{body_html}</div>'
            f'</div>'
        )

    return (
        _CSS
        + f'<div id="traj-viewer">'
          f'  <div class="traj-controls">{btns}</div>'
          f'  <div class="traj-timeline">{"".join(rows)}</div>'
          f'</div>'
        + _JS
    )


# ── public API ────────────────────────────────────────────────────────────────

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
        raise ValueError("Provide either source= (file path / URL) or events= (list of dicts).")
    if events is None:
        events = _load_events(source)

    print(f"Loaded {len(events)} events.")
    html = _build_html(events)

    wrapped = f"""
    <div style="border:1px solid #e0e0e0; border-radius:8px; overflow:auto;
                height:{height}px; padding:8px 12px; background:#fff;">
      {html}
    </div>
    """
    display(HTML(wrapped))


# ── CLI ───────────────────────────────────────────────────────────────────────

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