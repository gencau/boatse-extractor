from dotenv import load_dotenv
import pandas as pd
import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, Tuple, List, Optional

from .prompts.agent_context_prompt import AgentContextPrompt
from .logging.logger import RunLogger, _fmt_messages
from utils.validate_output import extract_ranked_files_from_any
from utils.git_utils import parse_changed_files_from_diff
from utils.json_utils import parse_json_safe
from utils.path_utils import is_missing_path, validate_ranked_files
from utils.prompt_utils import append_user_prompt, to_assistant_content
from .agent.run_context import RunContext
from .tools.tools import exec_tool

import utils.tokenization_utils as tk
from .workers.chat_request_process import CallTimeout, run_with_timeout
from .prompts.prompts import FUNCTION_CALLS_TURN_1, FUNCTION_CALLS_TURN_2, FUNCTION_CALLS_TURN_3, \
                          SYSTEM_PROMPT, PROMPT_TURN_1, PROMPT_TURN_2, PROMPT_TURN_3, PROMPT_TOOL_ERROR_FILE_PATH, \
                          PROMPT_RANKED_FILES_INVALID, ANSWER_REVIEW_PROMPT_W_VIEW, ANSWER_REVIEW_SYSTEM_PROMPT, \
                          SELF_EVAL_POST_VIEW_PROMPT, PROMPT_MAX_FILE_VIEWS

# =========================
# Constants / Templates
# =========================
CONTENT_LIMIT = 512 # max tokens for view_file content
DEFAULT_MODEL = "qwen/qwen3-30b-a3b-instruct-2507"

class BoatseAgent:
    def __init__(self, api_key: str, 
                 dataset_path: str, repo_base_path: str, 
                 issue_index: str,
                 extracted_info: Dict[str, Any],
                 extracted_option: str = "all", top_k: int = 5, 
                 model_name: str = DEFAULT_MODEL, 
                 results_csv_path: str = "/content/results/", log_dir: str = "/content/logs/"):
        self.model_name = model_name
        self.api_key = api_key
        self.dataset_path = dataset_path
        self.repo_base_path = repo_base_path
        self.extracted_info = extracted_info
        self.extracted_option = extracted_option
        self.results_csv_path = results_csv_path
        self.top_k = top_k
        self.prompt = AgentContextPrompt()
        self.log_dir = log_dir
        self.issue_index = issue_index

    # =========================
    # Utils
    # =========================
    def _handle_tool_error_and_reprompt(
        self,
        messages,
        ctx: RunContext,
        logger: Optional[RunLogger],
        step_label: str,
        tool_called: str,
        tool_args: Dict[str, Any],
        error_text: str,
        function_calls: str,
        max_attempts: int = 2,
    ):
        """
        After a tool failure (e.g., bad path), tell the model the exact error and let it fix itself.
        Returns (tool_called, tool_args, tool_call_result).
        Stops immediately if the model returns ranked_files or a non-missing-path result.
        """
        for attempt in range(1, max_attempts + 1):
            # Show the tool error to the model
            messages.append({"role": "assistant", "content": to_assistant_content(error_text)})
            append_user_prompt(
                messages,
                PROMPT_TOOL_ERROR_FILE_PATH,
                tool_call=tool_called,
                args=tool_args,
                error=error_text,
                function_calls=function_calls,
            )
            if logger:
                logger.event("tool_error_reprompt", step=f"{step_label}_attempt_{attempt}",
                            tool=tool_called, args=tool_args, error=error_text)

            # Ask model what to do next
            try:
                parsed = self._model_json(messages, step=f"{step_label}_attempt_{attempt}", logger=logger, ctx=ctx)
            except CallTimeout as e:
                print("Call to model timed out")
                if logger:
                    logger.event("timeout", step=f"{step_label}_attempt_{attempt}", reason="model timeout")
                    logger.close()
                continue
            
            rf = extract_ranked_files_from_any(parsed)
            if rf:
                if logger:
                    logger.event("final_answer", step=step_label, ranked_files=rf)
                return "exit", {}, {"ranked_files": rf}
            
            if not isinstance(parsed, dict):
                continue

            # If model already exits with a final answer, capture it and stop
            if parsed.get("ranked_files"):
                if logger:
                    logger.event("final_answer", step=step_label, ranked_files=parsed["ranked_files"])
                return "exit", {}, {"ranked_files": parsed["ranked_files"]}

            # Otherwise execute the new tool call
            new_tool_called, new_tool_args, new_result = self._parse_tool_call(parsed)
            new_tool_called, new_tool_args, new_result = exec_tool(
                new_tool_called, new_tool_args, ctx, logger, step=f"{step_label}_tool", CONTENT_LIMIT=CONTENT_LIMIT
            )

            # Repo-aware path validation 
            missing, err_msg = is_missing_path(new_tool_called, new_tool_args, new_result, ctx.repo_dir)
            if not missing:
                return new_tool_called, new_tool_args, new_result

            # Still missing: loop with the new error
            tool_called, tool_args, error_text = new_tool_called, new_tool_args, err_msg

        # Give up; return last error so the outer loop can decide what to do
        return tool_called, tool_args, error_text

    def _format_invalids(self, invalids: List[Tuple[str, str]]) -> str:
        # Nice, compact bullet list for the model
        return "\n".join(f"- {p}  —  {reason}" for p, reason in invalids)

    # =========================
    # Model calling
    # =========================
    def _chat_completion_request(self, messages, step, logger: RunLogger, ctx : RunContext, timeout_sec: int = 600): # 10 minutes timeout 
        try:
            tok = tk.TokenizationUtils(ctx.model_name)
            token_count = tok.count_messages_tokens(messages)
            print(f"Token count: {token_count}")
            if logger:
                logger.event("token_count", step=step, token_count=token_count, context_size=tok._context_size)
            if token_count > tok._context_size:
                if logger:
                    logger.event("Context Overflow", step=step, token_count=token_count)

            model_kwargs = dict(
                model=ctx.model_name,
                temperature=0,
                max_tokens=4096,
                seed=42,
                api_key=self.api_key,
            )
            if logger:
                logger.event("prompt", step=step, messages=_fmt_messages(messages))

            response = run_with_timeout(timeout_s=timeout_sec, messages=messages, model_kwargs=model_kwargs)
            if logger:
                logger.event("response", step=step, response=getattr(response, "content", str(response)))
            return response
        except CallTimeout as e:
            print(f"Call to model timed out after {timeout_sec} seconds")
            if logger:
                logger.event("timeout", step=step, timeout_sec=timeout_sec, error=str(e))
            return e
        except Exception as e:
            print("Unable to generate chat completion response")
            print(f"Exception: {e}")
            if logger:
                logger.event("error", step=step, error=str(e))
            return e

    def _model_json(self, messages, step: str, logger: RunLogger, ctx : RunContext):
        """Send to model and parse JSON-ish output."""
        resp = self._chat_completion_request(messages, step=step, logger=logger, ctx=ctx)
        content = getattr(resp, "content", resp)
        return parse_json_safe(content, logger=logger, step=step)

    # =========================
    # Parse + Re-prompt
    # =========================
    def _parse_tool_call(self, response, step="unknown", logger=None):
        tool_name, tool_args, tool_call_results = "", {}, ""
        response_content = getattr(response, "content", response)
        response_json = parse_json_safe(response_content, logger=logger, step=step)

        if not isinstance(response_json, dict):
            if logger:
                logger.event("error", step=step, error="model did not return a JSON object",
                            raw=str(response_content))
            return tool_name, tool_args, None  # triggers your reprompt path

        tool_name = response_json.get("function_call")
        tool_args = response_json.get("args", {}) or {}
        return tool_name, tool_args, tool_call_results

    def _reprompt_model_invalid_json(self, messages, last_tool_call_result, step="reprompt", logger: Optional[RunLogger]=None, ctx : RunContext = None):
        max_iterations = 3
        error_messages = list(messages)  # shallow copy

        # Avoid re-adding the exact same tool result
        last = error_messages[-1] if error_messages else {}
        tool_blob = to_assistant_content(last_tool_call_result)
        if not (last.get("role") == "assistant" and last.get("content") == tool_blob):
            error_messages.append({"role": "assistant", "content": tool_blob})

        for attempt in range(1, max_iterations + 1):
            user_msg = (
                'The final response you provided is not valid JSON. '
                'The format is {"ranked_files": ["file1", "file2", ...]}. '
                'Provide only the JSON response.'
            )
            error_messages.append({"role": "user", "content": user_msg})

            if logger:
                logger.inc_reprompt(step=step, attempt=attempt, last_msg=user_msg)

            # Resend full convo and only the last error message, for now
            combined_error_and_context = messages.copy()
            combined_error_and_context.append(error_messages[-2])  # last assistant message
            combined_error_and_context.append(error_messages[-1])  # last user message
            try:
                parsed = self._model_json(combined_error_and_context, step=f"{step}_attempt_{attempt}", logger=logger, ctx=ctx)
            except CallTimeout as e:
                print("Call to model timed out")
                if logger:
                    logger.event("timeout", step=f"{step}_attempt_{attempt}", reason="model timeout")
                    logger.close()
                continue

            rf = extract_ranked_files_from_any(parsed)
            if rf:
                if logger:
                    logger.event("reprompt_success", step=step, attempt=attempt, ranked_files=rf)
                return {"ranked_files": rf}
            
            elif isinstance(parsed, dict) and "ranked_files" in parsed:
                if logger:
                    logger.event("reprompt_success", step=step, attempt=attempt, ranked_files=parsed.get("ranked_files"))
                print(f"Ranked files after re-prompt: {parsed.get('ranked_files')}")
                return parsed
            else:
                json = parse_json_safe(parsed, logger=logger, step=step)
                if isinstance(json, dict):
                    if logger:
                        logger.event("reprompt_success", step=step, attempt=attempt, parsed=json)
                    return json

        if logger:
            logger.event("reprompt_failed", step=step, attempts=max_iterations)
        return None

    def _reprompt_invalid_ranked_files(
        self,
        messages: List[Dict[str, str]],
        ctx: RunContext,
        logger: Optional[RunLogger],
        invalids: List[Tuple[str, str]],
        max_attempts: int = 2,
    ) -> Optional[List[str]]:
        """
        Tell the model which ranked files are invalid and ask for a corrected list.
        Returns corrected ranked_files or None if unable to correct.
        """
        invalids_pretty = self._format_invalids(invalids)
        if invalids_pretty == "":
            invalids_pretty = invalids = "Ÿou provided an empty list."
            return None  # nothing to fix

        for attempt in range(1, max_attempts + 1):
            # Show what failed
            messages.append({"role": "assistant", "content": to_assistant_content(
                {"ranked_files_validation": {"invalid": invalids}}
            )})
            # Ask for a corrected JSON
            append_user_prompt(
                messages,
                PROMPT_RANKED_FILES_INVALID,
                invalids_pretty=invalids_pretty, n=ctx.num_files
            )
            if logger:
                logger.event("ranked_files_reprompt", step=f"ranked_files_fix_attempt_{attempt}",
                            invalid=invalids)

            try:
                parsed = self._model_json(messages, step=f"ranked_files_fix_attempt_{attempt}", logger=logger, ctx=ctx)
            except CallTimeout:
                print("Call to model timed out")
                if logger:
                    logger.event("timeout", step=f"ranked_files_fix_attempt_{attempt}", reason="model timeout")
                    logger.close()
                    continue
            if not isinstance(parsed, dict):
                continue
            new_list = parsed.get("ranked_files")
            if not isinstance(new_list, list):
                continue

            # Validate again
            valid, invalid_again = validate_ranked_files(new_list, ctx.repo_dir)
            if logger:
                logger.event("ranked_files_validation", step=f"ranked_files_fix_attempt_{attempt}",
                            submitted=new_list, valid=valid, invalid=invalid_again)

            if not invalid_again and valid:
                return valid  # fully valid & non-empty
            # If still invalid, try again; otherwise loop ends and we return None

        return None

    # =========================
    # Pipeline
    # =========================
    def _run_turn(
        self,
        *,
        messages: List[Dict[str, str]],
        prompt_tmpl: str,
        prompt_kwargs: Dict[str, Any],
        step: str,
        ctx: RunContext,
        logger: RunLogger,
        content_limit: int,
        reprompt_step_suffix: str = "_reprompt",
    ) -> Tuple[str, Dict[str, Any], Any]:
        """Append prompt -> model -> (re)parse -> exec tool -> handle bad paths. Returns (tool_called, tool_args, tool_result)."""
        append_user_prompt(messages, prompt_tmpl, **prompt_kwargs)
        try:
            resp = self._model_json(messages, step=step, logger=logger, ctx=ctx)
        except Exception:
            # Treat any hard failure as a timeout-like condition
            logger.event("timeout", step=step, reason="model call failed")
            # Propagate a no-op result to let caller decide how to proceed
            return "", {}, {}

        if not isinstance(resp, dict):
            # one-shot reprompt to coerce JSON tool call
            rep = self._reprompt_model_invalid_json(messages, resp, logger=logger, step=f"{step}{reprompt_step_suffix}", ctx=ctx)
            if not isinstance(rep, dict):
                # caller can decide to abort/continue
                logger.event("abort", f"{step}{reprompt_step_suffix}", reason="still invalid after reprompts")
                return "", {}, {}

            resp = rep

        tool_called, tool_args, _ = self._parse_tool_call(resp, step=step, logger=logger)
        tool_called, tool_args, tool_result = exec_tool(
            tool_called, tool_args, ctx, logger, step, CONTENT_LIMIT=content_limit
        )

        # If the chosen tool hit a bad path, give the model a chance to fix itself.
        missing, err = is_missing_path(tool_called, tool_args, tool_result, ctx.repo_dir)
        if missing:
            tool_called, tool_args, tool_result = self._handle_tool_error_and_reprompt(
                messages=messages,
                ctx=ctx,
                logger=logger,
                step_label=f"{step}_fix",
                tool_called=tool_called,
                tool_args=tool_args,
                error_text=err,
                function_calls=FUNCTION_CALLS_TURN_3,  # safe to provide superset
                max_attempts=2,
            )

        return tool_called, tool_args, tool_result

    def _validate_ranked_files(self, parsed, ctx, logger, messages):
        # If we got (or can extract) ranked files, validate/correct and finish
        rf = extract_ranked_files_from_any(parsed)
        if rf:
            valid, invalids = validate_ranked_files(rf, ctx.repo_dir)
            if invalids:
                logger.event("ranked_files_validation", step="turn_3_validate",
                                proposed=rf, valid=valid, invalid=invalids)
                corrected = self._reprompt_invalid_ranked_files(messages, ctx, logger, invalids, max_attempts=3)
                if corrected:
                    logger.event("final_answer", "turn_3_corrected", ranked_files=corrected)
                    return {"ranked_files": corrected}
                logger.event("final_answer", "turn_3_partial", ranked_files=valid, note="partial due to invalid paths")
                return {"ranked_files": valid}
            logger.event("final_answer", "turn_3", ranked_files=rf)
            return {"ranked_files": rf}
        return {"ranked_files": []}


    def _turn3_loop(
        self,
        *,
        messages: List[Dict[str, str]],
        tool_called: str,
        tool_args: Dict[str, Any],
        tool_result: Any,
        ctx: RunContext,
        logger: RunLogger,
        content_limit: int
    ) -> Dict[str, List[str]]:
        """Run until we get ranked_files or exit. Returns final {'ranked_files': [...]} (possibly empty)."""
        first_prompt = True
        invalid_reprompts_left = 3

        while tool_called != "exit":
            # Feed last tool result to the chat
            messages.append({"role": "assistant", "content": to_assistant_content(tool_result)})

            # First time we also give the turn-3 instruction payload
            if first_prompt:
                append_user_prompt(
                    messages, PROMPT_TURN_3,
                    tool_call=tool_called, args=tool_args,
                    function_calls=FUNCTION_CALLS_TURN_3, n=ctx.num_files
                )
                first_prompt = False
            else:
                append_user_prompt(
                    messages,
                    SELF_EVAL_POST_VIEW_PROMPT, n=ctx.num_files
                )

            # Ask again
            try:
                model_out = self._model_json(messages, step="turn_3", logger=logger, ctx=ctx)
            except Exception:
                logger.event("timeout", step="turn_3", reason="model call failed")
                return {"ranked_files": []}

            parsed = model_out if isinstance(model_out, dict) else parse_json_safe(model_out)
            logger.event("parse", "turn_3", parsed=bool(parsed))

            # If we got (or can extract) ranked files, validate/correct and finish
            rf = extract_ranked_files_from_any(parsed)
            if rf:
                valid, invalids = validate_ranked_files(rf, ctx.repo_dir)
                if invalids:
                    logger.event("ranked_files_validation", step="turn_3_validate",
                                proposed=rf, valid=valid, invalid=invalids)
                    corrected = self._reprompt_invalid_ranked_files(messages, ctx, logger, invalids, max_attempts=3)
                    if corrected:
                        logger.event("final_answer", "turn_3_corrected", ranked_files=corrected)
                        return {"ranked_files": corrected}
                    logger.event("final_answer", "turn_3_partial", ranked_files=valid, note="partial due to invalid paths")
                    return {"ranked_files": valid}
                logger.event("final_answer", "turn_3", ranked_files=rf)
                return {"ranked_files": rf}

            # Otherwise, expect another tool call
            if isinstance(parsed, dict):
                tool_called, tool_args, tool_result = self._parse_tool_call(parsed, step="turn_3_tool", logger=logger)
                tool_called, tool_args, tool_result = exec_tool(
                    tool_called, tool_args, ctx, logger, "turn_3_tool", CONTENT_LIMIT=content_limit
                )

                if tool_called == "max_views_reached":
                    # Prevent infinite loops on max_views_reached
                    logger.event("abort file view", "turn_3", reason="max_views_reached called")
                    # ask for final ranked files
                    append_user_prompt(
                        messages, PROMPT_MAX_FILE_VIEWS,
                        n=ctx.num_files
                    )

                    try:
                        model_out = self._model_json(messages, step="turn_3", logger=logger, ctx=ctx)
                    except Exception:
                        logger.event("timeout", step="turn_3", reason="model call failed")
                        return {"ranked_files": []}
                    
                    parsed = model_out if isinstance(model_out, dict) else parse_json_safe(model_out)
                    logger.event("parse", "turn_3", parsed=bool(parsed))

                    candidate = self._validate_ranked_files(parsed, ctx, logger, messages)
                    if candidate and candidate.get("ranked_files"):
                        return candidate
                    parsed = model_out if isinstance(model_out, dict) else parse_json_safe(model_out)
                    if not isinstance(parsed, dict):
                        # fallthrough to reprompt
                        model_out = parsed

                missing, err = is_missing_path(tool_called, tool_args, tool_result, ctx.repo_dir)
                if missing:
                    tool_called, tool_args, tool_result = self._handle_tool_error_and_reprompt(
                        messages=messages,
                        ctx=ctx,
                        logger=logger,
                        step_label="turn_3_fix",
                        tool_called=tool_called,
                        tool_args=tool_args,
                        error_text=err,
                        function_calls=FUNCTION_CALLS_TURN_3,
                        max_attempts=2,
                    )
                    # if we got a final answer during error handling, return it
                    if tool_called == "exit":
                        candidate = extract_ranked_files_from_any(tool_result)
                        if candidate:
                            return {"ranked_files": candidate}
                continue

            # If we’re here, parsed wasn’t dict and had no ranked files -> last-ditch reprompt
            invalid_reprompts_left -= 1
            if invalid_reprompts_left <= 0:
                logger.event("abort", "turn_3_reprompt", reason="too many invalid re-prompts")
                return {"ranked_files": []}

            rep = self._reprompt_model_invalid_json(messages, tool_result, logger=logger, step="turn_3_reprompt", ctx=ctx)
            if rep and rep.get("ranked_files"):
                logger.event("final_answer", "turn_3_reprompt", ranked_files=rep["ranked_files"])
                return rep

        # If the loop ended with "exit" but no ranked files, return empty
        return {"ranked_files": []}


    def _self_evaluate_and_adopt(
        self,
        ranked_files: List[str],
        ctx: RunContext,
        logger: RunLogger,
    ) -> List[str]:
        """
        Ask the model to re-check its answer; allow it to request view_file a few times,
        and adopt a revised ranked_files only if it's valid & non-empty.
        """

        # ensure optional fields exist
        if not hasattr(ctx, "viewed_files") or ctx.viewed_files is None:
            ctx.viewed_files = []
        if not hasattr(ctx, "bm25_results"):
            ctx.bm25_results = None

        orig_rf = [f for f in ranked_files if isinstance(f, str) and f.strip()]

        # Build a fresh, small context for self-eval (reduces token bloat)
        new_messages: List[Dict[str, str]] = []
        new_messages.append({"role": "system", "content": ANSWER_REVIEW_SYSTEM_PROMPT})
        append_user_prompt(
            new_messages,
            ANSWER_REVIEW_PROMPT_W_VIEW,
            issue_description=ctx.issue_description,
            bm25_retrieved_pretty=json.dumps(ctx.bm25_results, ensure_ascii=False, indent=2) if ctx.bm25_results else "None",
            viewed_files_pretty=json.dumps(ctx.viewed_files, ensure_ascii=False) if ctx.viewed_files else "None",
            answer=json.dumps(ranked_files, ensure_ascii=False), n=ctx.num_files
        )

        # ---- helpers ----
        def _dedupe_keep_order(seq: List[str]) -> List[str]:
            seen, out = set(), []
            for s in seq:
                if isinstance(s, str) and s not in seen:
                    seen.add(s); out.append(s)
            return out

        def _parse_self_eval(resp: Any) -> Tuple[str, Any]:
            """
            Returns ('rank', List[str]) or ('view', str) or ('', None)
            """
            # 1) direct ranked-files extraction (robust)
            rf = extract_ranked_files_from_any(resp)
            if rf:
                return ("rank", rf)

            # 2) raw dict forms we might accept
            if isinstance(resp, dict):
                # {"ranked_files": [...]}
                if isinstance(resp.get("ranked_files"), list):
                    return ("rank", resp["ranked_files"])

                # {"view_file": "path"} (simple intent form)
                if isinstance(resp.get("view_file"), str):
                    return ("view", resp["view_file"])

                # {"function_call": "view_file", "args": {"file_path": "..."}}
                if resp.get("function_call") == "view_file":
                    args = resp.get("args", {}) or {}
                    p = args.get("file_path") or args.get("path") or args.get("view_file")
                    if isinstance(p, str) and p.strip():
                        return ("view", p)

            # 3) nothing usable
            return ("", None)

        # ---- main self-eval loop ----
        MAX_VIEWS = 5  # prevent infinite loops
        views_used = 0

        while True:
            try:
                eval_resp = self._model_json(new_messages, step="self_eval", logger=logger, ctx=ctx)
            except Exception:
                logger.event("timeout", step="self_eval", reason="model call failed")
                return ranked_files

            intent, payload = _parse_self_eval(eval_resp)

            if intent == "rank":
                new_rf = _dedupe_keep_order(list(payload))[:ctx.num_files]
                valid, invalids = validate_ranked_files(new_rf, ctx.repo_dir)
                # keep valid subset if any
                if valid and not invalids:
                    logger.event("self_eval_applied", step="self_eval", ranked_files=valid)
                    return valid
                # keep valid subset if any
                if valid:
                    logger.event("self_eval_partial", step="self_eval", adopted=valid, dropped=invalids)
                    return valid
                # otherwise revert
                logger.event("self_eval_skipped", step="self_eval", reason="invalid_or_empty_eval")
                return orig_rf            

            if intent == "view":
                path = str(payload).strip()
                if not path:
                    logger.event("self_eval_skipped", step="self_eval", reason="empty_view_file_request")
                    return ranked_files
                if path in ctx.viewed_files:
                    # Already provided; avoid loops by asking the model to proceed
                    new_messages.append({"role": "assistant", "content": json.dumps(
                        {"note": f"File '{path}' was already provided. Please proceed with ranked_files."}
                    )})
                    append_user_prompt(new_messages, SELF_EVAL_POST_VIEW_PROMPT, n=ctx.num_files)
                    continue

                if views_used >= MAX_VIEWS:
                    logger.event("self_eval_skipped", step="self_eval", reason="view_limit_reached")
                    return ranked_files

                # Execute view_file tool and feed back
                logger.event("self_eval_view_file", step="self_eval", view_file=path)
                _, _, tool_result = exec_tool(
                    "view_file", {"file_path": path}, ctx, logger, "self_eval_tool", CONTENT_LIMIT=CONTENT_LIMIT
                )
                ctx.viewed_files.append(path)
                new_messages.append({"role": "assistant", "content": to_assistant_content(tool_result)})
                append_user_prompt(new_messages, SELF_EVAL_POST_VIEW_PROMPT, n=ctx.num_files)
                views_used += 1
                continue

            # No usable intent; keep original
            logger.event("self_eval_skipped", step="self_eval", reason="no_ranked_files_in_eval")
            return ranked_files
        
    def _get_issue(self, issue_index: str) -> Dict[str, Any]:
        df = pd.read_csv(self.dataset_path)
        if 'instance_id' in df.columns:
            issue = df[df['instance_id'] == issue_index].iloc[0].to_dict()
        elif 'id' in df.columns:
            issue = df[df['id'] == issue_index].iloc[0].to_dict()
        else:
            issue = df.iloc[int(issue_index)].to_dict()
        return issue

    # =========================
    # Entry point: localization pipeline
    # =========================
    def localize(self):
        dp = self._get_issue(self.issue_index)
        dataset_name = os.path.splitext(os.path.basename(self.dataset_path))[0]

        results_csv_path = f"{self.results_csv_path}/{dataset_name}/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_results_{dataset_name}_{self.model_name}_{self.extracted_option}_top{self.top_k}.csv"
        os.makedirs(os.path.dirname(results_csv_path), exist_ok=True)

        log_dir = f"{self.log_dir}/{dataset_name}/" + self.model_name + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        tokenizer = tk.TokenizationUtils(self.model_name)

        if dataset_name == "lca":
            run_id = f"{dp['repo_owner']}__{dp['repo_name']}__{dp['base_sha']}_{self.issue_index}"
            repo_name = f"{dp['repo_owner']}/{dp['repo_name']}"
            issue_description = f"{dp['issue_title']} {dp['issue_body']}"
            repo_dir = os.path.join(self.repo_base_path, f"{dp['repo_owner']}__{dp['repo_name']}")
        else:
            repo_name = dp['repo']
            repo_name = repo_name.replace("/", "__")
            repo_name = f"{repo_name}__{dp['base_commit']}"
            run_id = f"{repo_name}_{self.issue_index}"
            issue_description = dp['problem_statement']
            repo_dir = os.path.join(self.repo_base_path, f"{repo_name}")

        logger = RunLogger(log_dir=log_dir, run_id=run_id, model_name=self.model_name)
        try:
            print(f"\n\n=== Processing issue {self.issue_index} in {repo_name} ===")

            ctx = RunContext(
                repo_name=repo_name, repo_dir=repo_dir, dataset=dataset_name, issue_index=self.issue_index,
                    issue_description=issue_description, model_name=self.model_name, tk=tokenizer, extracted_option=self.extracted_option,
                    viewed_files=[], viewed_files_full={}, bm25_results=[], extracted_info=self.extracted_info, num_files=self.top_k
                )

            logger.event("start", "pipeline",
                        dataset=dataset_name,
                        repo_name=repo_name, repo_base_path=self.repo_base_path)

            messages = [{"role": "system", "content": SYSTEM_PROMPT.format(n=ctx.num_files)}]

            # ---- Turn 1
            t1_call, t1_args, t1_res = self._run_turn(
                messages=messages,
                prompt_tmpl=PROMPT_TURN_1,
                prompt_kwargs=dict(issue_description=ctx.issue_description,
                                function_calls=FUNCTION_CALLS_TURN_1,
                                repo_name=ctx.repo_name, n=ctx.num_files),
                step="turn_1",
                ctx=ctx, logger=logger, content_limit=CONTENT_LIMIT,
            )

            # ---- Turn 2
            messages.append({"role": "assistant", "content": to_assistant_content(t1_res)})
            t2_call, t2_args, t2_res = self._run_turn(
                messages=messages,
                prompt_tmpl=PROMPT_TURN_2,
                prompt_kwargs=dict(tool_call=t1_call, args=t1_args,
                                function_calls=FUNCTION_CALLS_TURN_2, n=ctx.num_files),
                step="turn_2",
                ctx=ctx, logger=logger, content_limit=CONTENT_LIMIT,
            )

            # ---- Turn 3 (loop)
            final_obj = self._turn3_loop(
                messages=messages,
                tool_called=t2_call, tool_args=t2_args, tool_result=t2_res,
                ctx=ctx, logger=logger, content_limit=CONTENT_LIMIT,
            )

            ranked_files = final_obj.get("ranked_files", [])
            self_reviewed_files = self._self_evaluate_and_adopt(
                ranked_files=ranked_files, ctx=ctx, logger=logger
                )
            if len(self_reviewed_files) > 0:
                ranked_files = self_reviewed_files

            print(f"Ranked files: {ranked_files}")
            logger.event("end", "pipeline", ranked_files=ranked_files, reprompts=logger.summary["reprompts"])

            summary_row = {
                "id": dp["id"] if dataset_name == "lca" else dp['instance_id'],
                "run_id": logger.run_id,
                "repo": ctx.repo_name,
                "base_sha": dp["base_sha"] if dataset_name == "lca" else dp['base_commit'],
                "changed_files": dp["changed_files"] if dataset_name == "lca" else parse_changed_files_from_diff(dp['patch']),
                "model": ctx.model_name,
                "reprompts": logger.summary["reprompts"],
                "final_files": ranked_files,
                "started_at": logger.summary["started_at"],
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
            (pd.DataFrame([summary_row])
                .to_csv(results_csv_path, mode="a", index=False, header=not os.path.exists(results_csv_path)))

        except Exception as e:
            logger.event("error", "pipeline", error=str(e))
            logger.close()
            raise
        finally:
            logger.close()