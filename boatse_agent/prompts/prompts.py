SYSTEM_PROMPT = """
You are an expert software engineer specialized in bug localization. 
Your task is to identify the files most likely to contain bugs based on bug descriptions. 
You have access to several tools to help you in this task. 
From the files you have seen, you will select {n} files that you believe are most likely to contain the bug.
"""

PROMPT_TURN_1 = """
<bug_description>
{issue_description}
</bug_description>

The repository name is:
<repo_name>
{repo_name}
</repo_name>

You have access to the following tools:
{function_calls}
You must use these tools to find {n} files that are most likely to contain the bug. Start by analyzing the issue description and then decide which tool to use first.
If you are done, provide your final ranked list of {n} files most likely to contain bugs in a JSON array as follows: {{"ranked_files": ["file1", "file2", ...]}}
Otherwise, return only the tool call as JSON. Example: {{"function_call": "func_name", "args": {{"arg_name": 10}} }}
Reply ONLY with the JSON response.

"""

FUNCTION_CALLS_TURN_1 = """
* extract_relevant() -> Retrieve the most relevant info from the bug. Returns a JSON object with explanation, relevant code snippets, error messages and file paths mentioned in the bug. *\n\
"""


PROMPT_TURN_2 = """
Based on the tool call results, please select your next action. Recall that you need to provide the {n} files most likely to contain bugs.
<tool_called>
{tool_call}
</tool_called>
<args>
{args}
</args>


You have access to the following tools: {function_calls}
If you are done, provide your final ranked list of {n} files most likely to contain bugs in a JSON array as follows: {{"ranked_files": ["file1", "file2", ...]}}
Otherwise, return only the tool call as JSON. Example: {{"function_call": "func_name", "args": {{"arg_name": 10}} }}
Reply ONLY with the JSON response.

"""

FUNCTION_CALLS_TURN_2 = """
* get_bm25_top10() -> Retrieve the top-10 best results with BM25 search. *\n\
* get_bm25_top20() -> Retrieve the top-20 best results with BM25 search.*\n\
* get_bm25_top30() -> Retrieve the top-30 best results with BM25 search. *\n\
"""


PROMPT_TURN_3 = """
Based on the tool call results, please select your next action. Recall that you need to provide the {n} files most likely to contain bugs.

<tool_called>
{tool_call}
</tool_called>
<args>
{args}
</args>


You have access to the following tools: {function_calls}
If you are done, provide your final ranked list of {n} files most likely to contain bugs in a JSON array as follows: {{"ranked_files": ["file1", "file2", ...]}}
Otherwise, return only the tool call as JSON. Example: {{"function_call": "func_name", "args": {{"arg_name": 10}} }}

Reply ONLY with the JSON response. Remember, do not try to complete the code.
"""

FUNCTION_CALLS_TURN_3 = """
* view_file(`file_path`) -> View the content of the specified file. *\n\
* view_readme() -> View the content of the README file in the repository. *\n\
"""

PROMPT_TOOL_ERROR_FILE_PATH = """
The last tool call failed.

<tool_called>
{tool_call}
</tool_called>
<args>
{args}
</args>
<error>
{error}
</error>

Call the tool view_file again with a valid path.
Reminder of tool documentation:
* view_file(`file_path`) -> View the content of the specified file. *\n\

Reply ONLY with the JSON response.
"""

PROMPT_RANKED_FILES_INVALID = """
Some of the file paths in your last ranked_files are invalid for this repository.

<invalid_paths>
{invalids_pretty}
</invalid_paths>

Please return ONLY a corrected JSON object containing {n} files ranked from most to least likely to contain bugs, in the format:
{{"ranked_files": ["path1", "path2", ...]}}
"""

PROMPT_MAX_FILE_VIEWS = """
You have reached the maximum number of allowed file views.

Use the information retrieved previously to construct your final ranked list of {n} files most likely to contain bugs in a JSON array as follows: {{"ranked_files": ["file1", "file2", ...]}}
Reply ONLY with the JSON response.
"""

ANSWER_REVIEW_SYSTEM_PROMPT = """
You are an expert software engineer specialized in bug localization. 
Your task is to review a ranked list of files most likely to contain bugs based on a bug description. 
You have access to the evidence seen during the investigation, such as file views and BM25 retrievals.
Your goal is to review and/or adjust the ranked list of files based on your confidence that each file contains the bug.
"""

ANSWER_REVIEW_PROMPT_W_VIEW = """
This is the bug description:
<bug_description>
{issue_description}
</bug_description>

BM25 retrieved the following files:
<bm25_retrieved>
{bm25_retrieved_pretty}
</bm25_retrieved>

You have also viewed the following files during your investigation:
<viewed_files>
{viewed_files_pretty}
</viewed_files>

\n This is the final ranked list of files you provided: {answer}.

You are required to return {n} files most likely to contain bugs.

Provide a confidence score for each file in the list based on this rubric:
<RUBRIC>
9-10 (Near-certain): Direct file mention in bug description or stack frame, you have seen the file content and it matches the bug context.
7-8 (Strong): Clear functional ownership and multiple strong signals (identifiers + behavior + BM25 retrieval).
5-6 (Plausible): Good topical/structural match with partial identifier/behavior evidence.
3-4 (Weak): Mostly retrieval/path proximity with little concrete alignment, no evidence.
1-2 (Very unlikely): Invalid/irrelevant, or only superficial similarity, no evidence.
</RUBRIC>

Adjust the ranking based on your confidence scores.
If your confidence score is below 7 for the first 2-3 files, you should consider replacing it with a file for which you have higher confidence.

You may reply with ONE of these JSON shapes ONLY:
A) {{"ranked_files": ["file1", "file2", ...], "confidence_scores": ["score1", "score2", ...]}}
B) {{"function_call":"view_file","args":{{"file_path":string}}}}
Never include code blocks, file contents, or extra keys.
If you are about to paste file content, STOP and return JSON instead.
Reply ONLY with the required JSON format.

"""

SELF_EVAL_POST_VIEW_PROMPT = """
You are analyzing code. The code may be truncated. 
DO NOT attempt to complete incomplete functions or classes. 
If you see incomplete code, work with what's available.

Select your next action. Remember you need to provide the {n} files most likely to contain bugs.
You may reply with ONE of these JSON shapes ONLY:
A) {{"ranked_files": ["file1", "file2", ...], "confidence_scores": ["score1", "score2", ...]}}
B) {{"function_call":"view_file","args":{{"file_path":string}}}}
Never include code blocks, file contents, or extra keys.

Reply ONLY with the required JSON format.

"""
