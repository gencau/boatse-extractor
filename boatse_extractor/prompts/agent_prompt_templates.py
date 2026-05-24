AGENT_SYSTEM_PROMPT_TEMPLATE = """
You are an expert software engineer, specialized in file localization from bug reports.
"""

AGENT_EXTRACT_INFO_PROMPT = """
You are given a bug description. extract the relevant information from the bug description to localize the bug in a repository. your response must be a json object
with the fields: explanation, path, filename, identifiers, code snippet, stacktrace, error_message. If one of the fields is not provided in the bug description, return an empty string for that field.
Include in the explanation the essential information required to localize the bug, concisely.

<bug_description>
{}
</bug_description>

Stricly follow this template:
{{"explanation": "concise description of the bug", "path": "file path as mentioned in the bug", "filename": "name of file", "identifiers": ["variable_name", "function_name"], "code snippet": "if (0) print("hello world")", "stacktrace": "any stack trace", "error_message": "error messages only"}}

Only return the JSON object.
"""

AGENT_FIND_BEST_MATCHES_PROMPT = """
You are given a bug description along with summarized information in JSON format and a list of file paths that may be relevant to the bug. 
Your task is to identify the files that are most likely to contain the bug based on the provided information.
Your response must be a JSON object with a single field "files", which is a list of file paths, relative to the repository root, from the provided list that you believe are most relevant to the bug description.

<bug_description>
{}
</bug_description>
<extracted_information>
{}
</extracted_information>
<file_list>
{}
</file_list>
"""