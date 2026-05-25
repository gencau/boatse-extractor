# boatse_extractor/extractor.py

from __future__ import annotations
from typing import Any, Dict, Optional
from openai import OpenAI

from boatse_extractor.prompts.agent_context_prompt import AgentContextPrompt
from boatse_extractor.utils.json_utils import parse_json_safe


class BugInfoExtractor:
    """
    Clean interface for extracting structured info from a bug description.

    Usage:
        extractor = BugInfoExtractor()
        result = extractor("App crashes when uploading files > 10MB")
    """

    DEFAULT_MODEL = "qwen/qwen3-30b-a3b-instruct-2507"

    def __init__(
        self,
        prompt: AgentContextPrompt,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL,
        
    ):
        self._prompt = prompt
        self._model_name = model_name

        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            temperature=0.0,
            seed=42,
            max_tokens=2048,
        )


    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def __call__(self, bug_description: str, **kwargs) -> Dict[str, Any]:
        """Extract structured info from a bug description."""
        return self.extract(bug_description, **kwargs)

    def extract(self, bug_description: str, **kwargs) -> Dict[str, Any]:
        """
        Parameters
        ----------
        bug_description : str
            Free-text description of the bug or issue.

        Returns
        -------
        dict with key ``summarized_info`` containing the structured extraction.
        """
        messages = self._build_messages(bug_description)
        raw = self._generate(messages)
        summarized_info = self._parse(raw)
        return {"summarized_info": summarized_info}

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _build_messages(self, bug_description: str) -> list:
        return [
            {"role": "system",  "content": self._prompt.get_system_prompt()},
            {"role": "user",    "content": self._prompt.get_base_extract_prompt(
                issue_description=bug_description
            )},
        ]

    def _generate(self, messages: list) -> str:
        response = self._client.chat.completions.create(
            model=self._model_name, 
            messages=messages,
            max_tokens=2048,
            temperature=0.0,
        )
        return response.choices[0].message.content

    def _parse(self, raw: str) -> Any:
        try:
            return parse_json_safe(raw)
        except Exception:
            return {}