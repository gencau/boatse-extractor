# boatse-extractor/extractor.py

from __future__ import annotations
from typing import Any, Dict, Optional
from transformers import pipeline, AutoTokenizer
import torch

from extractor.prompts.agent_context_prompt import AgentContextPrompt
from extractor.utils.json_utils import parse_json_safe


class BugInfoExtractor:
    """
    Clean interface for extracting structured info from a bug description.

    Usage:
        extractor = BugInfoExtractor()
        result = extractor("App crashes when uploading files > 10MB")
    """

    DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # swap for any ≤7B HF model

    def __init__(
        self,
        prompt: AgentContextPrompt,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
    ):
        self._prompt = prompt
        self._model_name = model_name

        # auto-detect GPU in Colab, fall back to CPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

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
        # apply_chat_template handles model-specific formatting (Llama, Qwen, etc.)
        prompt_str = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        output = self._pipe(
            prompt_str,
            max_new_tokens=2048,
            temperature=1e-6,   # near-deterministic; some models reject 0.0
            do_sample=False,
        )
        return output[0]["generated_text"][len(prompt_str):]

    def _parse(self, raw: str) -> Any:
        try:
            return parse_json_safe(raw)
        except Exception:
            return {}