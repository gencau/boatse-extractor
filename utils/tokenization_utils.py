from typing import List

from pathlib import Path
import anthropic
import tiktoken
from transformers import AutoTokenizer
from sentencepiece import SentencePieceProcessor


class TokenizationUtils:
    """A wrapper for two tokenization-related operations:
    - estimating the number of tokens for a prompt
    - truncating a prompt to first X tokens.
    """

    PROFILE_NAME_TO_PROVIDER_AND_MODEL = {
        "deepseek-ai/deepseek-coder-1.3b-instruct": {"model_provider": "huggingface",
                                                     "model_name": "deepseek-ai/deepseek-coder-1.3b-instruct",
                                                     "context_size": 16384},
        "chat-llama-v2-7b": {"model_provider": "huggingface", "model_name": "codellama/CodeLlama-7b-Instruct-hf",
                             "context_size": 16000},
        "anthropic-claude": {"model_provider": "anthropic", "model_name": "claude", "context_size": 16000},

        "gpt-3.5-turbo-0613": {"model_provider": "openai", "model_name": "gpt-3.5-turbo", "context_size": 4096},
        "gpt-3.5-turbo-1106": {"model_provider": "openai", "model_name": "gpt-3.5-turbo", "context_size": 16385},
        "gpt-4-0613": {"model_provider": "openai", "model_name": "gpt-3.5-turbo", "context_size": 8192},
        "gpt-4-1106-preview": {"model_provider": "openai", "model_name": "gpt-4", "context_size": 128000},
        "llama32-3b-8k": {"model_provider": "meta", "model_name": "llama3.2", "context_size": 128000},
        "llama3.1-128k": {"model_provider": "meta", "model_name": "llama3.1", "context_size": 128000},
        "deepseek-r1-7b-8k": {"model_provider": "meta", "model_name": "deepseek-r1", "context_size": 8192},
        "deepseek-r1-7b-16k": {"model_provider": "meta", "model_name": "deepseek-r1", "context_size": 16384},
        "deepseek-r1-7b-32k": {"model_provider": "meta", "model_name": "deepseek-r1", "context_size": 32768},
        "deepseek-r1-14b-8k": {"model_provider": "meta", "model_name": "deepseek-r1", "context_size": 8192},
        "deepseek-r1-14b-16k": {"model_provider": "meta", "model_name": "deepseek-r1", "context_size": 16384},
        "deepseek-r1-14b-32k": {"model_provider": "meta", "model_name": "deepseek-r1", "context_size": 32768},
        "qwen2.5-coder-14b-8k": {"model_provider": "meta", "model_name": "qwen2.5-coder", "context_size": 8192},
        "qwen2.5-coder-14b-16k": {"model_provider": "meta", "model_name": "qwen2.5-coder", "context_size": 16384},
        "qwen3-8b-16k": {"model_provider": "meta", "model_name": "qwen2.5-coder", "context_size": 16384},
        "qwen3-14b-16k": {"model_provider": "meta", "model_name": "qwen3-coder", "context_size": 16384},
        "qwen2.5-coder-7b-8k": {"model_provider": "meta", "model_name": "qwen2.5-coder", "context_size": 16384},
        "qwen2.5-coder-7b-16k": {"model_provider": "meta", "model_name": "qwen2.5-coder", "context_size": 16384},
        "qwen3-32b-16k": {"model_provider": "meta", "model_name": "qwen3", "context_size": 16384},
        "qwen2.5-coder-7b-32k": {"model_provider": "meta", "model_name": "qwen2.5-coder", "context_size": 32768},
        "qwen2.5-coder-32b-8k": {"model_provider": "meta", "model_name": "qwen2.5-coder", "context_size": 8192},
        "qwen2.5-coder-32b-16k": {"model_provider": "huggingface", "model_name": "Qwen/Qwen2.5-Coder-32B-Instruct", "context_size": 16384},
        "qwen2.5-coder-32b-32k": {"model_provider": "meta", "model_name": "qwen2.5-coder", "context_size": 32768},
        "codellama-7b-8k": {"model_provider": "meta", "model_name": "codellama", "context_size": 8192},
        "codellama-7b-16k": {"model_provider": "meta", "model_name": "codellama", "context_size": 16384},
        "codellama-13b-8k": {"model_provider": "meta", "model_name": "codellama", "context_size": 8192},
        "codellama-13b-16k": {"model_provider": "meta", "model_name": "codellama", "context_size": 16384},
        "llama3.3-70b-16k": {"model_provider": "huggingface", "model_name": "meta-llama/Llama-3.3-70B-Instruct", "context_size": 16384},
        "qwen3-coder-16k": {"model_provider": "huggingface", "model_name": "Qwen/Qwen3-Coder-30B-A3B-Instruct", "context_size": 16384},
        "deepcoder-14b-16k": {"model_provider": "huggingface", "model_name": "agentica-org/DeepCoder-14B-Preview", "context_size": 16384},        
    }

    def __init__(self, profile_name: str):
        model_info = self.PROFILE_NAME_TO_PROVIDER_AND_MODEL.get(profile_name, None)
        if not model_info:
            raise ValueError(f"Unknown profile {profile_name}.")

        self._model_provider = model_info["model_provider"]
        self._model_name = model_info["model_name"]
        self._context_size = model_info["context_size"]
        self._tokenizer = None

        if (self._model_name == "codellama"):
            self._tokenizer = SentencePieceProcessor()
            current_dir = Path(__file__).parent
            self._tokenizer.Load(str(current_dir) + "/tokenizers/codellama/tokenizer.model")

        if self._tokenizer == None:
            if self._model_provider == "openai":
                self._tokenizer = tiktoken.encoding_for_model(self._model_name)
            elif self._model_provider == "anthropic":
                self._tokenizer = anthropic.Anthropic().get_tokenizer()
            elif self._model_provider == "huggingface":
                self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            elif self._model_provider == "meta":
                self._tokenizer = tiktoken.encoding_for_model("gpt2")

    def _encode(self, text: str) -> List[str]:
        """Estimates the number of tokens for a given string."""
        if self._model_provider == "openai" or self._model_provider == "meta":
            return self._tokenizer.encode(text)
        if self._model_provider == "anthropic":
            return self._tokenizer.encode(text)
        if self._model_provider == "huggingface":
            return self._tokenizer(text).input_ids

        raise ValueError(f"{self._model_provider} is currently not supported for token estimation.")

    def _decode(self, text: str) -> List[str]:
        """Estimates the number of tokens for a given string."""
        if self._model_provider == "openai" or self._model_provider == "meta":
            return self._tokenizer.decode(text)
        if self._model_provider == "anthropic":
            return self._tokenizer.decode(text)

        raise ValueError(f"{self._model_provider} is currently not supported for token estimation.")

    def count_text_tokens(self, text: str) -> int:
        """Estimates the number of tokens for a given string."""
        return len(self._encode(text))

    def count_messages_tokens(self, messages: list[dict[str, str]]) -> int:
        """Estimates the number of tokens for a given list of messages.

        Note: Currently, for some agents (e.g., OpenAI) the returned number might be slightly lower than the actual number of tokens, because the
        special tokens are not considered.
        """
        return sum([self.count_text_tokens(value) for message in messages for key, value in message.items()])

    def truncate(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Truncates a given list of messages to first `max_num_tokens` tokens.

        Note: A current version only truncates a last message, which might not be suitable for all use-cases.
        """
        num_tokens_except_last = self.count_messages_tokens(messages[:-1])
        messages[-1]["content"] = self._truncate(
            messages[-1]["content"], max_num_tokens=self._context_size - num_tokens_except_last
        )
        return messages

    def messages_match_context_size(self, messages: list[dict[str, str]]) -> bool:
        return self.count_messages_tokens(messages) <= self._context_size

    def text_match_context_size(self, text: str) -> bool:
        return self.text_match_context_size(text) <= self._context_size

    def _truncate(self, text: str, max_tokens: int) -> str:
        # Always produce a flat list[int]
        enc = self._tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        ids = enc.get("input_ids", None)
        if ids is None:
            # Fallback: encode returns List[int]
            ids = self._tokenizer.encode(text, add_special_tokens=False)

        # Normalize to List[int]
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            # batched -> take first batch
            ids = ids[0]

        # Guard rails
        if not isinstance(ids, list) or (ids and not isinstance(ids[0], int)):
            raise TypeError(f"Unexpected token ids shape/type: {type(ids)} / {type(ids[0]) if ids else None}")

        # Handle empty inputs safely
        if not ids:
            return ""

        clipped_ids = ids[:max_tokens]
        return self._tokenizer.decode(clipped_ids, skip_special_tokens=True)


if __name__ == '__main__':
    print(TokenizationUtils("gpt-4-0613").count_text_tokens("sfef efwe ef"))
