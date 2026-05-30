from typing import List

from src.baselines.utils.type_utils import ChatMessage
from src.baselines.backbones.agent.prompts.agent_prompt_templates import AGENT_EXTRACT_INFO_PROMPT, AGENT_SYSTEM_PROMPT_TEMPLATE, \
                                                                         AGENT_FIND_BEST_MATCHES_PROMPT


class AgentContextPrompt():
    def get_system_prompt(self) -> str:
        return AGENT_SYSTEM_PROMPT_TEMPLATE

    def get_base_extract_prompt(self, issue_description: str) -> str:
        return AGENT_EXTRACT_INFO_PROMPT.format(issue_description)
    
    def get_base_find_matches_prompt(self) -> str:
        return AGENT_FIND_BEST_MATCHES_PROMPT
    
    def get_full_prompt(self, issue_description: str, summarized_info: str, file_list: str) -> str:
        return AGENT_FIND_BEST_MATCHES_PROMPT.format(issue_description, summarized_info, file_list)

    def chat(self, system_prompt: str, prompt: str) -> List[ChatMessage]:
        return [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            },
        ]
