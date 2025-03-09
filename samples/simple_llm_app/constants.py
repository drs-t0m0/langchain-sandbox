from typing import Dict


class Constant:
    def __init__(self, **kwargs):
        self._AZURE_OPENAI_CONFIG = {
            "gpt-4o-mini": {
                "AZURE_OPENAI_DEPLOYMENT_NAME": "gpt-4o-mini",
                "AZURE_OPENAI_API_VERSION": "2024-02-15-preview"
            },
            "text-embedding-3-small": {
                "AZURE_OPENAI_DEPLOYMENT_NAME": "text-embedding-3-small",
                "AZURE_OPENAI_API_VERSION": "2024-05-01-preview"
            },
            "text-embedding-3-large": {
                "AZURE_OPENAI_DEPLOYMENT_NAME": "text-embedding-3-large",
                "AZURE_OPENAI_API_VERSION": "2024-05-01-preview"
            },
        }

    def get_azure_openai_config(self, model: str) -> Dict[str, str]:
        return self._AZURE_OPENAI_CONFIG[model]

    @staticmethod
    def get_azure_system_prompt_template() -> str:
        return """\
You are an AI assistant created by OpenAI to be helpful, harmless, and honest.
Your goal is to provide informative and substantive responses to queries while avoiding potential harms.
"""
