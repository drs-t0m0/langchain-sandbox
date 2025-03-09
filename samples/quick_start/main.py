from typing import Tuple

from constants import Constant

from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()


def get_azure_chat_openai_info(model: str) -> Tuple[str, str]:
    constant = Constant()
    config = constant.get_azure_openai_config(model)
    return config["AZURE_OPENAI_DEPLOYMENT_NAME"], config["AZURE_OPENAI_API_VERSION"]


def print_message(message: str, response):
    print(f"question:")
    print(message)
    print()

    print(f"answer:")
    print(response['content'])
    print()

    print(f"total_tokens: {response['usage_metadata']['total_tokens']}")


def main(message: str):
    AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_API_VERSION = get_azure_chat_openai_info("gpt-4o-mini")

    model = AzureChatOpenAI(
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    messages = [
        ("system", Constant.get_azure_system_prompt_template()),
        ("human", message),
    ]

    response = model.invoke(messages).model_dump()
    print_message(message, response)


if __name__ == "__main__":
    message = "Hello, AI!"
    main(message)
