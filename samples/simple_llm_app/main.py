from typing import Tuple

from constants import Constant

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def get_azure_chat_openai_info(model: str) -> Tuple[str, str]:
    constant = Constant()
    config = constant.get_azure_openai_config(model)
    return config["AZURE_OPENAI_DEPLOYMENT_NAME"], config["AZURE_OPENAI_API_VERSION"]


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

    system_template = "Translate the following from English into {language}"

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")],
    )

    prompt = prompt_template.invoke({"language": "Japanese", "text": message})

    # Runnable
    response = model.invoke(prompt).model_dump_json()
    print(response)

    # Streaming
    # for token in model.stream(messages):
    #     print(token.content, end="|")


if __name__ == "__main__":
    message = "hi!"
    main(message)
