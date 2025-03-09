import faiss
from typing import Tuple, List

from constants import Constant

from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()


def get_azure_chat_openai_info(model: str) -> Tuple[str, str]:
    constant = Constant()
    config = constant.get_azure_openai_config(model)
    return config["AZURE_OPENAI_DEPLOYMENT_NAME"], config["AZURE_OPENAI_API_VERSION"]


def main():
    AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_API_VERSION = get_azure_chat_openai_info("text-embedding-3-large")

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    # InMemory:
    vector_store = InMemoryVectorStore(embeddings)

    # FAISS:
    # index = faiss.IndexFlatL2(len(embeddings.embed_query("How were Nike's margins impacted in 2023?")))
    # vector_store = FAISS(
    #     embedding_function=embeddings,
    #     index=index,
    #     docstore=InMemoryDocstore(),
    #     index_to_docstore_id={},
    # )

    # documents = [
    #     Document(
    #         page_content="Dogs are great companions, known for their loyalty and friendliness.",
    #         metadata={"source": "mammal-pets-doc"},
    #     ),
    #     Document(
    #         page_content="Cats are independent pets that often enjoy their own space.",
    #         metadata={"source": "mammal-pets-doc"},
    #     ),
    # ]

    file_path = "data/nke-10k-2023.pdf"
    loader = PyPDFLoader(file_path)

    docs = loader.load()

    # print(len(docs))
    # print(f"{docs[0].page_content[:200]}\n")
    # print(docs[0].metadata)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    # print(len(all_splits))

    # vector_1 = embeddings.embed_query(all_splits[0].page_content)
    # vector_2 = embeddings.embed_query(all_splits[1].page_content)
    #
    # assert len(vector_1) == len(vector_2)
    # print(f"Generated vectors of length {len(vector_1)}\n")
    # print(vector_1[:10])

    ids = vector_store.add_documents(documents=all_splits)

    # Sync query:
    # results = vector_store.similarity_search("How many distribution centers does Nike have in the US?")
    # print(results[0])

    # Async query:
    # results = await vector_store.asimilarity_search("When was Nike incorporated?")
    # print(results[0])

    # Return scores:
    results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
    doc, score = results[0]
    print(f"Score: {score}\n")
    print(doc)

    # Embedded query:
    # embedding = embeddings.embed_query("How were Nike's margins impacted in 2023?")
    # results = vector_store.similarity_search_by_vector(embedding)
    # print(results[0])

    # @chain
    # def retriever(query: str) -> List[Document]:
    #     return vector_store.similarity_search(query, k=1)

    # retriever = vector_store.as_retriever(
    #     search_type="similarity",
    #     search_kwargs={"k": 1},
    # )
    #
    # retriever.batch(
    #     [
    #         "How many distribution centers does Nike have in the US?",
    #         "When was Nike incorporated?",
    #     ],
    # )


if __name__ == "__main__":
    main()
