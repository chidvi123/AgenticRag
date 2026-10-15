from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader,UnstructuredURLLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

urls=[
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]


docs = [UnstructuredURLLoader(urls=[url],chunking_strategy="basic",max_characters=1000000).load() for url in urls]

docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,chunk_overlap=0
)

doc_splits = text_splitter.split_documents(docs_list)


"""
vectorstore = Chroma.from_documents(
    documents = doc_splits,
    collection_name='rag-chroma',
    embedding=HuggingFaceEmbeddings(),
    persist_directory="./.chroma",
)

"""

retriever = Chroma(
    collection_name='rag-chroma',
    embedding_function=HuggingFaceEmbeddings(),
    persist_directory="./.chroma",
).as_retriever()


