from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]

# Load documents
loader = WebBaseLoader(urls)
docs_list = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,
    chunk_overlap=0
)

doc_splits = text_splitter.split_documents(docs_list)

# Create embeddings and store documents in Chroma
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=HuggingFaceEmbeddings(),
    persist_directory="./.chroma",
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

# Test retrieval
results = retriever.invoke("What is agent memory?")

for i, doc in enumerate(results, 1):
    print(f"\nDocument {i}")
    print("Source:", doc.metadata.get("source"))
    print("Content:", doc.page_content[:300])