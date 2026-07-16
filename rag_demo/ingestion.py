from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
    loader=TextLoader("/Users/ramkrishnakhan/Agentic AI/Agentic Workflow/langchain-course/rag_demo/blog.txt", encoding="utf8")
    docs = loader.load()
    print("loaded documents...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    print("split documents...")
    embeddings=OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    PineconeVectorStore.from_documents(texts,embeddings, index_name=os.environ.get("PINECONE_INDEX_NAME"))
    print("ingested documents...")






