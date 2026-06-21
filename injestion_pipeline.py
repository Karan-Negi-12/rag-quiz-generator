import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from functools import partial

# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------
load_dotenv()

def load_documents(docs_path="docs"):
    print(f"\nLoading documents from: {docs_path}")
    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"{docs_path} directory not found."
        )
    loader = DirectoryLoader(
    path=docs_path,
    glob="*.txt",
    loader_cls=partial(
        TextLoader,
        encoding="utf-8"
    )
)
    documents = loader.load()
    if not documents:
        raise Exception(
            "No txt files found inside docs folder."
        )
    print(f"\nLoaded {len(documents)} documents")
    for i, doc in enumerate(documents[:2]):
        print("\n-------------------------")
        print(f"Document {i+1}")
        print("-------------------------")
        print(
            f"Source: "
            f"{os.path.basename(doc.metadata['source'])}"
        )
        print(
            f"Length: "
            f"{len(doc.page_content)} characters"
        )
    return documents
# ==================================================
# SPLIT DOCUMENTS
# ==================================================
def split_documents(
    documents,
    chunk_size=1200,
    chunk_overlap=50
):
    print("\nSplitting documents...")
    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    print(f"\nCreated {len(chunks)} chunks")
    print("\n========== FIRST 5 CHUNKS ==========")
    for i, chunk in enumerate(chunks[:5]):
        print(f"\nChunk #{i+1}")
        print(
            f"Source: "
            f"{os.path.basename(chunk.metadata['source'])}"
        )
        print(
            f"Length: "
            f"{len(chunk.page_content)} chars"
        )
        print("-" * 50)
        print(chunk.page_content[:500])
        print("-" * 50)
    return chunks
# ==================================================
# CREATE VECTOR STORE
# ==================================================
def create_vector_store(
    chunks,
    persist_directory="db/chroma_db",
    batch_size=98,
    sleep_time=60
):
    print("\nInitializing Gemini Embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )
    print("Creating ChromaDB...")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    total_chunks = len(chunks)
    print(f"\nTotal Chunks: {total_chunks}")
    print(f"Batch Size: {batch_size}")
    for start_idx in range(
        0,
        total_chunks,
        batch_size
    ):
        end_idx = min(
            start_idx + batch_size,
            total_chunks
        )
        batch = chunks[start_idx:end_idx]
        print("\n================================")
        print(
            f"Processing Batch "
            f"{start_idx // batch_size + 1}"
        )
        print(
            f"Chunks "
            f"{start_idx + 1} -> {end_idx}"
        )
        print("================================")
        vectorstore.add_documents(batch)
        current_count = (
            vectorstore._collection.count()
        )
        print(
            f"Current Chroma Count: "
            f"{current_count}"
        )
        if end_idx < total_chunks:
            print(
                f"\nSleeping "
                f"{sleep_time} seconds..."
            )
            time.sleep(sleep_time)

    print("\nVector Store Created Successfully")
    return vectorstore
# ==================================================
# MAIN
# ==================================================

def main():
    DOCS_PATH = "docs"
    DB_PATH = "db/chroma_db"
    print(
        "\n========== RAG INGESTION =========="
    )
    if os.path.exists(DB_PATH):
        print(
            "\nExisting Vector Store Found."
        )
        embeddings = (
            GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001"
            )
        )
        db = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings
        )
        print(
            f"Stored Chunks: "
            f"{db._collection.count()}"
        )
        return
    # Step 1
    documents = load_documents(DOCS_PATH)
    # Step 2
    chunks = split_documents(
        documents,
        chunk_size=1200,
        chunk_overlap=50
    )
    # Step 3
    create_vector_store(
        chunks=chunks,
        persist_directory=DB_PATH,
        batch_size=98,
        sleep_time=60
    )
    print(
        "\n✅ Ingestion Completed Successfully"
    )
    
if __name__ == "__main__":
    main()