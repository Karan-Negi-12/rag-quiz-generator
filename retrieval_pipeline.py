import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()
# ==========================================
# CONFIG
# ==========================================
DB_PATH = "db/chroma_db"
# ==========================================
# LOAD EMBEDDINGS
# ==========================================
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)
# ==========================================
# LOAD CHROMA DB
# ==========================================
db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)
print(
    f"\nLoaded Vector DB "
    f"with {db._collection.count()} chunks"
)
# ==========================================
# LOAD LLM
# ==========================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)
# ==========================================
# CHAT LOOP
# ==========================================
while True:
    query = input(
        "\nAsk a Question (or type exit): "
    )
    if query.lower() == "exit":
        break
    print("\nSearching Vector DB...")
    retriever = db.as_retriever(
        search_kwargs={"k": 5}
    )
    relevant_docs = retriever.invoke(query)
    print(
        f"\nRetrieved "
        f"{len(relevant_docs)} chunks"
    )
    print(
        "\n========== TOP 5 CHUNKS =========="
    )
    for i, doc in enumerate(
        relevant_docs,
        start=1
    ):
        print(f"\nChunk {i}")
        print("-" * 50)
        print(
            doc.page_content[:500]
        )
        print("-" * 50)
    # =========================================
    # BUILD CONTEXT
    # ==========================================
    context = "\n\n".join(
        [
            doc.page_content
            for doc in relevant_docs
        ]
    )
    prompt = f"""
You are a helpful AI assistant. Answer ONLY using the provided context. If the answer is not present in the context,
say: "I don't have enough information in the provided documents."
Question:
{query}
Context:
{context}
"""
    # ==========================================
    # GENERATE ANSWER
    # ==========================================
    response = llm.invoke(prompt)
    print(
        "\n========== ANSWER =========="
    )
    print(response.content)