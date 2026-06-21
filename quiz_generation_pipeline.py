import json
import random
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()
DB_PATH = "db/chroma_db"

db = Chroma(
    persist_directory=DB_PATH
)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

all_data = db.get()
all_chunks = all_data["documents"]
print(f"\nTotal Chunks Available: {len(all_chunks)}")

NUM_RANDOM_CHUNKS = 50
selected_chunks = random.sample(
    all_chunks,
    min(NUM_RANDOM_CHUNKS, len(all_chunks))
)
print(
    f"Selected {len(selected_chunks)} random chunks"
)

context = "\n\n".join(selected_chunks)

prompt = f"""
You are an expert quiz generator.
Using ONLY the provided context,
generate 20 multiple choice questions.
Rules:
1. Each question must have 4 options.
2. Only one option should be correct.
3. Include the correct answer.
4. Return ONLY valid JSON.
5. Do not include explanations.
6. Questions should be based strictly on the context.

JSON Format:
{{
  "mcqs": [
    {{
      "question": "",
      "options": [
        "",
        "",
        "",
        ""
      ],
      "answer": ""
    }}
  ]
}}
Context:
{context}
"""
response = llm.invoke(prompt)
print("\n========== GENERATED QUIZ ==========\n")
try:
    quiz_json = json.loads(response.content)
    print(
        json.dumps(
            quiz_json,
            indent=4
        )
    )
except Exception:
    print(response.content)