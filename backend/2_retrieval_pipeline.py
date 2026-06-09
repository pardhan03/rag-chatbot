from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

persist_directory = "db/chroma_db"

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"
)

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model
)

query = "Where google office is located?"

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.3,
    }
)

relevant_docs = retriever.invoke(query)

# print(f"User Query: {query}")
# # Display results
# print("--- Context ---")
# for i, doc in enumerate(relevant_docs, 1):
#     print(f"Document {i}:\n{doc.page_content}\n")

# Combined the query and the relevant document contents
# combined_input = f"""
# Based on the following documents, please answer this question:

# {query}

# Documents:
# {chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

# Please provide a clear, helpful answer using only the information from these documents.

# If you can't find the answer in the documents, say:
# "I don't have enough information to answer that question based on the provided documents."
# """

if not relevant_docs:
    print("\nNo relevant documents found.")
    exit()

print(f"\nRetrieved {len(relevant_docs)} document chunks\n")

for i, doc in enumerate(relevant_docs, 1):
    print(f"----- Chunk {i} -----")
    print(doc.page_content[:300])
    print("\n")


context = "\n\n".join(
    doc.page_content
    for doc in relevant_docs
)

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

prompt = f"""
You are a helpful assistant.

Use ONLY the information provided in the context below.

If the answer cannot be found in the context, reply:

"I don't have enough information to answer that question based on the provided documents."


Context:
{context}

Question:
{query}

Answer:
"""

response = llm.invoke(prompt)

print("\n===== ANSWER =====\n")
print(response.content)

