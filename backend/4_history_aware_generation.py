from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage
)

load_dotenv()

# Connect to ChromaDB
persist_directory = "db/chroma_db"

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

# Load Local LLM
model = ChatOllama(
    model="llama3.2",
    temperature=0
)

# Chat Memory
chat_history = []

def ask_question(user_question):

    print(f"\n--- You asked: {user_question} ---")

    # STEP 1: Rewrite follow-up question
    if chat_history:

        rewrite_messages = [
            SystemMessage(
                content="""
                You are a query rewriter.

                Given chat history and a follow-up question,
                rewrite it into a standalone search query.

                Return ONLY the rewritten query.
                """
            )
        ] + chat_history + [
            HumanMessage(
                content=f"Follow-up Question: {user_question}"
            )
        ]

        result = model.invoke(rewrite_messages)

        search_question = result.content.strip()

        print(f"\nSearching for: {search_question}")

    else:
        search_question = user_question

    # STEP 2: Retrieve Documents
    retriever = db.as_retriever(
        search_kwargs={"k": 3}
    )

    docs = retriever.invoke(search_question)

    print(f"\nFound {len(docs)} documents")

    for i, doc in enumerate(docs, start=1):

        preview = doc.page_content[:150]

        print(f"\nDoc {i}:")
        print(preview)
        print("...")

    # STEP 3: Build Context
    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    # STEP 4: Final RAG Prompt
    messages = [
        SystemMessage(
            content="""
                You are a helpful RAG assistant.

                Answer ONLY from the provided context.

                If the answer is not present in the context,
                reply:

                "I don't have enough information to answer that question based on the provided documents."
                """
                        ),
                        HumanMessage(
                            content=f"""
                Context:
                {context}

                Question:
                {user_question}

                Answer:
                """
                        )
                    ]

    result = model.invoke(messages)
    answer = result.content

    # STEP 5: Save Memory
    chat_history.append(
        HumanMessage(content=user_question)
    )

    chat_history.append(
        AIMessage(content=answer)
    )

    print("\n===== ANSWER =====\n")
    print(answer)

    return answer

# Chat Loop
def start_chat():

    print("\n===== LOCAL RAG CHATBOT =====")
    print("Type 'quit' to exit")

    while True:

        question = input("\nYou: ")

        if question.lower() == "quit":
            print("\nGoodbye!")
            break

        ask_question(question)

# Main
if __name__ == "__main__":
    start_chat()