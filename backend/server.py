import os
import shutil
import time
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load environment variables
load_dotenv()

app = FastAPI(title="RAG Chatbot Backend API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
DB_DIR = os.path.join(BASE_DIR, "db", "chroma_db")

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_DIR), exist_ok=True)

# Global State for Ingestion
ingestion_state = {
    "status": "idle",  # idle, processing, completed, error
    "message": "No active ingestion",
    "logs": [],
    "total_chunks": 0,
    "start_time": None,
    "end_time": None
}

def log_message(message: str):
    timestamp = time.strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    print(formatted_msg)
    ingestion_state["message"] = message
    ingestion_state["logs"].append(formatted_msg)

# Request Models
class QueryRequest(BaseModel):
    question: str
    history: List[Dict[str, str]] = []  # [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    model_provider: str = "ollama"  # ollama or openai
    model_name: str = "llama3.2"    # llama3.2 or gpt-3.5-turbo / gpt-4o
    embedding_provider: str = "ollama"  # ollama or openai
    embedding_model: str = "nomic-embed-text" # nomic-embed-text or text-embedding-3-small
    search_type: str = "similarity"  # similarity, similarity_score_threshold, mmr
    k: int = 4
    score_threshold: float = 0.2
    openai_api_key: Optional[str] = None

class IngestRequest(BaseModel):
    splitter_type: str = "recursive"  # character, recursive, semantic, agentic
    chunk_size: int = 1500
    chunk_overlap: int = 200
    breakpoint_threshold_type: str = "percentile"  # percentile, standard_deviation
    breakpoint_threshold_amount: int = 70
    model_provider: str = "ollama"
    model_name: str = "llama3.2"
    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text"
    openai_api_key: Optional[str] = None

# Helper functions
def get_embeddings(provider: str, model_name: str, api_key: Optional[str] = None):
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if provider == "openai":
        if not api_key:
            raise HTTPException(status_code=400, detail="OpenAI API Key is required for OpenAI embeddings.")
        return OpenAIEmbeddings(model=model_name, openai_api_key=api_key)
    else:  # ollama
        return OllamaEmbeddings(model=model_name)

def get_llm(provider: str, model_name: str, api_key: Optional[str] = None, temperature: float = 0):
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if provider == "openai":
        if not api_key:
            raise HTTPException(status_code=400, detail="OpenAI API Key is required for OpenAI LLM.")
        return ChatOpenAI(model=model_name, temperature=temperature, openai_api_key=api_key)
    else:  # ollama
        return ChatOllama(model=model_name, temperature=temperature)

# Background Ingestion Pipeline Task
def run_ingestion_task(params: IngestRequest):
    global ingestion_state
    try:
        ingestion_state["status"] = "processing"
        ingestion_state["logs"] = []
        ingestion_state["start_time"] = time.time()
        
        log_message("Starting document ingestion pipeline...")
        
        # 1. Load Documents
        log_message(f"Loading documents from: {DOCS_DIR}")
        if not os.path.exists(DOCS_DIR) or not os.listdir(DOCS_DIR):
            raise ValueError(f"Docs directory is empty or does not exist: {DOCS_DIR}")
            
        loader = DirectoryLoader(
            path=DOCS_DIR,
            glob="*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
        log_message(f"Loaded {len(documents)} document files.")
        
        for doc in documents:
            filename = os.path.basename(doc.metadata.get("source", "unknown"))
            log_message(f" - {filename} ({len(doc.page_content)} characters)")

        # 2. Chunking Documents
        log_message(f"Splitting documents using strategy: {params.splitter_type}")
        
        chunks = []
        if params.splitter_type == "character":
            splitter = CharacterTextSplitter(
                chunk_size=params.chunk_size,
                chunk_overlap=params.chunk_overlap,
                separator="\n"
            )
            chunks = splitter.split_documents(documents)
            
        elif params.splitter_type == "recursive":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=params.chunk_size,
                chunk_overlap=params.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = splitter.split_documents(documents)
            
        elif params.splitter_type == "semantic":
            log_message("Initializing Semantic Chunker (requires embedding)...")
            embed_model = get_embeddings(params.embedding_provider, params.embedding_model, params.openai_api_key)
            splitter = SemanticChunker(
                embeddings=embed_model,
                breakpoint_threshold_type=params.breakpoint_threshold_type,
                breakpoint_threshold_amount=float(params.breakpoint_threshold_amount) / 100.0
            )
            chunks = splitter.split_documents(documents)
            
        elif params.splitter_type == "agentic":
            log_message("Initializing Agentic Chunker using LLM...")
            llm = get_llm(params.model_provider, params.model_name, params.openai_api_key, temperature=0)
            
            # Agentic chunking logic (conceptually similar to 7_agentic_chunking.py)
            for doc in documents:
                source = doc.metadata.get("source", "unknown")
                log_message(f"Agentic chunking file: {os.path.basename(source)}")
                
                # Split large text into rough paragraphs to avoid blowing context limits
                rough_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
                rough_chunks = rough_splitter.split_documents([doc])
                
                for i, r_chunk in enumerate(rough_chunks):
                    prompt = f"""You are a text chunking expert. Split this text into logical chunks.
Rules:
- Each chunk should be around 200-500 characters
- Split at natural topic or paragraph boundaries
- Keep related information together
- Put "<<<SPLIT>>>" between chunks

Text:
{r_chunk.page_content}

Return the text with <<<SPLIT>>> markers where you want to split:"""
                    
                    try:
                        response = llm.invoke(prompt)
                        marked_text = response.content
                        sub_chunks = marked_text.split("<<<SPLIT>>>")
                        for sc in sub_chunks:
                            cleaned = sc.strip()
                            if cleaned:
                                # Re-create Document metadata
                                metadata = doc.metadata.copy()
                                metadata["rough_part"] = i
                                from langchain_core.documents import Document
                                chunks.append(Document(page_content=cleaned, metadata=metadata))
                    except Exception as e:
                        log_message(f"Agentic split failed for a chunk: {e}. Falling back to recursive character splitter for this part.")
                        fallback_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                        chunks.extend(fallback_splitter.split_documents([r_chunk]))
                        
        else:
            raise ValueError(f"Unknown splitter type: {params.splitter_type}")
            
        log_message(f"Document splitting finished. Total chunks created: {len(chunks)}")
        ingestion_state["total_chunks"] = len(chunks)

        # 3. Create Vector Store
        log_message("Re-initializing Vector Store...")
        
        embed_model = get_embeddings(params.embedding_provider, params.embedding_model, params.openai_api_key)
        
        # Connect to Chroma and clear existing docs if database exists
        if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
            try:
                db = Chroma(persist_directory=DB_DIR, embedding_function=embed_model)
                all_ids = db.get()['ids']
                if all_ids:
                    db.delete(ids=all_ids)
                    log_message(f"Cleared {len(all_ids)} existing chunks from vector database.")
                else:
                    log_message("Vector database was already empty.")
            except Exception as e:
                log_message(f"Notice: Could not clear existing database chunks: {e}. Re-initializing client...")
        else:
            os.makedirs(DB_DIR, exist_ok=True)
        
        log_message("Creating embeddings and saving to ChromaDB...")
        
        # Batch upload to avoid request timeouts
        batch_size = 50
        vectorstore = None
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            log_message(f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} (chunks {i} to {min(i+batch_size, len(chunks))})...")
            if vectorstore is None:
                vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=embed_model,
                    persist_directory=DB_DIR,
                    collection_metadata={"hnsw:space": "cosine"}
                )
            else:
                vectorstore.add_documents(batch)
                
        log_message("Vector database successfully built and persisted.")
        ingestion_state["status"] = "completed"
        ingestion_state["end_time"] = time.time()
        duration = round(ingestion_state["end_time"] - ingestion_state["start_time"], 2)
        log_message(f"Ingestion pipeline completed in {duration} seconds! Ready for search and chat.")
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        log_message(f"ERROR: Ingestion failed: {str(e)}")
        print(error_trace)
        ingestion_state["status"] = "error"
        ingestion_state["end_time"] = time.time()

# Endpoints

@app.get("/api/docs")
def list_docs():
    """List all documents in the docs folder"""
    files = []
    if os.path.exists(DOCS_DIR):
        for name in os.listdir(DOCS_DIR):
            if name.endswith(".txt"):
                path = os.path.join(DOCS_DIR, name)
                stat = os.stat(path)
                # Word and char count
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                files.append({
                    "name": name,
                    "size": stat.st_size,
                    "chars": len(content),
                    "words": len(content.split()),
                    "modified": stat.st_mtime
                })
    return sorted(files, key=lambda x: x["name"])

@app.post("/api/docs/upload")
async def upload_doc(file: UploadFile = File(...)):
    """Upload a text file to the docs folder"""
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported currently.")
    
    file_path = os.path.join(DOCS_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"filename": file.filename, "message": "File uploaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

@app.delete("/api/docs/{filename}")
def delete_doc(filename: str):
    """Delete a document from the docs folder"""
    file_path = os.path.join(DOCS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    try:
        os.remove(file_path)
        return {"filename": filename, "message": "File deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

@app.post("/api/ingest")
def start_ingestion(params: IngestRequest, background_tasks: BackgroundTasks):
    """Trigger background document ingestion"""
    global ingestion_state
    if ingestion_state["status"] == "processing":
        raise HTTPException(status_code=400, detail="An ingestion pipeline is already running.")
        
    background_tasks.add_task(run_ingestion_task, params)
    return {"message": "Ingestion pipeline triggered in the background.", "status": "processing"}

@app.get("/api/ingest/status")
def get_ingestion_status():
    """Get the status of the current or last ingestion pipeline"""
    return ingestion_state

@app.get("/api/db/stats")
def get_db_stats(embedding_provider: str = "ollama", embedding_model: str = "nomic-embed-text", openai_api_key: Optional[str] = None):
    """Get statistics about the vector database"""
    try:
        if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
            return {"status": "empty", "total_chunks": 0, "collection_name": None}
            
        embed_model = get_embeddings(embedding_provider, embedding_model, openai_api_key)
        db = Chroma(persist_directory=DB_DIR, embedding_function=embed_model)
        
        # Get count
        col_data = db.get()
        return {
            "status": "active",
            "total_chunks": len(col_data["ids"]),
            "collection_name": db._collection.name if hasattr(db, "_collection") else "unknown",
            "documents_represented": list(set([os.path.basename(meta.get("source", "unknown")) for meta in col_data["metadatas"] if meta]))
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "total_chunks": 0}

@app.post("/api/db/clear")
def clear_db():
    """Clear all data in the vector database"""
    global ingestion_state
    try:
        if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
            try:
                from langchain_ollama import OllamaEmbeddings
                embed_model = OllamaEmbeddings(model="nomic-embed-text")
                db = Chroma(persist_directory=DB_DIR, embedding_function=embed_model)
                all_ids = db.get()['ids']
                if all_ids:
                    db.delete(ids=all_ids)
            except Exception as e:
                print(f"Failed to clear database documents: {e}")
                
        ingestion_state = {
            "status": "idle",
            "message": "Vector database cleared",
            "logs": [],
            "total_chunks": 0,
            "start_time": None,
            "end_time": None
        }
        return {"message": "Vector database cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {str(e)}")

@app.post("/api/chat")
def chat(req: QueryRequest):
    """Query the RAG chatbot with history-aware retrieval"""
    try:
        # Verify vector db exists
        if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
            raise HTTPException(
                status_code=400, 
                detail="Vector database is empty. Please upload documents and run ingestion first."
            )
            
        # 1. Initialize Embeddings & LLM
        embed_model = get_embeddings(req.embedding_provider, req.embedding_model, req.openai_api_key)
        llm = get_llm(req.model_provider, req.model_name, req.openai_api_key, temperature=0)
        
        # Connect to ChromaDB
        db = Chroma(persist_directory=DB_DIR, embedding_function=embed_model)
        
        # 2. Build history messages for LangChain
        chat_history_messages = []
        for msg in req.history:
            if msg["role"] == "user":
                chat_history_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history_messages.append(AIMessage(content=msg["content"]))
                
        # 3. Query Rewriting (if history is present)
        search_question = req.question
        rewritten = False
        
        if chat_history_messages:
            rewrite_prompt = [
                SystemMessage(
                    content="""You are a search query rewriter for a RAG chatbot.
Given a chat history and a follow-up question, rewrite it into a single, standalone search query that contains all necessary context from history.
Return ONLY the rewritten query. Do not add any conversational text, explanations, or quotes."""
                )
            ] + chat_history_messages + [
                HumanMessage(content=f"Follow-up Question: {req.question}")
            ]
            
            try:
                rewrite_res = llm.invoke(rewrite_prompt)
                rewritten_query = rewrite_res.content.strip()
                if rewritten_query and rewritten_query != req.question:
                    search_question = rewritten_query
                    rewritten = True
            except Exception as e:
                print(f"Query rewrite failed: {e}. Using original question.")

        # 4. Document Retrieval
        # We perform search with scores to display in UI and apply threshold
        retrieved_docs_with_scores = []
        try:
            if req.search_type == "similarity":
                retrieved_docs_with_scores = db.similarity_search_with_score(search_question, k=req.k)
            elif req.search_type == "similarity_score_threshold":
                # Cosine distance in Chroma: lower means more similar (0 is identical, 1 or 2 is completely different)
                # Langchain similarity_search_with_relevance_scores converts score, but similarity_search_with_score returns raw distance
                raw_results = db.similarity_search_with_score(search_question, k=req.k)
                # Map distance to score: score = 1.0 - distance
                retrieved_docs_with_scores = []
                for doc, dist in raw_results:
                    score = 1.0 - dist
                    if score >= req.score_threshold:
                        retrieved_docs_with_scores.append((doc, dist))
            elif req.search_type == "mmr":
                # MMR search doesn't return scores directly, so we fetch documents and compute distance
                docs = db.max_marginal_relevance_search(search_question, k=req.k)
                # Query vector db with score for these specific docs
                retrieved_docs_with_scores = []
                for doc in docs:
                    # Search for this exact document page content
                    results = db.similarity_search_with_score(doc.page_content, k=1)
                    if results:
                        retrieved_docs_with_scores.append(results[0])
                    else:
                        retrieved_docs_with_scores.append((doc, 0.5))
            else:
                retrieved_docs_with_scores = db.similarity_search_with_score(search_question, k=req.k)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

        # Build clean chunk results for frontend
        chunks_info = []
        context_parts = []
        
        for doc, score in retrieved_docs_with_scores:
            # Normalize score (Chroma returns L2/cosine distance; cosine distance is [0, 2])
            # For HNSW space cosine: distance = 1 - cosine_similarity. So similarity = 1 - distance
            # Let's show similarity percentage: (1 - distance) * 100
            sim_score = max(0.0, min(1.0, 1.0 - score))
            source_file = os.path.basename(doc.metadata.get("source", "unknown"))
            
            chunks_info.append({
                "content": doc.page_content,
                "source": source_file,
                "score": float(round(sim_score, 4))
            })
            context_parts.append(doc.page_content)
            
        context = "\n\n".join(context_parts)

        # 5. Final Generation
        if not context_parts:
            answer = "I don't have enough information to answer that question based on the provided documents (no matching chunks found)."
        else:
            messages = [
                SystemMessage(
                    content="""You are a helpful RAG (Retrieval-Augmented Generation) assistant.
Answer the user's question using ONLY the provided context. Do not make up facts or use external knowledge.
If the answer is not present in the context, reply exactly with:
"I don't have enough information to answer that question based on the provided documents."
Provide a clear, cohesive response."""
                ),
                HumanMessage(
                    content=f"""Context:
{context}

Question:
{req.question}

Answer:"""
                )
            ]
            
            generator_res = llm.invoke(messages)
            answer = generator_res.content.strip()

        return {
            "answer": answer,
            "rewritten_query": search_question if rewritten else None,
            "chunks": chunks_info
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Chat execution failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
