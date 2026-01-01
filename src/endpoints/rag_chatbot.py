# """
# RAG CHATBOT - Learn Vector Embeddings, VectorDB, and RAG
# ==========================================================

# This file will teach you step-by-step how RAG works!

# WHAT WE'LL BUILD:
# 1. Load company PDF documents
# 2. Split them into chunks
# 3. Convert chunks to vectors (embeddings)
# 4. Store in VectorDB (ChromaDB) - WITH PERSISTENCE!
# 5. Search and answer questions using RAG

# KEY FEATURE: Documents are processed ONCE and saved to disk!
# """

# import os
# import json
# import asyncio
# from dotenv import load_dotenv
# from openai import AsyncOpenAI
# from pydantic import BaseModel
# from typing import List, Dict
# import hashlib
# from fastapi import APIRouter
# try:
#     import chromadb  # VectorDB - stores our vectors
#     from chromadb.config import Settings
#     import PyPDF2  # Read PDF files
#     print("✅ All RAG libraries imported successfully!")
# except ImportError as e:
#     print(f"❌ Missing library: {e}")
#     print("Please run: pip install chromadb pypdf2")
#     exit(1)


# router = APIRouter()
# load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# DOCUMENTS_FOLDER = "Falcon_Reality_blog_Pdf"  # Change this to your folder name

# VECTORDB_PATH = "./vectordb_storage"  # Data persists here!

# # This function reads PDF files and extracts text
# def load_pdf_documents(folder_path: str) -> List[Dict[str, str]]:
#     """
#     WHAT IT DOES: Reads all PDF files from a folder
    
#     ANALOGY: Like opening all books in a library and copying the text
    
#     INPUT: Folder path with PDFs
#     OUTPUT: List of documents with text and metadata
#     """
#     documents = []
    
#     if not os.path.exists(folder_path):
#         print(f"❌ Folder not found: {folder_path}")
#         return documents
    
#     print(f"📂 Loading PDFs from: {folder_path}")
    
#     # Get all PDF files
#     pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    
#     for pdf_file in pdf_files:
#         pdf_path = os.path.join(folder_path, pdf_file)
        
#         try:
#             # Open and read PDF
#             with open(pdf_path, 'rb') as file:
#                 pdf_reader = PyPDF2.PdfReader(file)
#                 text = ""
                
#                 # Extract text from all pages
#                 for page_num in range(len(pdf_reader.pages)):
#                     page = pdf_reader.pages[page_num]
#                     text += page.extract_text()
                
#                 # Store document with metadata
#                 documents.append({
#                     "content": text,
#                     "source": pdf_file,
#                     "page_count": len(pdf_reader.pages)
#                 })
                
#                 print(f"  ✅ Loaded: {pdf_file} ({len(pdf_reader.pages)} pages)")
        
#         except Exception as e:
#             print(f"  ❌ Error loading {pdf_file}: {e}")
    
#     print(f"\n📚 Total documents loaded: {len(documents)}\n")
#     return documents

# # Split documents into smaller pieces
# def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
#     """
#     WHAT IT DOES: Splits large text into smaller chunks
    
#     WHY: 
#     - ChatGPT has token limits
#     - Smaller chunks = more precise search
#     - Overlap ensures we don't cut sentences in half
    
#     ANALOGY: Like cutting a long rope into smaller pieces with some overlap
#     so you don't lose the connection between pieces
    
#     EXAMPLE:
#     Text: "AI is amazing. It helps us learn. Machine learning is a subset of AI."
#     Chunks: ["AI is amazing. It helps us learn.", 
#              "It helps us learn. Machine learning is a subset of AI."]
#     (Notice the overlap!)
#     """
#     chunks = []
#     words = text.split()
    
#     for i in range(0, len(words), chunk_size - overlap):
#         chunk = " ".join(words[i:i + chunk_size])
#         chunks.append(chunk)
    
#     return chunks

# # Convert text to vectors using OpenAI
# async def generate_embeddings(texts: List[str]) -> List[List[float]]:
#     """
#     WHAT IT DOES: Converts text to vectors (numbers)
    
#     HOW: Uses OpenAI's text-embedding-3-small model
    
#     SIMPLE EXPLANATION:
#     - "I love dogs" → [0.23, 0.67, 0.12, 0.89, ...] (1536 numbers)
#     - "I love cats" → [0.25, 0.69, 0.10, 0.87, ...] (similar numbers!)
#     - "The sky is blue" → [0.01, 0.15, 0.92, 0.34, ...] (different numbers)
    
#     WHY VECTORS:
#     - Computers understand numbers better than words
#     - Similar meanings = similar numbers
#     - We can calculate "distance" between vectors to find similar content
#     """
#     print(f"🔄 Generating embeddings for {len(texts)} chunks...")
    
#     try:
#         response = await client.embeddings.create(
#             model="text-embedding-3-small",  # OpenAI's embedding model
#             input=texts
#         )
        
#         embeddings = [item.embedding for item in response.data]
#         print(f"✅ Generated {len(embeddings)} embeddings (each has {len(embeddings[0])} dimensions)")
#         return embeddings
    
#     except Exception as e:
#         print(f"❌ Error generating embeddings: {e}")
#         return []

# # Store and search vectors - DATA SAVED TO DISK!
# class VectorStore:
#     """
#     WHAT IT IS: A special database that stores vectors
    
#     🔑 KEY FEATURE: PERSISTENCE!
#     - First run: Processes documents and saves to disk
#     - Next runs: Loads from disk (no re-processing!)
    
#     HOW PERSISTENCE WORKS:
#     1. Data saved in 'vectordb_storage' folder
#     2. Each document gets a unique hash (fingerprint)
#     3. System checks if document already processed
#     4. Only new/changed documents are processed
    
#     ANALOGY: Like a smart library where you say "Find me books similar to this one"
#     and it instantly finds the most similar books!
#     And the library remembers all books even after you close it!
    
#     HOW IT WORKS:
#     - Each chunk has a vector (fingerprint)
#     - When you search, it converts your query to a vector
#     - Then finds chunks with similar vectors
#     - Returns the most relevant chunks
#     """
    
#     def __init__(self, collection_name: str = "company_docs", persist_directory: str = VECTORDB_PATH):
#         """
#         Initialize ChromaDB with PERSISTENCE
        
#         WHAT HAPPENS:
#         1. Creates folder to store data (if doesn't exist)
#         2. Loads existing data (if available)
#         3. Or creates new database (if first run)
#         """
#         print("🗄️ Initializing VectorDB (ChromaDB) with persistence...")
        
#         # Create persistence directory if it doesn't exist
#         os.makedirs(persist_directory, exist_ok=True)
        
#         # Create ChromaDB client with PERSISTENT storage
#         self.client = chromadb.PersistentClient(path=persist_directory)
        
#         # Create or get collection (like a table in database)
#         try:
#             self.collection = self.client.get_collection(collection_name)
#             count = self.collection.count()
#             print(f"✅ Loaded existing collection: {collection_name}")
#             print(f"📊 Found {count} existing chunks in database")
#         except:
#             self.collection = self.client.create_collection(collection_name)
#             print(f"✅ Created new collection: {collection_name}")
        
#         self.collection_name = collection_name
    
#     def get_document_hash(self, content: str) -> str:
#         """
#         Create unique fingerprint for document
        
#         WHY: To check if we already processed this document
        
#         HOW: Uses SHA-256 hash (like a unique ID)
        
#         EXAMPLE:
#         "Falcon Reality is awesome" → "a3f5b2c8d9e1..."
#         "Falcon Reality is awesome" → "a3f5b2c8d9e1..." (same!)
#         "Falcon Reality is great"   → "x7y9z2w4v6..." (different!)
#         """
#         return hashlib.sha256(content.encode()).hexdigest()
    
#     def is_document_processed(self, document_hash: str) -> bool:
#         """
#         Check if document already in database
        
#         RETURNS:
#         True = Already processed, skip it!
#         False = New document, process it!
#         """
#         try:
#             # Try to get document by hash
#             results = self.collection.get(
#                 ids=[f"hash_{document_hash}"],
#                 include=[]
#             )
#             return len(results['ids']) > 0
#         except:
#             return False
    
#     def add_documents(self, chunks: List[str], embeddings: List[List[float]], 
#                      metadatas: List[Dict], document_hash: str = None):
#         """
#         Add documents to VectorDB (PERSISTED TO DISK!)
        
#         PARAMETERS:
#         - chunks: The actual text pieces
#         - embeddings: Vector representations of chunks
#         - metadatas: Extra info (source file, page number, etc.)
#         - document_hash: Unique ID to track if document processed
        
#         WHAT HAPPENS:
#         1. Saves chunks + embeddings + metadata
#         2. Data written to disk in 'vectordb_storage' folder
#         3. Stays there forever (until you delete it)!
#         """
#         print(f"💾 Adding {len(chunks)} chunks to VectorDB...")
        
#         # Create unique IDs for each chunk
#         base_id = document_hash if document_hash else "doc"
#         ids = [f"{base_id}_{i}" for i in range(len(chunks))]
        
#         # Add hash marker to track processed documents
#         if document_hash:
#             ids.append(f"hash_{document_hash}")
#             chunks.append(f"[HASH_MARKER]")  # Marker chunk
#             embeddings.append(embeddings[0])  # Dummy embedding
#             metadatas.append({"type": "hash_marker", "hash": document_hash})
        
#         # Add to database (AUTOMATICALLY SAVED TO DISK!)
#         self.collection.add(
#             documents=chunks,      # The actual text
#             embeddings=embeddings, # The vectors
#             metadatas=metadatas,   # Source info
#             ids=ids                # Unique identifiers
#         )
        
#         print(f"✅ Successfully added {len(chunks)} chunks to VectorDB!")
#         print(f"💾 Data persisted to disk at: {VECTORDB_PATH}\n")
    
#     def search(self, query_embedding: List[float], top_k: int = 3) -> Dict:
#         """
#         Search for similar documents
        
#         WHAT IT DOES:
#         1. Takes your query vector
#         2. Compares it with all stored vectors
#         3. Finds the most similar ones
#         4. Returns top_k results
        
#         ANALOGY: Like asking "Show me the 3 most similar books to this one"
#         """
#         results = self.collection.query(
#             query_embeddings=[query_embedding],
#             n_results=top_k,
#             where={"type": {"$ne": "hash_marker"}}  # Exclude hash markers
#         )
        
#         return results
    
#     def get_stats(self):
#         """Get database statistics"""
#         count = self.collection.count()
#         print(f"📊 VectorDB Stats: {count} total items (including metadata)")
#         return count
    
#     def reset_database(self):
#         """
#         DELETE ALL DATA (use carefully!)
        
#         USE WHEN:
#         - You want to re-process all documents
#         - Database corrupted
#         - Testing
#         """
#         print("⚠️ RESETTING DATABASE - All data will be deleted!")
#         self.client.delete_collection(self.collection_name)
#         self.collection = self.client.create_collection(self.collection_name)
#         print("✅ Database reset complete")

# # Complete RAG implementation
# class RAGChatbot:
#     """
#     Complete RAG Chatbot System
    
#     HOW RAG WORKS (Simple Explanation):
    
#     WITHOUT RAG:
#     User: "What is Falcon Reality's vacation policy?"
#     ChatGPT: "I don't know about Falcon Reality" ❌
    
#     WITH RAG:
#     User: "What is Falcon Reality's vacation policy?"
#     → Convert question to vector
#     → Search VectorDB for similar company docs
#     → Found relevant chunks about vacation policy
#     → Give chunks to ChatGPT as context
#     ChatGPT: "According to company policy, employees get 20 days..." ✅
    
#     MAGIC: ChatGPT can now answer using YOUR private documents!
#     """
    
#     def __init__(self):
#         self.vector_store = VectorStore()
#         self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
#     async def initialize_knowledge_base(self, documents_folder: str, force_reprocess: bool = False):
#         """
#         Load all documents into the system
        
#         🔑 KEY FEATURE: SMART PROCESSING!
        
#         WHAT HAPPENS:
#         1. Check if documents already in database
#         2. If YES and force_reprocess=False → Skip! ✅
#         3. If NO or force_reprocess=True → Process! 🔄
        
#         PARAMETERS:
#         - documents_folder: Where PDFs are stored
#         - force_reprocess: True = re-process everything (default: False)
        
#         STEPS:
#         1. Load PDFs
#         2. Check each document hash
#         3. Split into chunks (only new docs)
#         4. Generate embeddings (only new docs)
#         5. Store in VectorDB (persisted to disk!)
#         """
#         print("=" * 60)
#         print("INITIALIZING RAG KNOWLEDGE BASE")
#         print("=" * 60)
        
#         # Check if database already has data
#         existing_count = self.vector_store.get_stats()
        
#         if existing_count > 0 and not force_reprocess:
#             print("\n🎯 SMART MODE: Database already has data!")
#             print(f"   Found {existing_count} items in VectorDB")
            
#             # Ask user what to do
#             print("\nOptions:")
#             print("  1. Use existing data (fast, no API costs) ✅")
#             print("  2. Re-process all documents (slow, costs money) 🔄")
            
#             choice = input("\nYour choice (1 or 2): ").strip()
            
#             if choice == "1":
#                 print("\n✅ Using existing VectorDB data!")
#                 print("=" * 60)
#                 print("✅ KNOWLEDGE BASE READY!")
#                 print("=" * 60)
#                 print()
#                 return True
#             else:
#                 print("\n🔄 Re-processing all documents...")
#                 force_reprocess = True
        
#         # Step 1: Load documents
#         documents = load_pdf_documents(documents_folder)
        
#         if not documents:
#             print("❌ No documents found! Please add PDFs to the folder.")
#             return False
        
#         # Step 2: Check which documents need processing
#         docs_to_process = []
        
#         for doc in documents:
#             doc_hash = self.vector_store.get_document_hash(doc["content"])
            
#             if force_reprocess or not self.vector_store.is_document_processed(doc_hash):
#                 docs_to_process.append({
#                     **doc,
#                     "hash": doc_hash
#                 })
#                 print(f"📝 Will process: {doc['source']}")
#             else:
#                 print(f"⏭️ Skipping (already processed): {doc['source']}")
        
#         if not docs_to_process:
#             print("\n✅ All documents already in database!")
#             print("=" * 60)
#             print("✅ KNOWLEDGE BASE READY!")
#             print("=" * 60)
#             print()
#             return True
        
#         print(f"\n📋 Processing {len(docs_to_process)} document(s)...\n")
        
#         # Step 3: Process each new document
#         for doc in docs_to_process:
#             print(f"\n🔄 Processing: {doc['source']}")
#             print("-" * 60)
            
#             # Chunk the document
#             chunks = chunk_text(doc["content"])
#             print(f"✂️ Created {len(chunks)} chunks")
            
#             # Generate embeddings
#             embeddings = await generate_embeddings(chunks)
            
#             if not embeddings:
#                 print(f"❌ Failed to generate embeddings for {doc['source']}")
#                 continue
            
#             # Create metadata for each chunk
#             metadatas = [
#                 {
#                     "source": doc["source"],
#                     "page_count": doc["page_count"],
#                     "chunk_index": i,
#                     "type": "content"
#                 }
#                 for i in range(len(chunks))
#             ]
            
#             # Store in VectorDB with document hash
#             self.vector_store.add_documents(
#                 chunks, 
#                 embeddings, 
#                 metadatas,
#                 document_hash=doc["hash"]
#             )
            
#             print(f"✅ Completed: {doc['source']}\n")
        
#         print("=" * 60)
#         print("✅ KNOWLEDGE BASE READY!")
#         print("=" * 60)
#         print(f"💾 All data saved to: {VECTORDB_PATH}")
#         print("💡 Next time you run, it will load from disk (no re-processing!)")
#         print()
        
#         return True
    
#     async def query(self, user_question: str) -> Dict:
#         """
#         Answer user question using RAG
        
#         THE RAG PROCESS:
#         1. Convert user question to vector
#         2. Search VectorDB for relevant chunks
#         3. Create prompt with found chunks as context
#         4. Ask ChatGPT to answer using the context
#         5. Return answer with sources
#         """
#         print(f"\n❓ User Question: {user_question}")
#         print("-" * 60)
        
#         # Step 1: Convert question to vector
#         print("🔄 Step 1: Converting question to vector...")
#         question_embedding = await generate_embeddings([user_question])
        
#         if not question_embedding:
#             return {"error": "Failed to generate question embedding"}
        
#         # Step 2: Search for relevant documents
#         print("🔍 Step 2: Searching VectorDB for relevant documents...")
#         search_results = self.vector_store.search(
#             query_embedding=question_embedding[0],
#             top_k=3  # Get top 3 most relevant chunks
#         )
        
#         # Extract relevant chunks
#         relevant_chunks = search_results['documents'][0]
#         sources = search_results['metadatas'][0]
        
#         print(f"✅ Found {len(relevant_chunks)} relevant chunks")
        
#         # Step 3: Create context from relevant chunks
#         context = "\n\n".join([
#             f"Document {i+1} (Source: {sources[i]['source']}):\n{chunk}"
#             for i, chunk in enumerate(relevant_chunks)
#         ])
        
#         print("\n📄 Context being sent to ChatGPT:")
#         print("-" * 60)
#         print(context[:500] + "..." if len(context) > 500 else context)
#         print("-" * 60)
        
#         # Step 4: Create prompt for ChatGPT
#         system_prompt = """You are a helpful AI assistant for Falcon Reality company.
        
# Answer questions based ONLY on the provided context from company documents.
# If the answer is not in the context, say "I don't have information about that in the company documents."

# Be specific and cite which document the information comes from."""
        
#         user_prompt = f"""Context from company documents:
# {context}

# Question: {user_question}

# Please provide a detailed answer based on the context above."""
        
#         # Step 5: Get answer from ChatGPT
#         print("\n🤖 Step 3: Asking ChatGPT to answer...")
        
#         try:
#             response = await self.client.chat.completions.create(
#                 model="gpt-4o-mini",  # You can use gpt-4 for better quality
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt}
#                 ],
#                 temperature=0.7
#             )
            
#             answer = response.choices[0].message.content
            
#             print("✅ Got response from ChatGPT!\n")
            
#             return {
#                 "answer": answer,
#                 "sources": [s["source"] for s in sources],
#                 "relevant_chunks": relevant_chunks
#             }
        
#         except Exception as e:
#             print(f"❌ Error getting response: {e}")
#             return {"error": str(e)}





# async def main():
#     """
#     Main function to run the RAG chatbot
    
#     🔑 PERSISTENCE EXPLAINED:
    
#     FIRST RUN:
#     - Loads PDFs
#     - Chunks them
#     - Generates embeddings (costs $$)
#     - Stores in VectorDB
#     - Saves to disk in 'vectordb_storage' folder
    
#     SECOND RUN (and onwards):
#     - Checks 'vectordb_storage' folder
#     - Finds existing data
#     - Asks: "Use existing or re-process?"
#     - If use existing: INSTANT loading! No API costs! ✅
    
#     WHEN TO RE-PROCESS:
#     - Added new PDF files
#     - Updated existing PDFs
#     - Database corrupted
#     - Want fresh start
#     """
#     print("\n" + "=" * 60)
#     print("🚀 RAG CHATBOT - LEARNING SYSTEM")
#     print("=" * 60)
    
#     # Initialize RAG system
#     rag = RAGChatbot()
    
#     # Load knowledge base from documents
#     # force_reprocess=False means it will use existing data if available
#     success = await rag.initialize_knowledge_base(
#         DOCUMENTS_FOLDER,
#         force_reprocess=False  # Change to True to always re-process
#     )
    
#     if not success:
#         print("❌ Failed to initialize knowledge base!")
#         return
    
#     # Show where data is stored
#     print(f"\n💾 VectorDB Location: {os.path.abspath(VECTORDB_PATH)}")
#     print(f"📁 Document Folder: {os.path.abspath(DOCUMENTS_FOLDER)}\n")
    
#     print("\n" + "=" * 60)
#     print("💬 INTERACTIVE MODE")
#     print("Type 'exit' to quit")
#     print("=" * 60)
    
#     while True:
#         user_input = input("\n❓ Your question: ").strip()
        
#         if user_input.lower() in ['exit', 'quit', 'q']:
#             print("👋 Goodbye!")
#             break
        
#         if not user_input:
#             continue
        
#         result = await rag.query(user_input)
        
#         if "error" in result:
#             print(f"❌ Error: {result['error']}")
#         else:
#             print(f"\n💬 ANSWER:")
#             print("-" * 60)
#             print(result["answer"])
#             print("-" * 60)
#             print(f"📚 Sources: {', '.join(result['sources'])}")

# @router.post("/rag-chat-bot-conversation")
# async def rag_chat_bot(request:str):
#     print("start converstion : {request}")
#     # Initialize RAG system
#     rag = RAGChatbot()
    
#     # Load knowledge base from documents
#     # force_reprocess=False means it will use existing data if available
#     success = await rag.initialize_knowledge_base(
#         DOCUMENTS_FOLDER,
#         force_reprocess=False  # Change to True to always re-process
#     )
    
#     if not success:
#         print("❌ Failed to initialize knowledge base!")
#         return
    
#     result = await rag.query(request)

#     return {"ai_res:":result["answer"]}







"""
RAG CHATBOT - Improved Version with Proper Endpoints
=====================================================

WHAT'S IMPROVED:
1. RAG system initializes ONCE at startup (not every request!)
2. Proper Pydantic models for request/response
3. Separate endpoint to add new documents
4. Better error handling
"""

import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import List, Dict, Optional
import hashlib
from fastapi import APIRouter, HTTPException
from contextlib import asynccontextmanager

try:
    import chromadb
    from chromadb.config import Settings
    import PyPDF2
    print("✅ All RAG libraries imported successfully!")
except ImportError as e:
    print(f"❌ Missing library: {e}")
    print("Please run: pip install chromadb pypdf2")
    exit(1)

router = APIRouter()
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

DOCUMENTS_FOLDER = "Falcon_Reality_blog_Pdf"
VECTORDB_PATH = "./vectordb_storage"

# ==================== PYDANTIC MODELS ====================
# These define what data the API expects and returns

class ChatRequest(BaseModel):
    """
    WHAT: Defines the shape of chat requests
    
    EXAMPLE:
    {
        "message": "What is VR?",
        "top_k": 3
    }
    """
    message: str
    top_k: int = 3  # How many relevant chunks to retrieve

class ChatResponse(BaseModel):
    """Response from the RAG chatbot"""
    answer: str
    sources: List[str]
    success: bool

class AddDocumentRequest(BaseModel):
    """Request to add a new document"""
    folder_path: Optional[str] = None  # If None, uses default folder

class AddDocumentResponse(BaseModel):
    """Response after adding documents"""
    success: bool
    message: str
    documents_processed: int

# ==================== HELPER FUNCTIONS ====================

def load_pdf_documents(folder_path: str) -> List[Dict[str, str]]:
    """Load all PDFs from a folder"""
    documents = []
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return documents
    
    print(f"📂 Loading PDFs from: {folder_path}")
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
                
                documents.append({
                    "content": text,
                    "source": pdf_file,
                    "page_count": len(pdf_reader.pages)
                })
                
                print(f"  ✅ Loaded: {pdf_file} ({len(pdf_reader.pages)} pages)")
        
        except Exception as e:
            print(f"  ❌ Error loading {pdf_file}: {e}")
    
    print(f"\n📚 Total documents loaded: {len(documents)}\n")
    return documents

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    words = text.split()
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

async def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Convert text to vectors using OpenAI"""
    print(f"🔄 Generating embeddings for {len(texts)} chunks...")
    
    try:
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        
        embeddings = [item.embedding for item in response.data]
        print(f"✅ Generated {len(embeddings)} embeddings")
        return embeddings
    
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        return []

# ==================== VECTOR STORE CLASS ====================

class VectorStore:
    """Manages the ChromaDB vector database"""
    
    def __init__(self, collection_name: str = "company_docs", persist_directory: str = VECTORDB_PATH):
        print("🗄️ Initializing VectorDB...")
        os.makedirs(persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        try:
            self.collection = self.client.get_collection(collection_name)
            count = self.collection.count()
            print(f"✅ Loaded existing collection: {collection_name}")
            print(f"📊 Found {count} existing chunks in database")
        except:
            self.collection = self.client.create_collection(collection_name)
            print(f"✅ Created new collection: {collection_name}")
        
        self.collection_name = collection_name
    
    def get_document_hash(self, content: str) -> str:
        """Create unique fingerprint for document"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def is_document_processed(self, document_hash: str) -> bool:
        """Check if document already in database"""
        try:
            results = self.collection.get(
                ids=[f"hash_{document_hash}"],
                include=[]
            )
            return len(results['ids']) > 0
        except:
            return False
    
    def add_documents(self, chunks: List[str], embeddings: List[List[float]], 
                     metadatas: List[Dict], document_hash: str = None):
        """Add documents to VectorDB"""
        print(f"💾 Adding {len(chunks)} chunks to VectorDB...")
        
        base_id = document_hash if document_hash else "doc"
        ids = [f"{base_id}_{i}" for i in range(len(chunks))]
        
        if document_hash:
            ids.append(f"hash_{document_hash}")
            chunks.append(f"[HASH_MARKER]")
            embeddings.append(embeddings[0])
            metadatas.append({"type": "hash_marker", "hash": document_hash})
        
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Successfully added chunks to VectorDB!")
    
    def search(self, query_embedding: List[float], top_k: int = 3) -> Dict:
        """Search for similar documents"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"type": {"$ne": "hash_marker"}}
        )
        return results
    
    def get_stats(self):
        """Get database statistics"""
        count = self.collection.count()
        return count

# ==================== RAG CHATBOT CLASS ====================

class RAGChatbot:
    """
    The main RAG Chatbot class
    
    IMPORTANT: This is initialized ONCE at startup!
    Not on every request like before.
    """
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.is_ready = False
    
    def check_ready(self):
        """Check if knowledge base has data"""
        count = self.vector_store.get_stats()
        self.is_ready = count > 0
        print(f"📊 VectorDB has {count} items. Ready: {self.is_ready}")
        return self.is_ready
    
    async def add_new_documents(self, folder_path: str = DOCUMENTS_FOLDER) -> Dict:
        """
        Add new documents to the knowledge base
        
        WHAT IT DOES:
        1. Loads PDFs from folder
        2. Checks which ones are new (not already processed)
        3. Only processes new documents
        4. Returns count of documents processed
        """
        print(f"\n{'='*60}")
        print("ADDING NEW DOCUMENTS")
        print(f"{'='*60}")
        
        documents = load_pdf_documents(folder_path)
        
        if not documents:
            return {"success": False, "message": "No documents found", "count": 0}
        
        docs_processed = 0
        
        for doc in documents:
            doc_hash = self.vector_store.get_document_hash(doc["content"])
            
            if self.vector_store.is_document_processed(doc_hash):
                print(f"⏭️ Skipping (already processed): {doc['source']}")
                continue
            
            print(f"\n🔄 Processing: {doc['source']}")
            
            chunks = chunk_text(doc["content"])
            print(f"✂️ Created {len(chunks)} chunks")
            
            embeddings = await generate_embeddings(chunks)
            
            if not embeddings:
                print(f"❌ Failed to generate embeddings for {doc['source']}")
                continue
            
            metadatas = [
                {
                    "source": doc["source"],
                    "page_count": doc["page_count"],
                    "chunk_index": i,
                    "type": "content"
                }
                for i in range(len(chunks))
            ]
            
            self.vector_store.add_documents(
                chunks, 
                embeddings, 
                metadatas,
                document_hash=doc_hash
            )
            
            docs_processed += 1
            print(f"✅ Completed: {doc['source']}")
        
        self.is_ready = True
        
        return {
            "success": True,
            "message": f"Processed {docs_processed} new documents",
            "count": docs_processed
        }
    
    async def query(self, user_question: str, top_k: int = 3) -> Dict:
        """
        Answer user question using RAG
        
        THE RAG PROCESS:
        1. Convert question to vector
        2. Search VectorDB for relevant chunks
        3. Create prompt with context
        4. Get answer from ChatGPT
        """
        if not self.is_ready:
            self.check_ready()
            if not self.is_ready:
                return {
                    "answer": "Knowledge base is empty. Please add documents first.",
                    "sources": [],
                    "success": False
                }
        
        print(f"\n❓ User Question: {user_question}")
        
        # Step 1: Convert question to vector
        question_embedding = await generate_embeddings([user_question])
        
        if not question_embedding:
            return {"answer": "Failed to process question", "sources": [], "success": False}
        
        # Step 2: Search for relevant documents
        search_results = self.vector_store.search(
            query_embedding=question_embedding[0],
            top_k=top_k
        )
        
        relevant_chunks = search_results['documents'][0]
        sources = search_results['metadatas'][0]
        
        print(f"✅ Found {len(relevant_chunks)} relevant chunks")
        
        # Step 3: Create context
        context = "\n\n".join([
            f"Document {i+1} (Source: {sources[i]['source']}):\n{chunk}"
            for i, chunk in enumerate(relevant_chunks)
        ])
        
        # Step 4: Get answer from ChatGPT
        system_prompt = """You are a helpful AI assistant for Falcon Reality company.
        
Answer questions based ONLY on the provided context from company documents.
If the answer is not in the context, don't say "I don't have information about that in the company documents." .say useing what context you got and do websearch and give answer

Be specific and cite which document the information comes from."""
        
        user_prompt = f"""Context from company documents:
{context}

Question: {user_question}

Please provide a detailed answer based on the context above."""
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "sources": list(set([s["source"] for s in sources])),
                "success": True
            }
        
        except Exception as e:
            print(f"❌ Error: {e}")
            return {"answer": str(e), "sources": [], "success": False}

# ==================== GLOBAL RAG INSTANCE ====================
# This is created ONCE when the server starts!

rag_chatbot = None

def get_rag_chatbot():
    """Get the global RAG chatbot instance"""
    global rag_chatbot
    if rag_chatbot is None:
        rag_chatbot = RAGChatbot()
        rag_chatbot.check_ready()
    return rag_chatbot

# ==================== API ENDPOINTS ====================

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    MAIN CHAT ENDPOINT
    
    WHAT IT DOES:
    - Takes user message
    - Searches knowledge base
    - Returns AI answer with sources
    
    EXAMPLE REQUEST:
    POST /api/chat
    {
        "message": "What is VR?",
        "top_k": 3
    }
    
    EXAMPLE RESPONSE:
    {
        "answer": "VR is...",
        "sources": ["file1.pdf", "file2.pdf"],
        "success": true
    }
    """
    rag = get_rag_chatbot()
    result = await rag.query(request.message, request.top_k)
    
    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        success=result["success"]
    )

@router.post("/add-documents", response_model=AddDocumentResponse)
async def add_documents(request: AddDocumentRequest = None):
    """
    ADD NEW DOCUMENTS ENDPOINT
    
    WHAT IT DOES:
    - Scans the documents folder
    - Processes only NEW documents (skips already processed)
    - Adds them to the VectorDB
    
    WHEN TO USE:
    - When you add new PDF files to the folder
    - Call this endpoint to process them
    
    EXAMPLE REQUEST:
    POST /api/add-documents
    {
        "folder_path": "Falcon_Reality_blog_Pdf"  // Optional
    }
    """
    rag = get_rag_chatbot()
    folder = request.folder_path if request and request.folder_path else DOCUMENTS_FOLDER
    
    result = await rag.add_new_documents(folder)
    
    return AddDocumentResponse(
        success=result["success"],
        message=result["message"],
        documents_processed=result["count"]
    )

@router.get("/status")
async def get_status():
    """
    CHECK RAG STATUS
    
    Returns info about the knowledge base
    """
    rag = get_rag_chatbot()
    count = rag.vector_store.get_stats()
    
    return {
        "status": "ready" if rag.is_ready else "empty",
        "document_chunks": count,
        "vectordb_path": VECTORDB_PATH,
        "documents_folder": DOCUMENTS_FOLDER
    }