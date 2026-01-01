
import os
import asyncio
import tempfile
import shutil
from fastapi import UploadFile, File
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

# These define what data the API expects and returns
class ChatRequest(BaseModel):
    message: str
    top_k: int = 3  # How many relevant chunks to retrieve

class ChatResponse(BaseModel):
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

#RAG CHATBOT CLASS 
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

# This is created ONCE when the server starts!
rag_chatbot = None

def get_rag_chatbot():
    """Get the global RAG chatbot instance"""
    global rag_chatbot
    if rag_chatbot is None:
        rag_chatbot = RAGChatbot()
        rag_chatbot.check_ready()
    return rag_chatbot

# API ENDPOINTS 
@router.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """
    UPLOAD DOCUMENT ENDPOINT
    
    Accepts a PDF file upload and processes it into the VectorDB
    
    EXAMPLE:
    POST /api/upload-document
    Content-Type: multipart/form-data
    file: [PDF file]
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Create a temporary file to store the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            # Copy uploaded file to temp file
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        try:
            # Read the PDF
            with open(temp_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
            
            # Create document object
            doc = {
                "content": text,
                "source": file.filename,
                "page_count": len(pdf_reader.pages)
            }
            
            # Get RAG instance
            rag = get_rag_chatbot()
            
            # Check if already processed
            doc_hash = rag.vector_store.get_document_hash(doc["content"])
            
            if rag.vector_store.is_document_processed(doc_hash):
                return {
                    "success": False,
                    "message": f"Document '{file.filename}' already exists in the knowledge base",
                    "filename": file.filename,
                    "already_exists": True
                }
            
            # Process the document
            print(f"\n📄 Processing uploaded file: {file.filename}")
            
            chunks = chunk_text(doc["content"])
            embeddings = await generate_embeddings(chunks)
            
            if not embeddings:
                raise HTTPException(status_code=500, detail="Failed to generate embeddings")
            
            metadatas = [
                {
                    "source": doc["source"],
                    "page_count": doc["page_count"],
                    "chunk_index": i,
                    "type": "content"
                }
                for i in range(len(chunks))
            ]
            
            rag.vector_store.add_documents(
                chunks, 
                embeddings, 
                metadatas,
                document_hash=doc_hash
            )
            
            rag.is_ready = True
            
            print(f"✅ Successfully processed: {file.filename}")
            
            return {
                "success": True,
                "message": f"Successfully processed '{file.filename}'",
                "filename": file.filename,
                "page_count": doc["page_count"],
                "chunks_created": len(chunks)
            }
            
        finally:
            # Clean up temp file
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

# Also add this endpoint to get list of documents
@router.get("/documents")
async def get_documents():
    """
    GET DOCUMENTS LIST
    
    Returns list of all documents in the knowledge base
    """
    rag = get_rag_chatbot()
    
    try:
        # Get all items from collection
        results = rag.vector_store.collection.get(
            where={"type": "content"},
            include=["metadatas"]
        )
        
        # Extract unique sources
        sources = set()
        for metadata in results['metadatas']:
            if 'source' in metadata:
                sources.add(metadata['source'])
        
        return {
            "success": True,
            "documents": sorted(list(sources)),
            "count": len(sources)
        }
    except Exception as e:
        return {
            "success": False,
            "documents": [],
            "count": 0,
            "error": str(e)
        }
    
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