import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

app = FastAPI(title="Qanoon Buddy NLP Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    query: str
    
class PredictResponse(BaseModel):
    response: str

# Globals for the loaded model and vector store
vector_store = None
rag_chain = None

@app.on_event("startup")
def load_models():
    global vector_store, rag_chain
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.environ.get("GEMINI_API_KEY"),
            temperature=0.3,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        
        system_prompt = (
            "You are Qanoon Buddy, an AI legal assistant for Pakistan. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer based on the context, say that you don't know and advise consulting a real lawyer. "
            "Use clear, professional, and empathetic language.\n"
            "CRITICAL: Always respond in the EXACT SAME language the user asked the question in. "
            "If the user asks in English, respond in English. If they ask in Urdu (اردو), respond in Urdu. "
            "If they ask in Roman Urdu, respond in Roman Urdu.\n\n"
            "Context: {context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(vector_store.as_retriever(search_kwargs={"k": 5}), question_answer_chain)
        print("Models loaded successfully!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Warning: Failed to load FAISS index or Models: {e}. Check if ingest.py has been run.")

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if not rag_chain:
        raise HTTPException(status_code=500, detail="RAG Chain not initialized (missing FAISS index?)")
        
    try:
        result = rag_chain.invoke({"input": req.query})
        return PredictResponse(response=result["answer"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-document")
async def analyze_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    try:
        # Save file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
            
        # Extract text using PyPDFLoader or direct pypdf
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(temp_path)
        pages = loader.load()
        full_text = "\n".join([page.page_content for page in pages])
        
        os.remove(temp_path)
        
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the PDF.")
            
        # Truncate text to avoid token limits. Gemini 1.5 minimum is 1M tokens, but we use flash. Still fine for 20k chars
        if len(full_text) > 30000:
            full_text = full_text[:30000] + "... [Text Truncated]"
            
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.environ.get("GEMINI_API_KEY"),
            temperature=0.2,
        )
        
        analysis_prompt = (
            "You are an expert legal AI assistant. Analyze the following legal document (or excerpt).\n"
            "Provide a concise summary, highlight the key clauses or main points, and identify any potential risks, liabilities, or obligations.\n\n"
            f"DOCUMENT TEXT:\n{full_text}"
        )
        
        result = llm.invoke(analysis_prompt)
        return {"analysis": result.content}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "index_loaded": vector_store is not None}
