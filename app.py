import os
from fastapi import FastAPI, HTTPException
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
            "Use clear, professional, and empathetic language.\n\n"
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

@app.get("/health")
async def health():
    return {"status": "ok", "index_loaded": vector_store is not None}
