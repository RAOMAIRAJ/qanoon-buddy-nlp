import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

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
    history: List[dict] = []
    
class PredictResponse(BaseModel):
    response: str

class RiskRequest(BaseModel):
    description: str

class RiskResponse(BaseModel):
    risk_level: str
    legal_consequences: List[str]

class FIRRequest(BaseModel):
    incident_details: str

class FIRResponse(BaseModel):
    fir_draft: str

class BailRequest(BaseModel):
    offense_description: str

class BailResponse(BaseModel):
    is_bailable: bool
    explanation: str

class MatchRequest(BaseModel):
    description: str

class MatchResponse(BaseModel):
    specialization: str | None
    city: str | None
    max_budget: int | None

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
            "You are Qanoon Buddy, an exceptionally intelligent, empathetic, and highly capable legal assistant for Pakistan. "
            "Your personality should be warm, deeply knowledgeable, and highly professional—similar to an expert attorney who truly cares about their client. "
            "Use the provided context to answer the user's question, but feel free to synthesize it naturally. "
            "Instead of dry recitations, use markdown formatting (like bullet points and bold text) to make your response easy to read and beautiful. "
            "If you don't know the answer, politely say so, but offer general guidance on who they should seek out (like a specialized lawyer).\n"
            "CRITICAL RULES:\n"
            "1. Always respond in the EXACT SAME language the user asked the question in (English, Urdu (اردو), or Roman Urdu).\n"
            "2. Keep a warm, assuring tone, using empathetic phrases ('I understand this is a difficult situation...', 'Here is what Pakistani law says...').\n"
            "3. Remember that you are an AI, not a human, and clarify that this does not constitute a formal attorney-client relationship.\n\n"
            "Context: {context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
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
        chat_history = []
        for msg in req.history:
            if msg.get("role") == "user":
                chat_history.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "ai":
                chat_history.append(AIMessage(content=msg.get("content", "")))
                
        result = rag_chain.invoke({
            "input": req.query,
            "chat_history": chat_history
        })
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

@app.post("/analyze-risk", response_model=RiskResponse)
async def analyze_risk(req: RiskRequest):
    try:
        import json
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.environ.get("GEMINI_API_KEY"),
            temperature=0.2,
        )
        prompt = (
            "You are a Pakistani legal AI expert. Analyze the following situation and return ONLY a valid JSON object. "
            "Do not include markdown blocks or any other text.\n\n"
            "Required JSON format:\n"
            "{\n"
            '  "risk_level": "Low", "Medium", or "High",\n'
            '  "legal_consequences": ["consequence 1", "consequence 2"]\n'
            "}\n\n"
            f"SITUATION: {req.description}"
        )
        result = llm.invoke(prompt)
        text = result.content.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        data = json.loads(text.strip())
        return RiskResponse(
            risk_level=data.get("risk_level", "Unknown"),
            legal_consequences=data.get("legal_consequences", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-fir", response_model=FIRResponse)
async def generate_fir(req: FIRRequest):
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.environ.get("GEMINI_API_KEY"),
            temperature=0.3,
        )
        prompt = (
            "You are a Pakistani legal AI expert. Based on the following incident details, draft a formal FIR (First Information Report) "
            "ready to be submitted to the police in Pakistan. Format it professionally. Reply in English or Urdu depending on the input language.\n\n"
            f"INCIDENT DETAILS: {req.incident_details}"
        )
        result = llm.invoke(prompt)
        return FIRResponse(fir_draft=result.content.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/calculate-bail", response_model=BailResponse)
async def calculate_bail(req: BailRequest):
    try:
        import json
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.environ.get("GEMINI_API_KEY"),
            temperature=0.2,
        )
        prompt = (
            "You are a Pakistani legal AI expert. Based on the following offense description or penal code, determine if the offense is bailable according to Pakistani law. "
            "Return ONLY a valid JSON object. Do not include markdown blocks or any other text.\n\n"
            "Required JSON format:\n"
            "{\n"
            '  "is_bailable": true or false,\n'
            '  "explanation": "Brief legal reasoning"\n'
            "}\n\n"
            f"OFFENSE: {req.offense_description}"
        )
        result = llm.invoke(prompt)
        text = result.content.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        data = json.loads(text.strip())
        
        is_bailable = data.get("is_bailable", False)
        if isinstance(is_bailable, str):
            is_bailable = is_bailable.lower() == "true"
            
        return BailResponse(
            is_bailable=is_bailable,
            explanation=data.get("explanation", "Could not determine explanation.")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/match-lawyer", response_model=MatchResponse)
async def match_lawyer(req: MatchRequest):
    try:
        import json
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.environ.get("GEMINI_API_KEY"),
            temperature=0.1,
        )
        prompt = (
            "You are a legal AI assistant for Pakistan. Your job is to parse a user's natural language request for a lawyer "
            "and extract the required city, specialization, and maximum budget (if mentioned).\n\n"
            "The specialization MUST be exactly one of the following strings (or null if not determinable):\n"
            "'family_law', 'criminal_law', 'corporate_law', 'tax_law', 'civil_law', 'real_estate', 'intellectual_property', 'labor_law', 'none'\n\n"
            "Return ONLY a valid JSON object. Do not include markdown blocks or any other text.\n"
            "Required JSON format:\n"
            "{\n"
            '  "specialization": "tax_law",\n'
            '  "city": "Lahore",\n'
            '  "max_budget": 5000\n'
            "}\n\n"
            f"USER REQUEST: {req.description}"
        )
        result = llm.invoke(prompt)
        text = result.content.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        data = json.loads(text.strip())
        
        spec = data.get("specialization")
        if spec == "none":
            spec = None
            
        return MatchResponse(
            specialization=spec,
            city=data.get("city"),
            max_budget=data.get("max_budget")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "index_loaded": vector_store is not None}
