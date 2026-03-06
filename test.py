import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

try:
    print("Loading HuggingFaceEmbeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Loading FAISS...")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    print("Loading Gemini...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
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
    
    print("Creating Chains...")
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(vector_store.as_retriever(search_kwargs={"k": 5}), question_answer_chain)
    print("SUCCESS")
except BaseException as e:
    import traceback
    traceback.print_exc()
    print("FAILED:", type(e), e)
