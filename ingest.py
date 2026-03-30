import os
import glob
import time
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data/laws"
INDEX_PATH = "faiss_index"

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def main():
    print(f"Scanning for PDFs in {DATA_DIR}...")
    pdf_files = glob.glob(os.path.join(DATA_DIR, "**/*.pdf"), recursive=True)
    if not pdf_files:
        print("No PDFs found!")
        return
        
    print(f"Found {len(pdf_files)} PDFs. Extracting text...")
    
    docs = []
    for i, path in enumerate(pdf_files):
        print(f"[{i+1}/{len(pdf_files)}] Reading {os.path.basename(path)}")
        try:
            text = extract_text_from_pdf(path)
            if len(text.strip()) > 50:
                docs.append({"text": text, "source": os.path.basename(path)})
        except Exception as e:
            print(f"Error reading {path}: {e}")

    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    
    chunks = []
    metadatas = []
    
    for doc in docs:
        split_texts = text_splitter.split_text(doc["text"])
        chunks.extend(split_texts)
        metadatas.extend([{"source": doc["source"]}] * len(split_texts))
        
    print(f"Created {len(chunks)} chunks. Initializing Google embedding model...")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.environ.get("GEMINI_API_KEY"),
    )
    
    # Using 100 chunks per batch to avoid "Tokens Per Minute" payload limits
    BATCH_SIZE = 100
    vectorstore = None
    
    import time
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i:i+BATCH_SIZE]
        batch_metas = metadatas[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Embedding batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)...")
        
        for attempt in range(4):
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_texts(batch_chunks, embeddings, metadatas=batch_metas)
                else:
                    batch_vs = FAISS.from_texts(batch_chunks, embeddings, metadatas=batch_metas)
                    vectorstore.merge_from(batch_vs)
                break
            except Exception as e:
                if attempt == 3:
                    print(f"FATAL ERROR on batch {batch_num}: {str(e)}")
                    if vectorstore is not None:
                        print(f"Saving partial index to {INDEX_PATH} before crashing...")
                        vectorstore.save_local(INDEX_PATH)
                    raise Exception(f"Failed to embed batch after 4 attempts. You may have exhausted your Gemini Daily Quota. Error: {e}")
                    
                if "429" in str(e) or "quota" in str(e).lower() or "rate" in str(e).lower() or "RESOURCE_EXHAUSTED" in str(e):
                    wait_time = 15 * (attempt + 1)
                    print(f"  Quota hit! Limits active. Waiting {wait_time}s before retry {attempt+1}/4... (Error: {str(e)[:100]})")
                    time.sleep(wait_time)
                else:
                    raise
    
    print(f"Saving index to {INDEX_PATH}...")
    vectorstore.save_local(INDEX_PATH)
    
    print("Done! The dataset is ready for Semantic Search.")

if __name__ == "__main__":
    main()
