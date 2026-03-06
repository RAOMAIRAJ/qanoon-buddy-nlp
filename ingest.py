import os
import glob
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

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
    # Read each PDF
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
        
    print(f"Created {len(chunks)} chunks. Initializing embedding model...")
    
    # We use a fast, small embedding model locally
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("Building FAISS vector index (this may take a few minutes)...")
    vectorstore = FAISS.from_texts(chunks, embeddings, metadatas=metadatas)
    
    print(f"Saving index to {INDEX_PATH}...")
    vectorstore.save_local(INDEX_PATH)
    
    print("Done! The dataset is ready for Semantic Search.")

if __name__ == "__main__":
    main()
