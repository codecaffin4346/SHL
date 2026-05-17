import json
import os
import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

def build_vector_store(catalog_path="shl_product_catalog.json", persist_path="faiss_index"):
    with open(catalog_path, 'r', encoding='utf-8') as f:
        catalog = json.load(f, strict=False)

    documents = []
    for item in catalog:
        # Prepare text representation for embedding
        name = item.get('name', '')
        desc = item.get('description', '')
        keys = item.get('keys', [])
        job_levels = item.get('job_levels', [])
        duration = item.get('duration', '')
        
        content = f"Name: {name}\nDescription: {desc}\nCategories/Keys: {', '.join(keys)}\nJob Levels: {', '.join(job_levels)}\nDuration: {duration}"
        
        metadata = {
            "name": name,
            "url": item.get('link', ''),
            "keys": keys,
            "description": desc,
            "job_levels": job_levels,
            "duration": duration,
            "languages": item.get('languages', [])
        }
        
        documents.append(Document(page_content=content, metadata=metadata))

    print(f"Loaded {len(documents)} documents. Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Building FAISS index...")
    vector_store = FAISS.from_documents(documents, embeddings)
    
    print(f"Saving FAISS index to {persist_path}...")
    vector_store.save_local(persist_path)
    print("Done!")

if __name__ == "__main__":
    build_vector_store()
