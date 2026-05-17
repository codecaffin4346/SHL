"""
Startup script: builds FAISS index if missing, then starts uvicorn.
Used by Render/Railway/Fly deployment.
"""
import os
import subprocess
import sys
from pathlib import Path

FAISS_INDEX = Path(__file__).parent / "faiss_index"
CATALOG = Path(__file__).parent / "shl_product_catalog.json"

def maybe_download_catalog():
    """Download catalog if not present (for cold deployments)."""
    if not CATALOG.exists():
        print("Downloading SHL product catalog...")
        import urllib.request
        url = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
        urllib.request.urlretrieve(url, CATALOG)
        print(f"Catalog downloaded: {CATALOG.stat().st_size} bytes")

def maybe_build_index():
    """Build FAISS index if it doesn't exist."""
    if not FAISS_INDEX.exists() or not any(FAISS_INDEX.iterdir()):
        print("Building FAISS vector store (first run — takes ~60s)...")
        from vector_store import build_vector_store
        build_vector_store(
            catalog_path=str(CATALOG),
            persist_path=str(FAISS_INDEX)
        )
        print("FAISS index built successfully.")
    else:
        print(f"FAISS index found at {FAISS_INDEX}, skipping build.")

if __name__ == "__main__":
    maybe_download_catalog()
    maybe_build_index()
    
    port = os.getenv("PORT", "8000")
    print(f"Starting uvicorn on port {port}...")
    os.execv(
        sys.executable,
        [sys.executable, "-m", "uvicorn", "main:app",
         "--host", "0.0.0.0", "--port", port]
    )
