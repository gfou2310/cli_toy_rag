from pathlib import Path

# Data source configuration
DATA_CONFIG = {
    "PDF_DIR": Path("data/pdfs/"),
}

# Model configurations
MODEL_CONFIG = {
    "EMBEDDING_MODEL": "sentence-transformers/all-mpnet-base-v2",
    "EMBEDDING_DIMENSION": 768,
    "LLM_MODEL": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

# Retriever parameters
RETRIEVER_CONFIG = {
    "TOP_K": 3,
}

# FAISS storage configuration
FAISS_CONFIG = {
    "FAISS_INDEX_PATH": Path("cache/faiss_document_store.faiss"),
    "FAISS_CONFIG_PATH": Path("cache/faiss_document_store.json")
}