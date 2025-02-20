from pathlib import Path

# Data source configuration
DATA_CONFIG = {
    "PDF_DIR": Path(__file__).parent.parent / "data" / "pdfs"
}

# Embedding configurations
EMBDD_CONFIG = {
    "EMBDD_MODEL": "sentence-transformers/all-mpnet-base-v2",
    "EMBDD_DIM": 768,
}

# Chat model configurations
CHAT_MODEL_CONFIG = {
    "MODEL": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "MAX_NEW_TOKENS": 250,
    "DO_SAMPLE": False, # Let us not allow our model to be creative/diverse
    "REPETITION_PENALTY": 1.0 # Let us encourage repetition to use more words found in the context/manual
}

# Retriever parameters
RETRIEVER_CONFIG = {
    "TOP_K": 1, # This is overfitted here as the first chunk we get is the one we are looking for
}

# FAISS storage configuration
FAISS_CONFIG = {
    "FAISS_INDEX_PATH": Path(__file__).parent.parent / "cache" / "faiss_document_store.faiss",
    "FAISS_CONFIG_PATH": Path(__file__).parent.parent / "cache" / "faiss_document_store.json"
}