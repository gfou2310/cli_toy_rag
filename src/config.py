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
    "TEMPERATURE": 0.2,  # Controls randomness
    "TOP_P": 0.9,  # Nucleus sampling parameter
    "FREQUENCY_PENALTY": 0.1,  # Reduces word repetition
    "PRESENCE_PENALTY": 0.2,  # Encourages discussing new topics
    "MAX_LENGTH": 500,  # Maximum length of the generated response
    "TOP_K": 50,  # Limits vocabulary to top K tokens during generation
}

# Retriever parameters
RETRIEVER_CONFIG = {
    "TOP_K": 1,
}

# FAISS storage configuration
FAISS_CONFIG = {
    "FAISS_INDEX_PATH": Path(__file__).parent.parent / "cache" / "faiss_document_store.faiss",
    "FAISS_CONFIG_PATH": Path(__file__).parent.parent / "cache" / "faiss_document_store.json"
}