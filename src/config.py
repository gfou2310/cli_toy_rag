from pathlib import Path

CONFIG = {
    "PDF_DIR": Path("data/pdfs"),
    "EMBEDDING_MODEL": "all-mpnet-base-v2",
    "LLM_MODEL": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "CHUNK_SIZE": 500,
    "CHUNK_OVERLAP": 50,
    "TOP_K": 3
}