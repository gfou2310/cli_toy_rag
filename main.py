from src.config import FAISS_CONFIG
from src.ingestion_pipeline import IngestionPipeline
from src.rag_pipeline import RAGPipeline
from src.vector_store import VectorStore

import pprint


def main():
    if FAISS_CONFIG["FAISS_INDEX_PATH"].exists() and FAISS_CONFIG["FAISS_CONFIG_PATH"].exists():
        print("Loading existing vector store...")
        _vector_store = VectorStore.load()
    else:
        print("No existing vector store found. Creating new one...")
        IngestionPipeline().run()
        print("New vector store created and saved.")

    rag_pipeline = RAGPipeline()
    result = rag_pipeline.run(query="How to remove the cylinder head?")
    print(result)

    pprint.pprint(result)


if __name__ == "__main__":
    main()