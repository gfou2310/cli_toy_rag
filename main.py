from src.rag_pipeline import RetrievalPipeline
from src.config import DATA_CONFIG, FAISS_CONFIG, MODEL_CONFIG
from src.document_processor import DocumentProcessor
from src.ingestion_pipeline import IngestionPipeline
from src.vector_store import VectorStore


def main():
    if FAISS_CONFIG["FAISS_INDEX_PATH"].exists() and FAISS_CONFIG["FAISS_CONFIG_PATH"].exists():
        print("Loading existing vector store...")
        _vector_store = VectorStore.load()
    else:
        print("No existing vector store found. Creating new one...")
        vector_store = VectorStore()

        print(DATA_CONFIG["PDF_DIR"])

        _ingestion_pipeline = IngestionPipeline(
            DocumentProcessor(),
            vector_store
        ).run(DATA_CONFIG["PDF_DIR"])

        vector_store.save()
        print("New vector store created and saved.")

    # retrieval = RetrievalPipeline(vector_store=_vector_store)
    #
    # result = retrieval.retrieve(
    #     query="How to remove the cylinder head?",
    #     top_k=5
    # )
    #
    # if result["success"]:
    #     for doc in result["documents"]:
    #         print(f"Document Start_________________________________: {doc.content} \n {doc.meta}"
    #               f"Document End__________________________________")
    # else:
    #     print(f"Error: {result['error']}")



if __name__ == "__main__":
    main()