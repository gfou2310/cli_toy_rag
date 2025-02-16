from src.config import CONFIG
from src.document_processor import DocumentProcessor
from src.document_store import VectorStore
from src.embeddings import EmbeddingModel
from src.rag_pipeline import RAGPipeline

def main():
    # Initialize components
    doc_processor = DocumentProcessor(
        chunk_size=CONFIG["CHUNK_SIZE"],
        chunk_overlap=CONFIG["CHUNK_OVERLAP"]
    )
    vector_store = VectorStore()
    embedding_model = EmbeddingModel(CONFIG["EMBEDDING_MODEL"])

    # Process documents
    documents = doc_processor.process_documents(CONFIG["PDF_DIR"])

    # Generate embeddings and add to vector store
    embeddings = embedding_model.embed([doc.content for doc in documents])
    vector_store.add_documents(documents, embeddings)

    # Initialize RAG pipeline
    rag = RAGPipeline(vector_store, embedding_model)

    # Example usage
    query = "How to remove the cylinder head?"
    response = rag.generate_response(query)
    print(f"Query: {query}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()