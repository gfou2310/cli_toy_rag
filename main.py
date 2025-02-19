import pprint

from src.config import FAISS_CONFIG
from src.ingestion_pipeline import IngestionPipeline
from src.rag_pipeline import RAGPipeline
from src.vector_store import VectorStore


def main():
    if FAISS_CONFIG["FAISS_INDEX_PATH"].exists() and FAISS_CONFIG["FAISS_CONFIG_PATH"].exists():
        print("Loading existing vector store...")
        vector_store = VectorStore(doc_store_exists=True)
    else:
        print("No existing vector store found. Creating new one...")
        vector_store = VectorStore()
        IngestionPipeline(vector_store=vector_store).run()
        print("New vector store created and saved.")

    rag_pipeline = RAGPipeline(vector_store=vector_store)

    # Continuous interaction loop
    print("\nWelcome to the RAG system! Type 'quit' or 'exit' to end the session.")
    while True:
        try:
            # Get user input
            query = input("\nUser input:").strip()

            # Check for exit commands
            if query.lower() in ['quit', 'exit']:
                print("Thank you for using the RAG system. Goodbye!")
                break

            # Skip empty queries
            if not query:
                print("Please enter a valid question.")
                continue

            result = rag_pipeline.run(query=query)
            print("\nAssistant Response:")
            pprint.pprint(result["results"][0])

        except KeyboardInterrupt:
            print("\nProgram interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again with a different question.")


if __name__ == "__main__":
    main()