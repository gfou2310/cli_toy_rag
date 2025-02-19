from haystack import Pipeline

from src.config import DATA_CONFIG
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore


class IngestionPipeline:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=DocumentProcessor(), name="DocumentProcessor", inputs=["File"])
        self.pipeline.add_node(component=self.vector_store, name="VectorStore", inputs=["DocumentProcessor"])

    def run(self) -> None:
        pdf_dir = DATA_CONFIG["PDF_DIR"]
        if not pdf_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

        file_paths = list(pdf_dir.glob("*.pdf"))
        if file_paths:
            self.pipeline.run(file_paths=[str(file) for file in file_paths])
            self.vector_store.save()
        else:
            raise ValueError(f"No PDF files found in directory: {pdf_dir}")



