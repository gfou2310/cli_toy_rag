from pathlib import Path
from typing import Union

from haystack import Pipeline


class IngestionPipeline:
    def __init__(self, document_processor, vector_store):
        self.pipeline = Pipeline()
        self.document_processor = document_processor
        self.vector_store = vector_store
        self.pipeline.add_node(component=self.document_processor, name="DocumentProcessor", inputs=["File"])
        self.pipeline.add_node(component=self.vector_store, name="VectorStore", inputs=["DocumentProcessor"])

    def run(self, document_path: Union[str, Path]):
        files = list(document_path.glob("*.pdf"))

        for file in files:
            doc_result = self.document_processor.run(file_paths=str(file))
            if doc_result and "documents" in doc_result:
                self.vector_store.run(documents=doc_result["documents"])
            print(f"Processed and stored: {file}")

        return f"Processed {len(files)} documents"
