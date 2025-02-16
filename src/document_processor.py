from haystack.nodes import PDFToTextConverter, PreProcessor
from pathlib import Path

class DocumentProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.pdf_converter = PDFToTextConverter()
        self.preprocessor = PreProcessor(
            clean_empty_lines=True,
            clean_whitespace=True,
            clean_header_footer=True,
            split_by="word",
            split_length=chunk_size,
            split_overlap=chunk_overlap,
            split_respect_sentence_boundary=True,
        )

    def process_documents(self, pdf_dir):
        documents = []
        pdf_files = Path(pdf_dir).glob("*.pdf")

        for pdf_file in pdf_files:
            # Convert PDF to text documents
            docs = self.pdf_converter.convert(file_path=pdf_file)

            # Preprocess and split documents
            processed_docs = self.preprocessor.process(docs)
            documents.extend(processed_docs)

        return documents