from typing import Any, Dict, List, Optional, Union

from haystack.nodes import BaseComponent, PDFToTextConverter, PreProcessor
from haystack.schema import Document, MultiLabel
from transformers import AutoTokenizer

from src.config import MODEL_CONFIG


class DocumentProcessor(BaseComponent):
    outgoing_edges = 1

    def __init__(self):
        super().__init__()
        self.pdf_converter = PDFToTextConverter(keep_physical_layout=True)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["EMBEDDING_MODEL"])

        self.page_preprocessor = PreProcessor(
            split_by="page",
            split_length=1,  # Keep each page as a separate document
            split_respect_sentence_boundary=False,
            add_page_number=True,
        )

        self.passage_preprocessor = PreProcessor(
                split_by="passage",
                tokenizer=self.tokenizer,
                split_length=self.tokenizer.model_max_length,
                split_respect_sentence_boundary=False,
                add_page_number = True
            )

    def run(
            self,
            query: Optional[str] = None,
            file_paths: Optional[List[str]] = None,
            labels: Optional[MultiLabel] = None,
            documents: Optional[List[Document]] = None,
            meta: Optional[dict] = None
    ) -> Dict[str, List[Document]]:

        pdf_docs = self.pdf_converter.convert(file_path=file_paths)
        page_docs = self.page_preprocessor.process(pdf_docs)

        final_docs = []
        for doc in page_docs:
            page_number = doc.meta.get("page", 1)  # Extract page number metadata
            processed_passages = self.passage_preprocessor.process([doc])

            for passage in processed_passages:
                passage.meta["page_number"] = page_number  # Retain page number in each passage
                final_docs.append(passage)

        return {"documents": final_docs}


    def run_batch(
            self,
            queries: Optional[Union[str, List[str]]] = None,
            file_paths: Optional[List[str]] = None,
            labels: Optional[Union[MultiLabel, List[MultiLabel]]] = None,
            documents: Optional[Union[List[Document], List[List[Document]]]] = None,
            meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            params: Optional[dict] = None,
            debug: Optional[bool] = None
    ):
        pass