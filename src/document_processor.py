from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from haystack.nodes import BaseComponent, PDFToTextConverter, PreProcessor
from haystack.schema import Document, MultiLabel
from transformers import AutoTokenizer

from src.config import EMBDD_CONFIG


class DocumentProcessor(BaseComponent):
    outgoing_edges = 1

    def __init__(self):
        super().__init__()
        self.pdf_converter = PDFToTextConverter(keep_physical_layout=True)
        self.tokenizer = AutoTokenizer.from_pretrained(EMBDD_CONFIG["EMBDD_MODEL"])

        self.page_preprocessor = PreProcessor(
            split_by="page",
            split_length=1,  # Keep each page as a separate document
            split_respect_sentence_boundary=False,
            progress_bar=False,
            max_chars_check=20_000,
            add_page_number=True,
        )

        self.passage_preprocessor = PreProcessor(
                split_by="passage",
                tokenizer=self.tokenizer,
                split_length=self.tokenizer.model_max_length,
                split_respect_sentence_boundary=False,
                progress_bar=False,
                max_chars_check=20_000,
                add_page_number = True
            )

    def run(
            self,
            query: Optional[str] = None,
            file_paths: Optional[List[str]] = None,
            labels: Optional[MultiLabel] = None,
            documents: Optional[List[Document]] = None,
            meta: Optional[dict] = None
    ) -> Tuple[Dict[str, List[Document]], str]:

        final_docs = []
        for file_path in file_paths:
            pdf_docs = self.pdf_converter.convert(file_path=Path(file_path))
            page_docs = self.page_preprocessor.process(pdf_docs)

            for doc in page_docs:
                page_number = doc.meta.get("page", 1)
                processed_passages = self.passage_preprocessor.process([doc])
                assert len(processed_passages) <= self.tokenizer.model_max_length

                for passage in processed_passages:
                    passage.meta["page_number"] = page_number
                    final_docs.append(passage)

        return {"documents": final_docs}, "output_1"


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