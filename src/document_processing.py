from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from haystack.nodes import BaseComponent
from haystack.nodes import PreProcessor
from haystack.schema import Document
from haystack.schema import MultiLabel
from transformers import AutoTokenizer
from pdf2image import convert_from_path

from src.config import EMBDD_CONFIG
from src.text_extraction import extract_text_from_pdf_image


class DocumentProcessor(BaseComponent):
    outgoing_edges = 1

    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(EMBDD_CONFIG["EMBDD_MODEL"])

        # Note: The tokenizer's model_max_length attribute returns 512 tokens.
        # However, the all-mpnet-base-v2 model is designed to process a maximum of 384 tokens.
        # To ensure consistency between the tokenizer and the model, we explicitly use 384 as max length.
        # https://huggingface.co/sentence-transformers/all-mpnet-base-v2/discussions/15
        self.tokenizer.model_max_length = 384
        assert self.tokenizer.model_max_length == 384

        self.chunk_preprocessor = PreProcessor(
            split_by="token",
            split_length=int(self.tokenizer.model_max_length - 25),
            tokenizer=self.tokenizer,
            split_overlap=10,
            split_respect_sentence_boundary=False,
            progress_bar=False,
            add_page_number=True,
        )

    def run(
        self,
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[dict] = None,
    ) -> Tuple[Dict[str, List[Document]], str]:

        final_chunks = []
        for file_path in file_paths:
            images = convert_from_path(file_path, dpi=300)

            for i, image in enumerate(images):
                page_number = i + 1
                texts = extract_text_from_pdf_image(image)

                for text in texts:
                    page_document = Document.from_dict({"content": text, "content_type": "text"})

                    page_chunks = self.chunk_preprocessor.process(page_document)
                    for chunk in page_chunks:
                        chunk.meta["page"] = page_number
                        chunk.meta["file_path"] = file_path
                        chunk.meta["file_name"] = file_path.split("/")[-1]

                        # We'll add here the prefix <New Document: file name> on each chunk followed by a <Page number:> pointer
                        chunk.content = f"New Document:{chunk.meta["file_name"]}\nPage number: {chunk.meta['page']}\n{chunk.content}"

                        # Let us do a sanity check here to be sure our chunks comply to the tokenizers max length.
                        chunk_tokens = self.tokenizer.encode(chunk.content, truncation=False, add_special_tokens=True)
                        if len(chunk_tokens) > self.tokenizer.model_max_length:
                            raise ValueError(
                                f"Chunk length is {len(chunk_tokens)} but max length is {self.tokenizer.model_max_length}"
                            )

                        final_chunks.append(chunk)

        return {"documents": final_chunks}, "output_1"

    def run_batch(
        self,
        queries: Optional[Union[str, List[str]]] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[Union[MultiLabel, List[MultiLabel]]] = None,
        documents: Optional[Union[List[Document], List[List[Document]]]] = None,
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        params: Optional[dict] = None,
        debug: Optional[bool] = None,
    ):
        pass
