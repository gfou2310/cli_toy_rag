from typing import Any, Dict, List, Optional, Tuple, Union

from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import BaseComponent, EmbeddingRetriever
from haystack.schema import Document, MultiLabel

from src.config import EMBDD_CONFIG, FAISS_CONFIG


class VectorStore(BaseComponent):
    outgoing_edges = 1

    def __init__(self, doc_store_exists: bool = False):
        super().__init__()

        self.doc_store_exists=doc_store_exists
        self.document_store = FAISSDocumentStore(
            embedding_dim=EMBDD_CONFIG["EMBDD_DIM"],
            similarity="dot_product",
            faiss_index_path=FAISS_CONFIG["FAISS_INDEX_PATH"] if self.doc_store_exists else None,
            faiss_config_path=FAISS_CONFIG["FAISS_CONFIG_PATH"] if self.doc_store_exists else None

        )
        self.retriever = EmbeddingRetriever(
            document_store=self.document_store,
            embedding_model=EMBDD_CONFIG["EMBDD_MODEL"]
        )

    def save(self) -> None:
        """Save the document store state"""
        index_path = FAISS_CONFIG["FAISS_INDEX_PATH"]
        config_path = FAISS_CONFIG["FAISS_CONFIG_PATH"]
        index_path.parent.mkdir(parents=True, exist_ok=True)
        self.document_store.update_embeddings(self.retriever)
        self.document_store.save(
            index_path=index_path,
            config_path=config_path
        )

    def run(
            self,
            query: Optional[str] = None,
            file_paths: Optional[List[str]] = None,
            labels: Optional[MultiLabel] = None,
            documents: Optional[List[Document]] = None,
            meta: Optional[dict] = None
    ) -> Tuple[Dict[str, List[Document]], str]:

        self.document_store.write_documents(documents)
        self.document_store.update_embeddings(self.retriever)

        return {"documents": documents}, "output_1"

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

    def get_retriever(self) -> EmbeddingRetriever:
        return self.retriever
