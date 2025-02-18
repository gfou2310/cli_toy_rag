from typing import Any, Dict, List, Optional, Union

from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import BaseComponent, EmbeddingRetriever
from haystack.schema import Document, MultiLabel

from src.config import FAISS_CONFIG, MODEL_CONFIG


class VectorStore(BaseComponent):
    outgoing_edges = 1

    def __init__(self):
        self.dimension=MODEL_CONFIG["EMBEDDING_DIMENSION"]
        super().__init__()
        self.document_store = FAISSDocumentStore(
            embedding_dim=self.dimension,
            similarity="dot_product",
            # faiss_index_path=FAISS_CONFIG["FAISS_INDEX_PATH"]

        )
        self.retriever = EmbeddingRetriever(
            document_store=self.document_store,
            embedding_model=MODEL_CONFIG["EMBEDDING_MODEL"]
        )

    def save(self):
        """Save the document store state"""
        index_path = FAISS_CONFIG["FAISS_INDEX_PATH"]
        config_path = FAISS_CONFIG["FAISS_CONFIG_PATH"]
        index_path.parent.mkdir(parents=True, exist_ok=True)
        self.document_store.update_embeddings(self.retriever)
        self.document_store.save(
            index_path=str(index_path),
            config_path=str(config_path)
        )

    @classmethod
    def load(cls):
        """Load from saved state"""
        instance = cls()
        if FAISS_CONFIG["FAISS_INDEX_PATH"].exists() and FAISS_CONFIG["FAISS_CONFIG_PATH"].exists():
            instance.document_store = FAISSDocumentStore.load(
                index_path=str(FAISS_CONFIG["FAISS_INDEX_PATH"]),
                config_path=str(FAISS_CONFIG["FAISS_CONFIG_PATH"])
            )
            instance.retriever = EmbeddingRetriever(
                document_store=instance.document_store,
                embedding_model=MODEL_CONFIG["EMBEDDING_MODEL"]
            )
        return instance

    def run(
            self,
            query: Optional[str] = None,
            file_paths: Optional[List[str]] = None,
            labels: Optional[MultiLabel] = None,
            documents: Optional[List[Document]] = None,
            meta: Optional[dict] = None
    ) -> Dict[str, List[Document]]:

        self.document_store.write_documents(documents)
        self.document_store.update_embeddings(self.retriever)

        return {"documents": documents}

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

    def get_retriever(self):
        return self.retriever
