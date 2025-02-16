from qdrant_client import QdrantClient
from qdrant_client.http import models
from haystack.document_stores import InMemoryDocumentStore
from haystack.schema import Document
import numpy as np

class VectorStore:
    def __init__(self):
        self.client = QdrantClient(":memory:")
        self.collection_name = "documents"

        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=768,  # matches the embedding model's output size
                distance=models.Distance.COSINE
            )
        )

    def add_documents(self, documents, embeddings):
        points = []
        for doc, embedding in zip(documents, embeddings):
            points.append(
                models.PointStruct(
                    id=hash(doc.content),
                    payload={"content": doc.content, "metadata": doc.meta},
                    vector=embedding.tolist()
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query_embedding, top_k=3):
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k
        )

        return [
            Document(
                content=hit.payload["content"],
                meta=hit.payload["metadata"],
                score=hit.score
            ) for hit in search_result
        ]