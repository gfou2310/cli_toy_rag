from typing import Optional

from haystack import Pipeline
from haystack.nodes import JoinDocuments, PromptNode

from src.config import CHAT_MODEL_CONFIG, RETRIEVER_CONFIG
from src.prompt_template import prompt_template
from src.vector_store import VectorStore


class RAGPipeline:
    def __init__(self):
        self.retriever = VectorStore().get_retriever()
        self.retriever.top_k = RETRIEVER_CONFIG["TOP_K"]
        self.join_documents = JoinDocuments()
        self.generator = PromptNode(
            max_length=500,
            model_name_or_path=CHAT_MODEL_CONFIG["MODEL"],
            default_prompt_template=prompt_template,
            truncate=False,
            model_kwargs={
                "temperature": 0.2,
                "top_p": 0.9,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.2,
            }
        )

        self.pipeline = Pipeline()
        self.pipeline.add_node(component=self.retriever, name="Retriever", inputs=["Query"])
        self.pipeline.add_node(component=self.join_documents, name="JoinDocuments", inputs=["Retriever"])
        self.pipeline.add_node(component=self.generator, name="Generator", inputs=["JoinDocuments"])

    def run(self, query: str) -> Optional[dict]:
        return self.pipeline.run(query=query)