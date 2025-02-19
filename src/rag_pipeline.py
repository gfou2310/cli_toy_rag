from typing import Optional

from haystack import Pipeline
from haystack.nodes import JoinDocuments, PromptNode

from src.config import CHAT_MODEL_CONFIG, RETRIEVER_CONFIG
from src.prompt_template import prompt_template
from src.vector_store import VectorStore


class RAGPipeline:
    def __init__(self, vector_store: VectorStore):
        self.retriever = vector_store.get_retriever()
        self.retriever.top_k = RETRIEVER_CONFIG["TOP_K"]
        self.join_documents = JoinDocuments()

        self.generator = PromptNode(
            max_length=CHAT_MODEL_CONFIG["MAX_LENGTH"],
            model_name_or_path=CHAT_MODEL_CONFIG["MODEL"],
            default_prompt_template=prompt_template,
            truncate=False,
            model_kwargs={
                "temperature": CHAT_MODEL_CONFIG["TEMPERATURE"],
                "top_p": CHAT_MODEL_CONFIG["TOP_P"],
                "frequency_penalty": CHAT_MODEL_CONFIG["FREQUENCY_PENALTY"],
                "presence_penalty": CHAT_MODEL_CONFIG["PRESENCE_PENALTY"]
            }
        )

        self.pipeline = Pipeline()
        self.pipeline.add_node(component=self.retriever, name="Retriever", inputs=["Query"])
        self.pipeline.add_node(component=self.join_documents, name="JoinDocuments", inputs=["Retriever"])
        self.pipeline.add_node(component=self.generator, name="Generator", inputs=["JoinDocuments"])

    def run(self, query: str) -> Optional[dict]:
        return self.pipeline.run(query=query)