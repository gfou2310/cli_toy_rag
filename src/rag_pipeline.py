from haystack import Pipeline


class RetrievalPipeline:
    def __init__(self, vector_store):
        # Get the retriever from vector store
        self.retriever = vector_store.get_retriever()

        # Create retrieval pipeline
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=self.retriever, name="Retriever", inputs=["Query"])

    def retrieve(self, query: str, top_k: int = 5) -> dict:
        try:
            # Run the pipeline
            result = self.pipeline.run(
                query=query,
                params={
                    "Retriever": {"top_k": top_k}
                }
            )

            return {
                "documents": result["documents"],
                "success": True
            }

        except Exception as e:
            return {
                "documents": None,
                "success": False,
                "error": str(e)
            }
