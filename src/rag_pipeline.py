from transformers import AutoTokenizer, AutoModelForCausalLM
from src.config import CONFIG

class RAGPipeline:
    def __init__(self, vector_store, embedding_model):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

        # Initialize LLM
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG["LLM_MODEL"])
        self.llm = AutoModelForCausalLM.from_pretrained(CONFIG["LLM_MODEL"])

    def generate_response(self, query):
        # Get query embedding and search for relevant documents
        query_embedding = self.embedding_model.embed([query])[0]
        relevant_docs = self.vector_store.search(query_embedding, top_k=CONFIG["TOP_K"])

        # Create prompt with context
        context = "\n".join([doc.content for doc in relevant_docs])
        prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Answer:"""

        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.llm.generate(
            inputs.input_ids,
            max_length=2048,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Answer:")[-1].strip()