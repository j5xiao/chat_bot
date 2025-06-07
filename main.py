import requests
from typing import List
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.base.embeddings.base import BaseEmbedding

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_TOKEN = "hf_..." ### please your API here
HF_API_TOKEN = input("Please your API: ")
print("Your API is {}".format(HF_API_TOKEN))

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}

class HuggingFaceEmbeddingAPI(BaseEmbedding):
    def _get_text_embedding(self, text: str) -> List[float]:
        url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2?wait_for_model=true"
        payload = {
            "inputs": {
                "source_sentence": text,
                "sentences": [text]
            }
        }
        response = requests.post(url, headers=HEADERS, json=payload)
        response.raise_for_status()
        embeddings = response.json()
        return embeddings

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embedding = self._get_text_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

def main():
    documents = SimpleDirectoryReader("data").load_data()
    print(f"Loaded {len(documents)} documents.")

    embeddings = HuggingFaceEmbeddingAPI()

    index = VectorStoreIndex.from_documents(documents, embed_model=embeddings)
    print("Vector index built successfully.")

    from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

    llm = HuggingFaceInferenceAPI(
        model_name="distilgpt2",  
        api_key=HF_API_TOKEN,
        task="text-generation", 
        max_new_tokens=256,
    )

    from llama_index.core import Settings
    Settings.llm = llm

    query_engine = index.as_query_engine(embed_model=embeddings, llm=None)

    while True:
        query = input("\nEnter your question (or type 'exit' to quit): ")
        if query.strip().lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        response = query_engine.query(query)
        print(f"\nAnswer: {response.response}")

if __name__ == "__main__":
    main()
