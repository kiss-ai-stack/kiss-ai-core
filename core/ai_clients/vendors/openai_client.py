from typing import List

from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction
from openai import OpenAI

from core.ai_clients.ai_client_abc import AIClientAbc
from core.models.config import AIClientProperties


class OpenAIClient(AIClientAbc):

    def __init__(self, properties: AIClientProperties):
        self.__properties = properties
        self.__client = None

    def embedding_function(self, embedding_model):
        return OpenAIEmbeddingFunction(
            api_key=self.__properties.api_key,
            model_name=embedding_model
        )

    def instance(self):
        return self.__client

    def initialize(self):
        self.__client = OpenAI(
            api_key=self.__properties.api_key
        )

    def generate_answer(self, query: str, chunks: List[str]) -> str:
        context = '\n\n'.join(chunks)
        prompt = f'''Given the following context, answer the question.
        If the answer cannot be found in the context, say so.
        
        Context:
        {context}
        
        Question:
        {query}
        
        Answer:'''

        response = self.__client.chat.completions.create(
            model=self.__properties.model,
            messages=[
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant that answers questions based on the provided context.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
