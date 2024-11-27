from typing import List, Optional

from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction
from openai import OpenAI

from core.ai_clients.ai_client_abc import AIClientAbc
from core.models.config import AIClientProperties
from core.models.enums import ToolKind


class OpenAIClient(AIClientAbc):

    def __init__(self, properties: AIClientProperties, tool_kind: ToolKind = ToolKind.PROMPT):
        self.__tool_kind = tool_kind
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

    def generate_answer(self, query: str, chunks: List[str] = None, temperature: Optional[float] = 0.7) -> str:
        prompt = ''
        base_content = ''

        if self.__tool_kind == ToolKind.RAG:
            if chunks is None:
                chunks = []
            context = '\n\n'.join(chunks)
            base_content = 'You are a helpful assistant that answers questions based on the provided context.'
            prompt = f'''Given the following context, answer the question.
            If the answer cannot be found in the context, say so.

            Context:
            {context}

            Question:
            {query}
            
            Answer:'''
        elif self.__tool_kind == ToolKind.PROMPT:
            base_content = 'You are a helpful assistant that responds to any given prompt.'
            prompt = query
        else:
            return 'Unknown tool kind!'

        response = self.__client.chat.completions.create(
            model=self.__properties.model,
            messages=[
                {
                    'role': 'system',
                    'content': base_content
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            temperature=temperature
        )
        return response.choices[0].message.content
