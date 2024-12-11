from typing import List, Optional

from kiss_ai_stack.core.models.config.ai_client import AIClientProperties
from kiss_ai_stack.core.models.enums.tool_kind import ToolKind

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    raise ImportError('OpenAI is not installed. Please install \'openai\' python package `pip install openai`.')

from kiss_ai_stack.core.ai_clients.ai_client_abc import AIClientAbc
from kiss_ai_stack.core.utilities.logger import LOG


class OpenAIClient(AIClientAbc):
    """
    Implementation of AIClientAbc for OpenAI.

    This class integrates with OpenAI to provide functionalities such as embeddings and
    prompt-based responses.
    """

    def __init__(self, properties: AIClientProperties, tool_kind: ToolKind = ToolKind.PROMPT):
        """
        Initialize the OpenAI client.

        Args:
            properties (AIClientProperties): Configuration properties for OpenAI.
            tool_kind (ToolKind): The type of tool (e.g., PROMPT or RAG). Defaults to ToolKind.PROMPT.
        """
        self.__tool_kind = tool_kind
        self.__properties = properties
        self.__client = None
        LOG.info(f'OpenAIClient :: initialized with tool kind: {tool_kind}')

    def instance(self):
        """
        Get the raw OpenAI client instance.

        Returns:
            OpenAI: The OpenAI client instance.
        """
        return self.__client

    def initialize(self):
        """
        Initialize the OpenAI client by setting up the API key.
        """
        self.__client = OpenAI(
            api_key=self.__properties.api_key
        )
        LOG.info('OpenAIClient :: client initialized successfully')

    def generate_answer(self, query: str, chunks: List[str] = None, temperature: Optional[float] = 0.7) -> str:
        """
        Generate an answer for the given query.

        Args:
            query (str): The input query to process.
            chunks (List[str], optional): Contextual chunks for RAG-style processing.
            temperature (float, optional): Controls response randomness. Defaults to 0.7.

        Returns:
            str: The AI-generated answer.
        """
        LOG.info('OpenAIClient :: generating answer for query: ****')
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
            error_message = 'Unknown tool kind!'
            LOG.error(f'OpenAIClient :: {error_message}')
            return error_message

        LOG.debug('OpenAIClient :: constructed prompt: ****')

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

        answer = response.choices[0].message.content
        LOG.info('OpenAIClient :: generated answer: ****')
        return answer

    def destroy(self):
        if hasattr(self.__client, 'destroy'):
            self.__client.close()
            LOG.info('OpenAIClient :: Closed')
