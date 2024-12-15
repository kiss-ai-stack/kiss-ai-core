from typing import List, Optional

from kiss_ai_stack.core.ai_clients.ai_client_abc import AIClientAbc
from kiss_ai_stack.core.config import AI_CLIENT
from kiss_ai_stack.core.models.config.ai_client import AIClientProperties
from kiss_ai_stack.core.models.enums.ai_client_vendor import AIClientVendor
from kiss_ai_stack.core.models.enums.tool_kind import ToolKind
from kiss_ai_stack.core.utilities import install_package
from kiss_ai_stack.core.utilities.logger import LOG


class OpenAIClient(AIClientAbc):
    """
    Implementation of AIClientAbc for OpenAI.

    This class integrates with OpenAI to provide functionalities such as embeddings and
    prompt-based responses asynchronously.
    """

    def __init__(self, properties: AIClientProperties, tool_kind: ToolKind = ToolKind.PROMPT):
        """
        Initialize the OpenAI client.

        :param properties: Configuration properties for OpenAI.
        :param tool_kind: The type of tool (e.g., PROMPT or RAG). Defaults to ToolKind.PROMPT.
        """
        self.__tool_kind = tool_kind
        self.__properties = properties
        self.__client: Optional['AsyncOpenAI'] = None
        LOG.info(f'OpenAIClient :: initialized with tool kind: {tool_kind}')

    def instance(self):
        """
        Get the raw OpenAI client instance.

        :return: The OpenAI client instance.
        """
        return self.__client

    def initialize(self):
        """
        Initialize the OpenAI client by setting up the API key.

        This method dynamically imports AsyncOpenAI if it is not already installed.
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            package_name = AI_CLIENT[AIClientVendor.OPENAI]
            LOG.warning(f'OpenAI is not installed. Attempting to auto-install {package_name}.')
            install_package(package_name)
            from openai import AsyncOpenAI

        self.__client = AsyncOpenAI(api_key=self.__properties.api_key)
        LOG.info('OpenAIClient :: client initialized successfully')

    async def generate_answer(self, query: str, chunks: List[str] = None, temperature: Optional[float] = 0.7) -> str:
        """
        Generate an answer for the given query.

        :param query: The input query to process.
        :param chunks: Contextual chunks for RAG-style processing. Defaults to None.
        :param temperature: Controls response randomness. Defaults to 0.7.

        :return: The AI-generated answer as a string.
        """
        LOG.info('OpenAIClient :: generating answer for query: ****')
        prompt = ''
        base_content = ''

        if self.__tool_kind == ToolKind.RAG:
            chunks = chunks or []
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

        response = await self.__client.chat.completions.create(
            model=self.__properties.model,
            messages=[
                {'role': 'system', 'content': base_content},
                {'role': 'user', 'content': prompt}
            ],
            temperature=temperature
        )

        answer = response.choices[0].message.content
        LOG.info('OpenAIClient :: generated answer: ****')
        return answer

    async def destroy(self):
        """
        Close the OpenAI client if it has a close method.
        """
        if hasattr(self.__client, 'close'):
            await self.__client.close()
            LOG.info('OpenAIClient :: Closed')
