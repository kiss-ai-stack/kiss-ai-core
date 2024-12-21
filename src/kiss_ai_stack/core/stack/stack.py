import os
from typing import Dict, List, Union, Optional, Any

from kiss_ai_stack.core.ai_clients.ai_client_abc import AIClientAbc
from kiss_ai_stack.core.ai_clients.ai_client_factory import AIClientFactory
from kiss_ai_stack.core.config.stack_properties import stack_properties
from kiss_ai_stack.core.models.config.stack_props import StackProperties
from kiss_ai_stack.core.models.core.query_classification_response import QueryClassificationResponse
from kiss_ai_stack.core.models.core.rag_response import ToolResponse
from kiss_ai_stack.core.models.enums.tool_kind import ToolKind
from kiss_ai_stack.core.tools.tool import Tool
from kiss_ai_stack.core.tools.tool_builder import ToolBuilder
from kiss_ai_stack.core.utilities.document_utils import file_to_docs
from kiss_ai_stack.core.utilities.logger import LOG
from pydantic import BaseModel


class Stack:

    def __init__(self, stack_id, temporary: Optional[bool] = True):
        """
        Initialize a stack with placeholders for stack components.

        This method sets up the stack with essential properties such as stack ID, stack properties, classifier,
        tool roles, and tools. The actual initialization happens later when the `initialize_stack` method is called.

        :param stack_id: The unique identifier for the stack.
        :param temporary: Flag to indicate if the stack session is temporary, affecting docs' persistence.
        """
        LOG.debug(f'Stack-{stack_id} :: ready!')
        self.__stack_id = stack_id
        self.__stack_properties: StackProperties | None = None
        self.__decision_maker: AIClientAbc | None = None
        self.__tool_roles: Dict[str, str] = {}
        self.__tools: Dict[str, Tool] = {}
        self.__temporary_stack = temporary
        self.__initialized: bool = False

    def __check_initialized(self):
        """
        Ensure the stack is fully initialized before usage.
        """
        LOG.debug(f'Stack-{self.__stack_id} :: Checking initialization status')
        if not self.__initialized:
            LOG.error(f'Stack-{self.__stack_id} :: Initialization check failed')
            raise RuntimeError('Stack has not been initialized.')

    async def __initialize_stack_properties(self):
        """
        Load stack properties from the configuration.
        """
        LOG.info(f'Stack-{self.__stack_id} :: Initializing stack properties')
        self.__stack_properties = await stack_properties()
        LOG.debug(f'Stack-{self.__stack_id} :: Stack properties loaded')

    def __initialize_decision_maker(self):
        """
        Initialize the AI based decision_maker client.
        """
        LOG.info(f'Stack-{self.__stack_id} :: Initializing decision maker')
        if self.__stack_properties:
            self.__decision_maker = AIClientFactory.get_ai_client(
                self.__stack_properties.decision_maker.ai_client, self.__stack_properties.decision_maker.kind)
            self.__decision_maker.initialize()
            LOG.debug(f'StackStack :: Decision maker initialized: {self.__decision_maker}')

    async def __initialize_tools(self):
        """
        Initialize tools and map their roles.
        """
        LOG.info(f'Stack-{self.__stack_id} :: Initializing tools')
        for tool_properties in self.__stack_properties.tools:
            LOG.debug(f'Stack-{self.__stack_id} :: Initializing tool: {tool_properties.name}')
            self.__tool_roles[tool_properties.name] = tool_properties.role
            self.__tools[tool_properties.name] = await ToolBuilder.build_tool(
                stack_id=self.__stack_id,
                tool_properties=tool_properties,
                vector_db_properties=self.__stack_properties.vector_db,
                temporary_stack=self.__temporary_stack
            )
        LOG.debug(f'Stack-{self.__stack_id} :: Tools initialized')

    async def initialize_stack(self):
        LOG.info(f'Stack-{self.__stack_id} :: Starting initialization')
        if not self.__initialized:
            await self.__initialize_stack_properties()
            self.__initialize_decision_maker()
            await self.__initialize_tools()
            self.__initialized = True
            LOG.info(f'Stack-{self.__stack_id} :: initialization completed')
        else:
            LOG.warning(f'Stack-{self.__stack_id} :: has been already initialized')

    async def classify_query(
            self,
            query: Union[str, Dict, List, BaseModel],
            rag: bool = False,
            classification_type: str = 'default'
    ) -> Union[str, QueryClassificationResponse]:
        """
        Classify the input query into one of the tool roles.

        :param query: Input query to classify. Can be string, dictionary, list, or Pydantic model.
        :param rag: If True, only consider RAG-type tools for classification.
        :param classification_type: Specifies the classification approach.

        :returns: Classified tool name or detailed classification response.
        """
        LOG.info(f'Stack-{self.__stack_id} :: Classifying query')
        LOG.debug(f'Stack-{self.__stack_id} :: Query: **** , Type: {classification_type}')
        self.__check_initialized()

        def normalize_input(input_data):
            if isinstance(input_data, str):
                return input_data
            elif isinstance(input_data, dict):
                return ' '.join(f'{k}: {v}' for k, v in input_data.items())
            elif isinstance(input_data, list):
                return ' '.join(str(item) for item in input_data)
            elif hasattr(input_data, 'dict'):
                return ' '.join(f'{k}: {v}' for k, v in input_data.dict().items())
            else:
                return str(input_data)

        normalized_query = normalize_input(query)

        filtered_tool_roles = {}
        if rag:
            filtered_tool_roles = {
                name: role for name, role in self.__tool_roles.items()
                if self.__tools[name].tool_kind() == ToolKind.RAG
            }
        else:
            filtered_tool_roles = self.__tool_roles

        if not filtered_tool_roles:
            LOG.error(f'Stack-{self.__stack_id} :: No tools available after RAG filtering')
            raise ValueError('No tools available for query classification')

        role_definitions = '\n'.join(
            [f'{name}: {role}' for name, role in filtered_tool_roles.items()]
        )

        if classification_type == 'detailed':
            prompt = f"""
               Carefully classify the following input into one of the tool categories.

               Available Categories: {', '.join(self.__tool_roles.values())}

               Category Definitions: 
               {role_definitions}

               Input: "{normalized_query}"

               Provide your response in the following format:
               - tool_name: [Selected tool name]
               - confidence: [Confidence score from 0.0 to 1.0]
               - reasoning: [Brief explanation of classification]
               """
            LOG.debug(f'Stack-{self.__stack_id} :: Classification prompt (detailed): ****')
            detailed_response = await self.__decision_maker.generate_answer(query=prompt)
            LOG.debug(f'Stack-{self.__stack_id} :: Detailed classification response: ****')

            try:
                response_lines = detailed_response.split('\n')
                tool_name = response_lines[0].split(':')[1].strip()
                confidence = float(response_lines[1].split(':')[1].strip())
                reasoning = response_lines[2].split(':')[1].strip()

                return QueryClassificationResponse(
                    tool_name=tool_name,
                    confidence=confidence,
                    reasoning=reasoning
                )
            except Exception:
                LOG.warning(f'Stack-{self.__stack_id} :: Default classification fallback')
                return await self.classify_query(query, rag=rag, classification_type='default')

        prompt = f"""
           Classify the following input into one of the categories: {', '.join(self.__tool_roles.values())}.

           Category definitions: 
           {role_definitions}

           Input: "{normalized_query}"

           Please return only the category name, without any extra text or prefix.
           """
        LOG.debug(f'Stack-{self.__stack_id} :: Classification prompt (default): ****')
        response = await self.__decision_maker.generate_answer(query=prompt)
        LOG.debug(f'Stack-{self.__stack_id} :: Classification result: ****')
        return response

    async def process_query(self, query: str) -> ToolResponse:
        """
        Process the input query, classify it, and use the appropriate tool.

        :param query: User prompt or query
        :returns: Generated answer
        """
        LOG.info(f'Stack-{self.__stack_id} :: Processing query: ****')
        self.__check_initialized()

        tool_name = await self.classify_query(query)
        LOG.debug(f'Stack-{self.__stack_id} :: Classified tool: {tool_name}')
        if tool_name not in self.__tools:
            LOG.error(f'Stack-{self.__stack_id} :: No tool found for role: {tool_name}')
            raise ValueError(f'No tool found for the classified role \'{tool_name}\'.')

        response = await self.__tools[tool_name].process_query(query=query)
        LOG.debug(f'Stack-{self.__stack_id} :: Query processed. Response: ****')
        return response

    async def store_documents(
            self,
            files: List[str],
            metadata: Optional[Dict[str, Any]] = None,
            classify_document: bool = True
    ) -> Dict[str, Union[List[str], str]]:
        """
        Store multiple documents in the appropriate vector database tool.

        :param files: (List[str]): List of file paths to store
        :param metadata: (Optional[Dict[str, Any]]): Optional metadata to associate with documents
        :param classify_document: (bool): Whether to classify each document before storing

        :returns Dict[str, Union[List[str], str]]: Dictionary containing stored document IDs and optional query response
        """
        LOG.info(f'Stack-{self.__stack_id} :: Storing documents')
        LOG.debug(f'Stack-{self.__stack_id} :: Files to store: {files}')

        self.__check_initialized()
        stored_documents = {}
        rag_tool_names = [
            name for name in self.__tool_roles.keys()
            if self.__tools[name].tool_kind() == ToolKind.RAG
        ]
        if len(rag_tool_names) == 0:
            LOG.error(f'Stack-{self.__stack_id} :: No tools available after RAG filtering')
            raise ValueError('No tools available for query classification')

        for file in files:
            try:
                LOG.debug(f'Stack-{self.__stack_id} :: Processing file: {file}')
                chunks, metadata_list = await file_to_docs(file)

                if metadata:
                    metadata_list = [
                        {**meta, **metadata} for meta in metadata_list
                    ]

                if classify_document and len(rag_tool_names) > 1:
                    classify_input = ' '.join(chunks[:3]) if len(chunks) > 3 else ' '.join(chunks)
                    if not classify_input:
                        classify_input = os.path.basename(file)
                    tool_name = await self.classify_query(classify_input, True)
                    LOG.debug(f'Stack-{self.__stack_id} :: Classified tool for file: {tool_name}')
                else:
                    tool_name = rag_tool_names[0]

                if not tool_name or tool_name not in self.__tools:
                    LOG.error(f'Stack-{self.__stack_id} :: No tool found for document: {file}')
                    raise ValueError(f'No tool found for document: {file}')

                tool = self.__tools[tool_name]
                document_ids = await tool.store_docs(
                    documents=chunks,
                    metadata_list=metadata_list
                )
                if tool_name not in stored_documents:
                    stored_documents[tool_name] = []
                stored_documents[tool_name].extend(document_ids)
                LOG.debug(f'Stack-{self.__stack_id} :: Stored document IDs: ****')

            except Exception as e:
                LOG.error(f'Stack-{self.__stack_id} :: Error processing file {file}')
                raise e

        LOG.info(f'Stack-{self.__stack_id} :: Document storage completed')
        LOG.debug(f'Stack-{self.__stack_id} :: Stored documents: ****')
        return stored_documents

    async def destroy_stack(self, cleanup: bool = False):
        """
        Destroy and clean up the stack's stack components.

        :param cleanup: Clean stored docs, preferably for temporary sessions.
        """
        LOG.info(f'Stack-{self.__stack_id} :: Starting destruction')

        for tool_name, tool in self.__tools.items():
            try:
                LOG.debug(f'Stack-{self.__stack_id} :: Destroying tool: {tool_name}')
                await tool.destroy(cleanup)
            except Exception as e:
                LOG.warning(f'Stack-{self.__stack_id} :: Error destroying tool {tool_name}: {str(e)}')
        if self.__decision_maker:
            try:
                LOG.debug(f'Stack-{self.__stack_id} :: Destroying decision_maker')
                await self.__decision_maker.destroy()
            except Exception as e:
                LOG.warning(f'Stack-{self.__stack_id} :: Error occurred while destroying decision_maker: {str(e)}')
        self.__stack_properties = None
        self.__decision_maker = None
        self.__tool_roles.clear()
        self.__tools.clear()
        self.__initialized = False

        LOG.info(f'Stack-{self.__stack_id} :: destruction completed')
