import os
from typing import Dict, List, Union

from pydantic import BaseModel

from core.ai_clients.ai_client_abc import AIClientAbc
from core.ai_clients.ai_client_factory import AIClientFactory
from core.config.stack_properties import stack_properties
from core.models.config import AgentProperties
from core.models.core.query_classification_response import QueryClassificationResponse
from core.models.core.rag_response import ToolResponse
from core.tools.tool import Tool
from core.tools.tool_builder import ToolBuilder
from core.utilities.document_utils import file_to_docs
from core.utilities.logger import LOG


class AgentStack:

    def __init__(self):
        """
        Initialize placeholders for stack components.
        Actual initialization happens in `initialize_stack`.
        """
        LOG.debug('AgentStack :: Initializing stack placeholders')
        self.__stack_properties: AgentProperties | None = None
        self.__classifier: AIClientAbc | None = None
        self.__tool_roles: Dict[str, str] = {}
        self.__tools: Dict[str, Tool] = {}
        self.__initialized: bool = False

    def __check_initialized(self):
        """
        Ensure the stack is fully initialized before usage.
        """
        LOG.debug('AgentStack :: Checking initialization status')
        if not self.__initialized:
            LOG.error('AgentStack :: Initialization check failed')
            raise RuntimeError('AgentStack has not been initialized. Call `initialize_stack` first.')

    def __initialize_stack_properties(self):
        """
        Load stack properties from the configuration.
        """
        LOG.info('AgentStack :: Initializing stack properties')
        self.__stack_properties = stack_properties()
        LOG.debug(f'AgentStack :: Stack properties loaded: {self.__stack_properties}')

    def __initialize_classifier(self):
        """
        Initialize the AI classifier client.
        """
        LOG.info('AgentStack :: Initializing classifier')
        if self.__stack_properties:
            self.__classifier = AIClientFactory.get_ai_client(
                self.__stack_properties.classifier.ai_client, self.__stack_properties.classifier.kind)
            self.__classifier.initialize()
            LOG.debug(f'AgentStack :: Classifier initialized: {self.__classifier}')

    def __initialize_tools(self):
        """
        Initialize tools and map their roles.
        """
        LOG.info('AgentStack :: Initializing tools')
        for tool_properties in self.__stack_properties.tools:
            LOG.debug(f'AgentStack :: Initializing tool: {tool_properties.name}')
            self.__tool_roles[tool_properties.name] = tool_properties.role
            self.__tools[tool_properties.name] = ToolBuilder.build_tool(
                tool_properties=tool_properties,
                vector_db_properties=self.__stack_properties.vector_db
            )
        LOG.debug(f'AgentStack :: Tools initialized: {self.__tools.keys()}')

    def initialize_stack(self):
        """
        Initialize the entire stack, including properties, classifier, and tools.
        """
        LOG.info('AgentStack :: Starting stack initialization')
        self.__initialize_stack_properties()
        self.__initialize_classifier()
        self.__initialize_tools()
        self.__initialized = True
        LOG.info('AgentStack :: Stack initialization completed')

    def classify_query(
            self,
            query: Union[str, Dict, List, BaseModel],
            classification_type: str = 'default'
    ) -> Union[str, QueryClassificationResponse]:
        """
        Classify the input query into one of the tool roles.

        Args:
            query: Input query to classify. Can be string, dictionary, list, or Pydantic model.
            classification_type: Specifies the classification approach.

        Returns:
            Classified tool name or detailed classification response.
        """
        LOG.info('AgentStack :: Classifying query')
        LOG.debug(f'AgentStack :: Query: **** , Type: {classification_type}')
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
        role_definitions = '\n'.join(
            [f'{name}: {role}' for name, role in self.__tool_roles.items()]
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
            LOG.debug('AgentStack :: Classification prompt (detailed): ****')
            detailed_response = self.__classifier.generate_answer(query=prompt)
            LOG.debug('AgentStack :: Detailed classification response: ****')

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
                LOG.warning('AgentStack :: Default classification fallback')
                return self.classify_query(query, 'default')

        prompt = f"""
           Classify the following input into one of the categories: {', '.join(self.__tool_roles.values())}.

           Category definitions: 
           {role_definitions}

           Input: "{normalized_query}"

           Please return only the category name, without any extra text or prefix.
           """
        LOG.debug('AgentStack :: Classification prompt (default): ****')
        response = self.__classifier.generate_answer(query=prompt)
        LOG.debug('AgentStack :: Classification result: ****')
        return response

    def process_query(self, query: str) -> ToolResponse:
        """
        Process the input query, classify it, and use the appropriate tool.
        """
        LOG.info('AgentStack :: Processing query: ****')
        self.__check_initialized()

        tool_name = self.classify_query(query)
        LOG.debug(f'AgentStack :: Classified tool: {tool_name}')
        if tool_name not in self.__tools:
            LOG.error(f'AgentStack :: No tool found for role: {tool_name}')
            raise ValueError(f'No tool found for the classified role \'{tool_name}\'.')

        response = self.__tools[tool_name].process_query(query)
        LOG.debug('AgentStack :: Query processed. Response: ****')
        return response

    def store_documents(self, files: List[str], classify_document: bool = True) -> Dict[str, List[str]]:
        """
        Store multiple documents in the appropriate vector database tool.

        Args:
            files (List[str]): List of file paths to store
            classify_document (bool): Whether to classify each document before storing

        Returns:
            Dict[str, List[str]]: Dictionary of tool names and their stored document IDs
        """
        LOG.info('AgentStack :: Storing documents')
        LOG.debug(f'AgentStack :: Files to store: {files}')
        self.__check_initialized()

        stored_documents = {}
        for file in files:
            try:
                LOG.debug(f'AgentStack :: Processing file: {file}')
                chunks, metadata_list = file_to_docs(file)

                if classify_document:
                    classify_input = ' '.join(chunks[:3]) if len(chunks) > 3 else ' '.join(chunks)
                    if not classify_input:
                        classify_input = os.path.basename(file)
                    tool_name = self.classify_query(classify_input)
                    LOG.debug(f'AgentStack :: Classified tool for file: {tool_name}')
                else:
                    tool_name = list(self.__tools.keys())[0] if self.__tools else None

                if not tool_name or tool_name not in self.__tools:
                    LOG.error(f'AgentStack :: No tool found for document: {file}')
                    raise ValueError(f'No tool found for document: {file}')

                tool = self.__tools[tool_name]
                document_ids = tool.store_docs(
                    documents=chunks,
                    metadata_list=metadata_list
                )
                if tool_name not in stored_documents:
                    stored_documents[tool_name] = []
                stored_documents[tool_name].extend(document_ids)
                LOG.debug('AgentStack :: Stored document IDs: ****')

            except Exception as e:
                LOG.error(f'Error processing file {file}')
                raise e

        LOG.info('AgentStack :: Document storage completed')
        LOG.debug('AgentStack :: Stored documents: ****')
        return stored_documents
