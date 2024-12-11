import os
from typing import Dict, List, Union

from pydantic import BaseModel

from kiss_ai_stack.core.ai_clients.ai_client_abc import AIClientAbc
from kiss_ai_stack.core.ai_clients.ai_client_factory import AIClientFactory
from kiss_ai_stack.core.config.stack_properties import stack_properties
from kiss_ai_stack.core.models.config.agent import AgentProperties
from kiss_ai_stack.core.models.core.query_classification_response import QueryClassificationResponse
from kiss_ai_stack.core.models.core.rag_response import ToolResponse
from kiss_ai_stack.core.tools.tool import Tool
from kiss_ai_stack.core.tools.tool_builder import ToolBuilder
from kiss_ai_stack.core.utilities.document_utils import file_to_docs
from kiss_ai_stack.core.utilities.logger import LOG


class Agent:

    def __init__(self, agent_id):
        """
        Initialize placeholders for stack components.
        Actual initialization happens in `initialize_stack`.
        """
        LOG.debug(f'Agent-{agent_id} :: agent ready!')
        self.__agent_id = agent_id
        self.__stack_properties: AgentProperties | None = None
        self.__classifier: AIClientAbc | None = None
        self.__tool_roles: Dict[str, str] = {}
        self.__tools: Dict[str, Tool] = {}
        self.__initialized: bool = False

    def __check_initialized(self):
        """
        Ensure the stack is fully initialized before usage.
        """
        LOG.debug(f'Agent-{self.__agent_id} :: Checking initialization status')
        if not self.__initialized:
            LOG.error(f'Agent-{self.__agent_id} :: Initialization check failed')
            raise RuntimeError('Agent has not been initialized.')

    def __initialize_stack_properties(self):
        """
        Load stack properties from the configuration.
        """
        LOG.info(f'Agent-{self.__agent_id} :: Initializing stack properties')
        self.__stack_properties = stack_properties()
        LOG.debug(f'Agent-{self.__agent_id} :: Stack properties loaded')

    def __initialize_classifier(self):
        """
        Initialize the AI classifier client.
        """
        LOG.info(f'Agent-{self.__agent_id} :: Initializing classifier')
        if self.__stack_properties:
            self.__classifier = AIClientFactory.get_ai_client(
                self.__stack_properties.classifier.ai_client, self.__stack_properties.classifier.kind)
            self.__classifier.initialize()
            LOG.debug(f'AgentStack :: Classifier initialized: {self.__classifier}')

    def __initialize_tools(self):
        """
        Initialize tools and map their roles.
        """
        LOG.info(f'Agent-{self.__agent_id} :: Initializing tools')
        for tool_properties in self.__stack_properties.tools:
            LOG.debug(f'Agent-{self.__agent_id} :: Initializing tool: {tool_properties.name}')
            self.__tool_roles[tool_properties.name] = tool_properties.role
            self.__tools[tool_properties.name] = ToolBuilder.build_tool(
                agent_id=self.__agent_id,
                tool_properties=tool_properties,
                vector_db_properties=self.__stack_properties.vector_db
            )
        LOG.debug(f'Agent-{self.__agent_id} :: Tools initialized')

    def initialize_stack(self):
        """
        Initialize the entire stack, including properties, classifier, and tools.
        """
        LOG.info(f'Agent-{self.__agent_id} :: Starting initialization')
        if not self.__initialized:
            self.__initialize_stack_properties()
            self.__initialize_classifier()
            self.__initialize_tools()
            self.__initialized = True
            LOG.info(f'Agent-{self.__agent_id} :: initialization completed')
        else:
            LOG.warning(f'Agent-{self.__agent_id} :: has been already initialized')

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
        LOG.info(f'Agent-{self.__agent_id} :: Classifying query')
        LOG.debug(f'Agent-{self.__agent_id} :: Query: **** , Type: {classification_type}')
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
            LOG.debug(f'Agent-{self.__agent_id} :: Classification prompt (detailed): ****')
            detailed_response = self.__classifier.generate_answer(query=prompt)
            LOG.debug(f'Agent-{self.__agent_id} :: Detailed classification response: ****')

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
                LOG.warning(f'Agent-{self.__agent_id} :: Default classification fallback')
                return self.classify_query(query, 'default')

        prompt = f"""
           Classify the following input into one of the categories: {', '.join(self.__tool_roles.values())}.

           Category definitions: 
           {role_definitions}

           Input: "{normalized_query}"

           Please return only the category name, without any extra text or prefix.
           """
        LOG.debug(f'Agent-{self.__agent_id} :: Classification prompt (default): ****')
        response = self.__classifier.generate_answer(query=prompt)
        LOG.debug(f'Agent-{self.__agent_id} :: Classification result: ****')
        return response

    def process_query(self, query: str) -> ToolResponse:
        """
        Process the input query, classify it, and use the appropriate tool.
        """
        LOG.info(f'Agent-{self.__agent_id} :: Processing query: ****')
        self.__check_initialized()

        tool_name = self.classify_query(query)
        LOG.debug(f'Agent-{self.__agent_id} :: Classified tool: {tool_name}')
        if tool_name not in self.__tools:
            LOG.error(f'Agent-{self.__agent_id} :: No tool found for role: {tool_name}')
            raise ValueError(f'No tool found for the classified role \'{tool_name}\'.')

        response = self.__tools[tool_name].process_query(query)
        LOG.debug(f'Agent-{self.__agent_id} :: Query processed. Response: ****')
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
        LOG.info('Agent-{self.__agent_id} :: Storing documents')
        LOG.debug(f'Agent-{self.__agent_id} :: Files to store: {files}')
        self.__check_initialized()

        stored_documents = {}
        for file in files:
            try:
                LOG.debug(f'Agent-{self.__agent_id} :: Processing file: {file}')
                chunks, metadata_list = file_to_docs(file)

                if classify_document:
                    classify_input = ' '.join(chunks[:3]) if len(chunks) > 3 else ' '.join(chunks)
                    if not classify_input:
                        classify_input = os.path.basename(file)
                    tool_name = self.classify_query(classify_input)
                    LOG.debug(f'Agent-{self.__agent_id} :: Classified tool for file: {tool_name}')
                else:
                    tool_name = list(self.__tools.keys())[0] if self.__tools else None

                if not tool_name or tool_name not in self.__tools:
                    LOG.error(f'Agent-{self.__agent_id} :: No tool found for document: {file}')
                    raise ValueError(f'No tool found for document: {file}')

                tool = self.__tools[tool_name]
                document_ids = tool.store_docs(
                    documents=chunks,
                    metadata_list=metadata_list
                )
                if tool_name not in stored_documents:
                    stored_documents[tool_name] = []
                stored_documents[tool_name].extend(document_ids)
                LOG.debug(f'Agent-{self.__agent_id} :: Stored document IDs: ****')

            except Exception as e:
                LOG.error(f'Error processing file {file}')
                raise e

        LOG.info(f'Agent-{self.__agent_id} :: Document storage completed')
        LOG.debug(f'Agent-{self.__agent_id} :: Stored documents: ****')
        return stored_documents

    def destroy_stack(self, cleanup: bool = False):
        """
        Destroy and clean up the agent's stack components.
        """
        LOG.info(f'Agent-{self.__agent_id} :: Starting destruction')

        for tool_name, tool in self.__tools.items():
            try:
                LOG.debug(f'Agent-{self.__agent_id} :: Destroying tool: {tool_name}')
                tool.destroy(cleanup)
            except Exception as e:
                LOG.warning(f'Agent-{self.__agent_id} :: Error destroying tool {tool_name}: {str(e)}')
        if self.__classifier:
            try:
                LOG.debug(f'Agent-{self.__agent_id} :: Destroying classifier')
                self.__classifier.destroy()
            except Exception as e:
                LOG.warning(f'Agent-{self.__agent_id} :: Error destroying classifier: {str(e)}')
        self.__stack_properties = None
        self.__classifier = None
        self.__tool_roles.clear()
        self.__tools.clear()
        self.__initialized = False

        LOG.info(f'Agent-{self.__agent_id} :: destruction completed')
