from typing import Dict, List

from core.ai_clients.ai_client_abc import AIClientAbc
from core.ai_clients.ai_client_factory import AIClientFactory
from core.config.stack_properties import stack_properties
from core.models.config import AgentProperties
from core.tools.tool import Tool
from core.tools.tool_builder import ToolBuilder


class AgentStack:

    def __init__(self):
        """
        Initialize placeholders for stack components.
        Actual initialization happens in `initialize_stack`.
        """
        self.__stack_properties: AgentProperties | None = None
        self.__classifier: AIClientAbc | None = None
        self.__tool_roles: Dict[str, str] = {}
        self.__tools: Dict[str, Tool] = {}
        self.__initialized: bool = False

    def __check_initialized(self):
        """
        Ensure the stack is fully initialized before usage.
        """
        if not self.__initialized:
            raise RuntimeError("AgentStack has not been initialized. Call `initialize_stack` first.")

    def __initialize_stack_properties(self):
        """
        Load stack properties from the configuration.
        """
        self.__stack_properties = stack_properties()

    def __initialize_classifier(self):
        """
        Initialize the AI classifier client.
        """
        if self.__stack_properties:
            self.__classifier = AIClientFactory.get_ai_client(
                self.__stack_properties.classifier.ai_client, self.__stack_properties.classifier.kind)
            self.__classifier.initialize()

    def __initialize_tools(self):
        """
        Initialize tools and map their roles.
        """
        for tool_properties in self.__stack_properties.tools:
            self.__tool_roles[tool_properties.name] = tool_properties.role
            self.__tools[tool_properties.name] = ToolBuilder.build_tool(
                tool_properties=tool_properties,
                vector_db_properties=self.__stack_properties.vector_db
            )

    def initialize_stack(self):
        """
        Initialize the entire stack, including properties, classifier, and tools.
        """
        self.__initialize_stack_properties()
        self.__initialize_classifier()
        self.__initialize_tools()
        self.__initialized = True

    def classify_query(self, query: str) -> str:
        """
        Classify the input query into one of the tool roles using the classifier.
        """
        self.__check_initialized()

        role_definitions = "\n".join(
            [f"{name}: {role}" for name, role in self.__tool_roles.items()]
        )

        prompt = f"""
        Classify the following query into one of the categories: {', '.join(self.__tool_roles.values())}.

        Category definitions: 
        {role_definitions}

        Query: "{query}"

        Please return only the category name, without any extra text or prefix.
        """
        print(prompt)
        return self.__classifier.generate_answer(query=prompt)

    def process_query(self, query: str) -> str:
        """
        Process the input query, classify it, and use the appropriate tool.
        """
        self.__check_initialized()

        tool_name = self.classify_query(query)
        if tool_name not in self.__tools:
            raise ValueError(f"No tool found for the classified role '{tool_name}'.")

        return self.__tools[tool_name].process_query(query)
