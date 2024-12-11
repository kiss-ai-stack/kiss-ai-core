import functools
from typing import Dict, List, Optional, Union, Callable, TypeVar

from kiss_ai_stack.core.agent.agent import Agent
from kiss_ai_stack.core.models.core.rag_response import ToolResponse
from kiss_ai_stack.core.utilities import LOG

T = TypeVar('T')



class AgentStack:
    """
    A centralized management class for creating, initializing, and interacting with agents.
    """

    __agents: Dict[str, Agent] = {}

    @staticmethod
    def _require_agent(func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator to validate agent existence before method execution.

        Args:
            func (Callable): Method to be decorated

        Returns:
            Callable: Decorated method with agent existence check

        Raises:
            KeyError: If the specified agent does not exist
        """

        @functools.wraps(func)
        def wrapper(cls, agent_id: str, *args, **kwargs):
            if agent_id not in cls.__agents:
                LOG.error(f'AgentStack :: No agent found with ID \'{agent_id}\'')
                raise KeyError(f'AgentStack :: Agent \'{agent_id}\' not found')
            return func(cls, agent_id, *args, **kwargs)

        return wrapper

    @classmethod
    def bootstrap_agent(cls, agent_id: str) -> None:
        """
        Initialize the stack for a specific agent.

        Args:
            agent_id (str): Identifier of the agent to bootstrap

        Raises:
            RuntimeError: If stack initialization fails
        """
        try:
            agent = Agent(agent_id=agent_id)
            cls.__agents[agent_id] = agent
            cls.__agents[agent_id].initialize_stack()
            LOG.info(f'AgentStack :: Agent \'{agent_id}\' initialized successfully')
        except Exception as e:
            LOG.error(f'AgentStack :: Stack initialization failed for agent \'{agent_id}\': {e}')
            raise

    @classmethod
    @_require_agent
    def generate_answer(
            cls,
            agent_id: str,
            query: Union[str, Dict, List]
    ) -> Optional[ToolResponse]:
        """
        Generate an answer for a given query using a specific agent.

        Args:
            agent_id (str): Identifier of the agent to use
            query (Union[str, Dict, List]): Input query to process

        Returns:
            Optional[ToolResponse]: Response from the agent's query processing

        Raises:
            ValueError: If query processing fails
        """
        try:
            response = cls.__agents[agent_id].process_query(query)
            LOG.info(f'AgentStack :: Query processed successfully for agent \'{agent_id}\'')
            return response
        except Exception as e:
            LOG.error(f'AgentStack :: Query processing failed for agent \'{agent_id}\': {e}')
            raise

    @classmethod
    @_require_agent
    def store_data(
            cls,
            agent_id: str,
            files: List[str],
            classify_document: bool = True
    ) -> Dict[str, List[str]]:
        """
        Store documents for a specific agent.

        Args:
            agent_id (str): Identifier of the agent to use
            files (List[str]): List of file paths to store
            classify_document (bool, optional): Whether to classify documents. Defaults to True.

        Returns:
            Dict[str, List[str]]: Dictionary of stored document IDs per tool

        Raises:
            ValueError: If document storage fails
        """
        try:
            stored_documents = cls.__agents[agent_id].store_documents(
                files=files,
                classify_document=classify_document
            )
            LOG.info(f'AgentStack :: Documents stored successfully for agent \'{agent_id}\'')
            return stored_documents
        except Exception as e:
            LOG.error(f'AgentStack :: Document storage failed for agent \'{agent_id}\': {e}')
            raise

    @classmethod
    @_require_agent
    def get_agent(cls, agent_id: str) -> Optional[Agent]:
        """
        Retrieve a specific agent by its ID.

        Args:
            agent_id (str): Identifier of the agent to retrieve

        Returns:
            Optional[Agent]: The agent instance if found, None otherwise
        """
        return cls.__agents.get(agent_id)

    @classmethod
    @_require_agent
    def destroy_agent(cls, agent_id: str, cleanup = False):
        """
        Destroys an Agent and remove it from the agents list.

        Args:
            agent_id: Identifier of the agent to destroy.
            cleanup: Prompt to remove user data if RAG tools present.
        """

        cls.__agents.get(agent_id).destroy_stack(cleanup)
        del cls.__agents[agent_id]
        LOG.info(f'AgentStack :: Agent\'{agent_id}\' closed successfully')
        return
