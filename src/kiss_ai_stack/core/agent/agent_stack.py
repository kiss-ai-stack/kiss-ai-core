import functools
from typing import Dict, List, Optional, Union, Callable, TypeVar, Any

from kiss_ai_stack.core.agent.agent import Agent
from kiss_ai_stack.core.models.core.rag_response import ToolResponse
from kiss_ai_stack.core.utilities.logger import LOG

T = TypeVar('T')


class AgentStack:
    """
    A centralized management class for creating, initializing, and interacting with session based AI agents.

    The AgentStack provides a comprehensive interface for:
    - Bootstrapping new agents
    - Storing and retrieving documents
    - Processing queries
    - Managing agent lifecycles

    Key Features:
    - Thread-safe agent management
    - Flexible agent initialization with temporary or persistent modes

    :param __agents: A private dictionary storing active agent instances, keyed by their unique identifiers.
    """

    __agents: Dict[str, Agent] = {}

    @staticmethod
    def _require_agent(func: Callable[..., T]) -> Callable[..., T]:
        """
        A decorator to enforce agent existence before method execution.

        This decorator ensures that an agent with the specified ID exists in the stack
        before allowing the decorated method to be called. If the agent is not found,
        it raises a KeyError with a descriptive message.

        :param func: The method to be decorated with agent validation.
        :returns: A wrapped method that checks for agent existence.
        :raises KeyError: If the specified agent ID is not found in the stack.
        """

        @functools.wraps(func)
        def wrapper(cls, agent_id: str, *args, **kwargs):
            if agent_id not in cls.__agents:
                LOG.error(f'AgentStack :: No agent found with ID \'{agent_id}\'')
                raise KeyError(f'AgentStack :: Agent \'{agent_id}\' not found')
            return func(cls, agent_id, *args, **kwargs)

        return wrapper

    @classmethod
    async def bootstrap_agent(cls, agent_id: str, temporary: Optional[bool] = True) -> None:
        """
        Initialize a new agent session in the stack with optional persistence configuration.

        This method creates a new Agent instance, adds it to the stack, and initializes
        its underlying components. It supports both temporary (stateless) and
        persistent agent configurations.

        :param agent_id: The unique identifier for the agent.
        :param temporary: If True (default), creates a stateless agent with ephemeral resources.
                          If False, creates a persistent agent with retained state and resources.

        :raises RuntimeError: If agent initialization fails due to configuration or resource issues.
        """
        try:
            agent = Agent(agent_id=agent_id, temporary=temporary)
            cls.__agents[agent_id] = agent
            await cls.__agents[agent_id].initialize_stack()
            LOG.info(f'AgentStack :: Agent \'{agent_id}\' initialized successfully')
        except Exception as e:
            LOG.error(f'AgentStack :: Stack initialization failed for agent \'{agent_id}\': {e}')
            raise

    @classmethod
    @_require_agent
    async def generate_answer(
            cls,
            agent_id: str,
            query: Union[str, Dict, List]
    ) -> Optional[ToolResponse]:
        """
        Process a query using a specific agent's capabilities.

        Supports flexible query formats and returns a structured tool response.
        Handles query processing, potential tool interactions, and result generation.

        :param agent_id: The identifier of the agent to process the query.
        :param query: A string query, a dictionary with structured query parameters,
                      or a list of query components.

        :returns: A structured response containing:
            - Generated answer
            - Used tools
            - Metadata about query processing

        :raises ValueError: If query processing encounters unrecoverable errors.
        """
        try:
            response = await cls.__agents[agent_id].process_query(query)
            LOG.info(f'AgentStack :: Query processed successfully for agent \'{agent_id}\'')
            return response
        except Exception as e:
            LOG.error(f'AgentStack :: Query processing failed for agent \'{agent_id}\': {e}')
            raise

    @classmethod
    @_require_agent
    async def store_data(
            cls,
            agent_id: str,
            files: List[str],
            metadata: Optional[Dict[str, Any]] = None,
            classify_document: bool = True
    ) -> Dict[str, List[str]]:
        """
        Store documents for a specific agent.

        :param agent_id: (str) Identifier of the agent to use
        :param files: (List[str]) List of file paths to store
        :param metadata: (Optional[Dict[str, Any]]) Optional metadata to associate with documents
        :param classify_document: (bool, optional) Whether to classify documents. Defaults to True.

        :returns: Dict[str, List[str]]: Dictionary of stored document IDs per tool

        :raises ValueError: If document storage fails
        """
        try:
            stored_documents = await cls.__agents[agent_id].store_documents(
                files=files,
                metadata=metadata,
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
        Retrieve a specific agent session by its Id.

        :param agent_id: (str) Identifier of the agent to retrieve
        :returns Optional[Agent]: The agent instance if found, None otherwise
        """
        return cls.__agents.get(agent_id)

    @classmethod
    @_require_agent
    async def destroy_agent(cls, agent_id: str, cleanup=False):
        """
        Destroys an Agent session and remove it from the agent sessions list.

        :param agent_id: Identifier of the agent to destroy.
        :param cleanup: Prompt to remove user data if RAG tools present.
        """
        agent = cls.__agents.get(agent_id)
        if agent:
            await agent.destroy_stack(cleanup)
            del cls.__agents[agent_id]
            LOG.info(f'AgentStack :: Agent\'{agent_id}\' closed successfully')
        else:
            LOG.warning(f'AgentStack :: Agent\'{agent_id}\' not found')
        return
