import functools
from typing import Dict, List, Optional, Union, Callable, TypeVar, Any

from kiss_ai_stack.core.stack.stack import Stack
from kiss_ai_stack.core.models.core.rag_response import ToolResponse
from kiss_ai_stack.core.utilities.logger import LOG

T = TypeVar('T')


class Stacks:
    """
    A centralized management class for creating, initializing, and interacting with session based tools.

    The Stacks class provides a comprehensive interface for:
    - Bootstrapping new stacks
    - Storing and retrieving documents
    - Processing queries
    - Managing stack lifecycles

    Key Features:
    - Thread-safe stack management
    - Flexible stack initialization with temporary or persistent modes

    :param __stacks: A private dictionary storing active stack instances, keyed by their unique identifiers.
    """

    __stacks: Dict[str, Stack] = {}

    @staticmethod
    def _require_stack(func: Callable[..., T]) -> Callable[..., T]:
        """
        A decorator to enforce stack existence before method execution.

        This decorator ensures that a stack with the specified ID exists in the stack
        before allowing the decorated method to be called. If the stack is not found,
        it raises a KeyError with a descriptive message.

        :param func: The method to be decorated with stack validation.
        :returns: A wrapped method that checks for stack existence.
        :raises KeyError: If the specified stack ID is not found in the stack.
        """

        @functools.wraps(func)
        def wrapper(cls, stack_id: str, *args, **kwargs):
            if stack_id not in cls.__stacks:
                LOG.error(f'Stacks :: No stack found with ID \'{stack_id}\'')
                raise KeyError(f'Stacks :: Stack \'{stack_id}\' not found')
            return func(cls, stack_id, *args, **kwargs)

        return wrapper

    @classmethod
    async def bootstrap_stack(cls, stack_id: str, temporary: Optional[bool] = True) -> None:
        """
        Initialize a new stack session in the stack with optional persistence configuration.

        This method creates a new Stack instance, adds it to the stack, and initializes
        its underlying components. It supports both temporary (stateless) and
        persistent stack configurations.

        :param stack_id: The unique identifier for the stack.
        :param temporary: If True (default), creates a stateless stack with ephemeral resources.
                          If False, creates a persistent stack with retained state and resources.

        :raises RuntimeError: If stack initialization fails due to configuration or resource issues.
        """
        try:
            stack = Stack(stack_id=stack_id, temporary=temporary)
            cls.__stacks[stack_id] = stack
            await cls.__stacks[stack_id].initialize_stack()
            LOG.info(f'Stacks :: Stack \'{stack_id}\' initialized successfully')
        except Exception as e:
            LOG.error(f'Stacks :: Stack initialization failed for stack \'{stack_id}\': {e}')
            raise

    @classmethod
    @_require_stack
    async def generate_answer(
            cls,
            stack_id: str,
            query: Union[str, Dict, List]
    ) -> Optional[ToolResponse]:
        """
        Process a query using a specific stack's capabilities.

        Supports flexible query formats and returns a structured tool response.
        Handles query processing, potential tool interactions, and result generation.

        :param stack_id: The identifier of the stack to process the query.
        :param query: A string query, a dictionary with structured query parameters,
                      or a list of query components.

        :returns: A structured response containing:
            - Generated answer
            - Used tools
            - Metadata about query processing

        :raises ValueError: If query processing encounters unrecoverable errors.
        """
        try:
            response = await cls.__stacks[stack_id].process_query(query)
            LOG.info(f'Stacks :: Query processed successfully for stack \'{stack_id}\'')
            return response
        except Exception as e:
            LOG.error(f'Stacks :: Query processing failed for stack \'{stack_id}\': {e}')
            raise

    @classmethod
    @_require_stack
    async def store_data(
            cls,
            stack_id: str,
            files: List[str],
            metadata: Optional[Dict[str, Any]] = None,
            classify_document: bool = True
    ) -> Dict[str, List[str]]:
        """
        Store documents for a specific stack.

        :param stack_id: (str) Identifier of the stack to use
        :param files: (List[str]) List of file paths to store
        :param metadata: (Optional[Dict[str, Any]]) Optional metadata to associate with documents
        :param classify_document: (bool, optional) Whether to classify documents. Defaults to True.

        :returns: Dict[str, List[str]]: Dictionary of stored document IDs per tool

        :raises ValueError: If document storage fails
        """
        try:
            stored_documents = await cls.__stacks[stack_id].store_documents(
                files=files,
                metadata=metadata,
                classify_document=classify_document
            )
            LOG.info(f'Stacks :: Documents stored successfully for stack \'{stack_id}\'')
            return stored_documents
        except Exception as e:
            LOG.error(f'Stacks :: Document storage failed for stack \'{stack_id}\': {e}')
            raise

    @classmethod
    @_require_stack
    def get_stack(cls, stack_id: str) -> Optional[Stack]:
        """
        Retrieve a specific stack session by its Id.

        :param stack_id: (str) Identifier of the stack to retrieve
        :returns Optional[Stack]: The stack instance if found, None otherwise
        """
        return cls.__stacks.get(stack_id)

    @classmethod
    @_require_stack
    async def destroy_stack(cls, stack_id: str, cleanup=False):
        """
        Destroys a stack based session and remove it from the stack sessions list.

        :param stack_id: Identifier of the stack to destroy.
        :param cleanup: Prompt to remove user data if RAG tools present.
        """
        stack = cls.__stacks.get(stack_id)
        if stack:
            await stack.destroy_stack(cleanup)
            del cls.__stacks[stack_id]
            LOG.info(f'Stacks :: Stack-\'{stack_id}\' closed successfully')
        else:
            LOG.warning(f'Stacks :: Stack-\'{stack_id}\' not found')
        return
