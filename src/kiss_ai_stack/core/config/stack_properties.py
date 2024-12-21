import os

from kiss_ai_stack.core.config.stack_validator import StackValidator
from kiss_ai_stack.core.models.config.stack_props import StackProperties
from kiss_ai_stack.core.utilities.yaml_reader import YamlReader


async def stack_properties(stack_config_env_var: str = 'STACK_CONFIG',
                           default_file: str = 'stack.yaml') -> StackProperties:
    """
    Asynchronously loads and validates stack properties from a YAML configuration file.

    Attempts to retrieve the stack configuration file path from an environment variable.
    If not found, it defaults to 'stack.yaml' located in the current working directory.
    Reads and validates the YAML configuration asynchronously.

    :param stack_config_env_var: The environment variable name for the stack config path.
                                  Defaults to 'STACK_CONFIG'.
    :param default_file: The default file name for the stack.yaml file if the environment
                         variable is not set. Defaults to 'stack.yaml'.

    :return: Validated stack properties from the YAML configuration file as an instance of
             StackProperties.
    :raises FileNotFoundError: If the configuration file is not found at the resolved path.
    :raises RuntimeError: If an error occurs while reading or validating the YAML configuration.
    """
    stack_config_path = os.getenv(stack_config_env_var)

    if stack_config_path:
        resolved_path = os.path.abspath(stack_config_path)
    else:
        command_dir = os.getcwd()
        resolved_path = os.path.join(command_dir, default_file)

    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f'stack_properties :: Configuration file not found at: {resolved_path}')

    try:
        async with YamlReader(resolved_path) as reader:
            config_dict = await reader.read()
            return StackValidator.validate(config_dict)
    except Exception as e:
        raise RuntimeError(f'stack_properties :: Failed to load or validate stack configuration: {e}')
