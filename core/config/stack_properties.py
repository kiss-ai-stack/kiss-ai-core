import os
from core.config import StackValidator
from core.models.config import AgentProperties
from core.utilities import YamlReader

def stack_properties(stack_config_env_var: str = 'STACK_CONFIG', default_file: str = 'stack.yaml') -> AgentProperties:
    """
    Load and validate stack properties from a YAML configuration file.

    Parameters:
        stack_config_env_var (str): The environment variable name for the stack config path.
        default_file (str): The default file name for the stack.yaml file.

    Returns:
        AgentProperties: Validated agent properties from the YAML file.
    """
    stack_config_path = os.getenv(stack_config_env_var)

    if stack_config_path:
        resolved_path = os.path.abspath(stack_config_path)
    else:
        command_dir = os.getcwd()
        resolved_path = os.path.join(command_dir, default_file)

    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f'Configuration file not found at: {resolved_path}')

    try:
        with YamlReader(resolved_path) as reader:
            config_dict = reader.read()
            return StackValidator.validate(config_dict)
    except Exception as e:
        raise RuntimeError(f'Failed to load or validate stack configuration: {e}')
