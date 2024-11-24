import os

from core.config import StackValidator
from core.models.config import AgentProperties
from core.utilities import YamlReader


def stack_properties(stack_config: str = 'stack.yaml') -> AgentProperties:
    try:
        with YamlReader(stack_config) as reader:
            config_dict = reader.read()
            return StackValidator.validate(config_dict)
    except Exception:
        raise
