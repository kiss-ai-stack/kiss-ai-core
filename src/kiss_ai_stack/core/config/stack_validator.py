from pydantic import ValidationError

from kiss_ai_stack.core.models.config.agent import AgentProperties


class StackValidator:

    @staticmethod
    def validate(data: dict) -> AgentProperties:
        """
        Validates the YAML data and returns an AgentConfig object using Pydantic models.

        :param data: The parsed YAML data
        :raises ValueError: If validation rules are violated
        :returns: AgentConfig instance
        """
        if 'agent' not in data:
            raise ValueError('StackValidator :: Missing \'agent\' section in YAML.')
        agent_data = data['agent']

        if 'decision_maker' not in agent_data:
            raise ValueError('StackValidator :: Missing or invalid \'decision_maker\' section in \'agent\'.')
        else:
            if agent_data['decision_maker']['kind'] != 'prompt':
                raise ValueError(f'StackValidator :: \'decision_maker\' :: only supports `prompt` kind.')

        if 'tools' not in agent_data or not isinstance(agent_data['tools'], list):
            raise ValueError('StackValidator :: Missing or invalid \'tools\' section in \'agent\'. It must be a list.')

        try:
            agent = AgentProperties(**agent_data)
        except ValidationError as e:
            raise ValueError(f'StackValidator :: Validation error: {e}')

        return agent
