from pydantic import ValidationError

from core.models.config import AgentProperties


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
            raise ValueError("Missing 'agent' section in YAML.")
        agent_data = data['agent']
        if 'classifier' not in agent_data:
            raise ValueError("Missing or invalid 'classifier' section in 'agent'.")
        if 'ai_clients' not in agent_data or not isinstance(agent_data['ai_clients'], list):
            raise ValueError("Missing or invalid 'ai_clients' section in 'agent'. It must be a list.")

        try:
            agent = AgentProperties(**agent_data)
        except ValidationError as e:
            raise ValueError(f"Validation error: {e}")

        return agent
