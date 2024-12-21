from pydantic import ValidationError

from kiss_ai_stack.core.models.config.stack_props import StackProperties


class StackValidator:

    @staticmethod
    def validate(data: dict) -> StackProperties:
        """
        Validates the YAML data and returns an StackProperties object using Pydantic models.

        :param data: The parsed YAML data
        :raises ValueError: If validation rules are violated
        :returns: StackProperties instance
        """
        if 'stack' not in data:
            raise ValueError('StackValidator :: Missing \'stack\' section in YAML.')
        stack_data = data['stack']

        if 'decision_maker' not in stack_data:
            raise ValueError('StackValidator :: Missing or invalid \'decision_maker\' section in \'stack\'.')
        else:
            if stack_data['decision_maker']['kind'] != 'prompt':
                raise ValueError(f'StackValidator :: \'decision_maker\' :: only supports `prompt` kind.')

        if 'tools' not in stack_data or not isinstance(stack_data['tools'], list):
            raise ValueError('StackValidator :: Missing or invalid \'tools\' section in \'stack\'. It must be a list.')

        try:
            stack = StackProperties(**stack_data)
        except ValidationError as e:
            raise ValueError(f'StackValidator :: Validation error: {e}')

        return stack
