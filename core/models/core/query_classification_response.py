from typing import Optional


class QueryClassificationResponse:
    """
    Structured response for query classification.
    """

    def __init__(
            self,
            tool_name: Optional[str] = None,
            confidence: float = 0.0,
            reasoning: Optional[str] = None
    ):
        self.tool_name = tool_name
        self.confidence = confidence
        self.reasoning = reasoning
