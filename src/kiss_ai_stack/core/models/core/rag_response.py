class ToolResponse:

    def __init__(self, answer, docs=None, metadata=None, distances=None):
        self.answer = answer
        self.supporting_documents = docs
        self.metadata = metadata
        self.distances = distances
