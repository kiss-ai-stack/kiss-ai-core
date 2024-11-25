class RagResponse:

    def __init__(self, answer, docs, metadata, distances):
        self.answer = answer
        self.supporting_documents = docs
        self.metadata = metadata
        self.distances = distances
