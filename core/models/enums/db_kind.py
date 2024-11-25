from enum import StrEnum


class VectorDBKind(StrEnum):
    IN_MEMORY = 'in_memory'
    STORAGE = 'storage'
    REMOTE = 'remote'
