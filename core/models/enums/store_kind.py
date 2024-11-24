from enum import StrEnum


class StoreKind(StrEnum):
    IN_MEMORY = 'in_memory'
    STORAGE = 'storage'
    REMOTE = 'remote'
