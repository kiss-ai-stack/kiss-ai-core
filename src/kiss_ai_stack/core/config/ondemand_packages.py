from kiss_ai_stack.core.models.enums.ai_client_vendor import AIClientVendor
from kiss_ai_stack.core.models.enums.db_vendor import VectorDBVendor

AI_CLIENT = {
    AIClientVendor.OPENAI: 'openai~=1.55.0',
}

VECTOR_DB = {
    VectorDBVendor.CHROMA: 'chromadb~=0.5.23'
}
