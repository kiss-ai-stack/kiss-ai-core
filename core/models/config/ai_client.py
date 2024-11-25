from pydantic import BaseModel


class AIClientProperties(BaseModel):
    provider: str
    model: str
    host: str
    api_key: str

    class Config:
        str_min_length = 1
        str_strip_whitespace = True
