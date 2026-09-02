from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    embedding_model: str = "intfloat/multilingual-e5-base"
    chat_model: str = "gpt-4o-mini"
    correction_priority: float = 10.0
    data_dir: str = "data"

    class Config:
        env_file = ".env"

settings = Settings()
