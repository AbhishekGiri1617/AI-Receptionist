import os
from pathlib import Path
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LiveKit Configuration
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str

    # LLM Configuration
    groq_api_key: str
    groq_model: str = "openai/gpt-oss-120b"

    # Document Ingestion & RAG
    docs_dir: str = "docs"
    embedding_model: str = "all-MiniLM-L6-v2"
    qdrant_path: str = "./data/qdrant"
    collection_name: str = "business_docs"
    chunk_size: int = 500
    top_k: int = 5

    # Business Config File Path
    business_config_path: str = "agent_config.yaml"

    # Voice Configuration
    stt_model: str = "deepgram/nova-3"
    stt_language: str = "multi"
    tts_model: str = "inworld/inworld-tts-2"
    tts_voice: str = "Dennis"


settings = Settings()

# Set environment variables for LiveKit & Groq
os.environ.setdefault("LIVEKIT_URL", settings.livekit_url)
os.environ.setdefault("LIVEKIT_API_KEY", settings.livekit_api_key)
os.environ.setdefault("LIVEKIT_API_SECRET", settings.livekit_api_secret)
os.environ.setdefault("GROQ_API_KEY", settings.groq_api_key)


def load_business_config(config_path: str = None) -> dict:
    """Load business and persona configuration from YAML, with sensible fallbacks."""
    base_dir = Path(__file__).parent
    target_path = Path(config_path or settings.business_config_path)
    if not target_path.is_absolute():
        target_path = base_dir / target_path

    if not target_path.exists():

        return {
            "company_name": "Our Company",
            "agent_name": "Assistant",
            "welcome_greeting": "Greet the caller warmly as the customer support assistant and ask how you can help.",
            "persona": {
                "role": "You are a helpful and professional customer support voice assistant.",
                "tone": "Warm, concise, and conversational. Keep responses to 1-2 sentences.",
                "guidelines": [
                    "Use search_knowledge_base to answer questions about products, services, or policies.",
                    "If the answer is not in the knowledge base, offer to take an inquiry or escalate.",
                ],
            },
            "tools": {
                "enable_knowledge_base": True,
                "enable_inquiry_capture": True,
                "enable_escalation": True,
            },
        }

    with open(target_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


business_config = load_business_config()

