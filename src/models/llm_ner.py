import requests
from src.config import (
    LLM_REQUEST_TIMEOUT,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_NAME,
)

class OllamaNERClient:
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model_name: str = OLLAMA_MODEL_NAME,
        timeout: int = LLM_REQUEST_TIMEOUT,
    ) -> None:
        self.base_url = base_url
        self.model_name = model_name
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": LLM_TEMPERATURE,
            },
        }

        response = requests.post(
            self.base_url,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        return data.get("response", "")