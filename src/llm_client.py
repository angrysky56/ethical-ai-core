"""
LLM Client - Unified interface for multiple LLM providers.
Supports: OpenRouter, Ollama, LM Studio, OpenAI
"""
import json
import os
import httpx
from typing import Optional
from src.config import get_llm_config, LLM_PROVIDER

class LLMClient:
    """
    Unified LLM client that works with OpenAI-compatible APIs.
    Ollama, LM Studio, and OpenRouter all expose OpenAI-compatible endpoints.
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or LLM_PROVIDER
        self.config = get_llm_config()
        self.timeout = float(os.getenv("LLM_TIMEOUT", "1200"))  # Increased default to 20m for local inference

    def _get_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.config["api_key"]:
            headers["Authorization"] = f"Bearer {self.config['api_key']}"
        # OpenRouter requires additional headers
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/ethical-ai-core"
            headers["X-Title"] = "Ethical AI Core"
        return headers

    def _get_endpoint(self) -> str:
        base = self.config["base_url"].rstrip("/")
        if self.provider == "ollama":
            return f"{base}/api/chat"
        else:
            return f"{base}/chat/completions"

    def list_models(self) -> list[str]:
        """List available models from the provider."""
        base = self.config["base_url"].rstrip("/")
        try:
            if self.provider == "ollama":
                resp = httpx.get(f"{base}/api/tags", timeout=5.0)
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    return [m["name"] for m in models]
            else:
                # OpenAI compatible
                resp = httpx.get(f"{base}/models", headers=self._get_headers(), timeout=5.0)
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    return [m["id"] for m in data]
        except Exception:
            pass
        return []

    def update_config(self, model: Optional[str] = None, timeout: Optional[float] = None):
        """Update runtime configuration."""
        if model:
            self.config["model"] = model
        if timeout:
            self.timeout = timeout

    def _format_request(self, messages: list[dict], temperature: float = 0.7) -> dict:
        """Format request body for the specific provider."""
        if self.provider == "ollama":
            # Ollama uses a slightly different format
            return {
                "model": self.config["model"],
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature}
            }
        else:
            # OpenAI-compatible format (OpenRouter, LM Studio, OpenAI)
            return {
                "model": self.config["model"],
                "messages": messages,
                "temperature": temperature,
            }

    def _parse_response(self, response: dict) -> str:
        """Extract content from provider-specific response format."""
        if self.provider == "ollama":
            return response.get("message", {}).get("content", "")
        else:
            return response.get("choices", [{}])[0].get("message", {}).get("content", "")

    def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        """
        Send a chat completion request.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": str}
            temperature: Sampling temperature (0.0 = deterministic)

        Returns:
            The assistant's response text.
        """
        endpoint = self._get_endpoint()
        headers = self._get_headers()
        body = self._format_request(messages, temperature)

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(endpoint, headers=headers, json=body)
            response.raise_for_status()
            return self._parse_response(response.json())

    def complete(self, prompt: str, system: str = "", temperature: float = 0.7) -> str:
        """
        Convenience method for single-turn completion.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, temperature)


# Singleton for easy import
_client: Optional[LLMClient] = None

def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
