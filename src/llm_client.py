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
        # No timeout - some requests can take hours

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

    def list_openrouter_models_with_pricing(self) -> dict:
        """
        Fetch OpenRouter models with full metadata including pricing.

        Returns:
            Dict with structure:
            {
                "provider_name": [
                    {
                        "id": "provider/model-name",
                        "name": "Model Name",
                        "context_length": 128000,
                        "input_price": 0.0001,  # per 1K tokens
                        "output_price": 0.0002,
                        "is_free": False
                    }
                ]
            }
        """
        try:
            resp = httpx.get(
                "https://openrouter.ai/api/v1/models",
                headers=self._get_headers(),
                timeout=15.0
            )
            if resp.status_code != 200:
                return {}

            data = resp.json().get("data", [])

            # Group by provider (first part of model ID)
            providers = {}
            for model in data:
                model_id = model.get("id", "")
                if "/" not in model_id:
                    continue

                provider = model_id.split("/")[0]
                pricing = model.get("pricing", {})

                # Convert pricing from per-token to per-1K tokens
                input_price_per_token = float(pricing.get("prompt", "0") or "0")
                output_price_per_token = float(pricing.get("completion", "0") or "0")

                model_info = {
                    "id": model_id,
                    "name": model.get("name", model_id),
                    "context_length": model.get("context_length", 0),
                    "input_price": input_price_per_token * 1000,  # per 1K tokens
                    "output_price": output_price_per_token * 1000,
                    "is_free": input_price_per_token == 0 and output_price_per_token == 0,
                    "description": model.get("description", "")[:100]
                }

                if provider not in providers:
                    providers[provider] = []
                providers[provider].append(model_info)

            # Sort providers alphabetically, put free models first within each
            for provider in providers:
                providers[provider].sort(key=lambda m: (not m["is_free"], m["input_price"]))

            return dict(sorted(providers.items()))

        except Exception as e:
            print(f"[ERROR] Failed to fetch OpenRouter models: {e}")
            return {}

    def update_config(self, model: Optional[str] = None):
        """Update runtime configuration."""
        if model:
            self.config["model"] = model

    def _format_request(self, messages: list[dict], temperature: Optional[float] = None) -> dict:
        """
        Format request body for the specific provider.

        Args:
            messages: Chat messages
            temperature: Optional sampling temperature. If None, uses provider default.
                        Not all models support temperature - omitting lets OpenRouter handle it.
        """
        if self.provider == "ollama":
            # Ollama uses a slightly different format
            request = {
                "model": self.config["model"],
                "messages": messages,
                "stream": False,
            }
            if temperature is not None:
                request["options"] = {"temperature": temperature}
            return request
        else:
            # OpenAI-compatible format (OpenRouter, LM Studio, OpenAI)
            request = {
                "model": self.config["model"],
                "messages": messages,
            }
            # Only include temperature if explicitly set
            # OpenRouter will use model defaults if omitted
            if temperature is not None:
                request["temperature"] = temperature
            return request

    def _parse_response(self, response: dict) -> str:
        """Extract content from provider-specific response format."""
        if self.provider == "ollama":
            return response.get("message", {}).get("content", "")
        else:
            return response.get("choices", [{}])[0].get("message", {}).get("content", "")

    def chat(self, messages: list[dict], temperature: Optional[float] = None) -> str:
        """
        Send a chat completion request.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": str}
            temperature: Optional sampling temperature. If None, uses provider default.

        Returns:
            The assistant's response text.
        """
        endpoint = self._get_endpoint()
        headers = self._get_headers()
        body = self._format_request(messages, temperature)

        with httpx.Client(timeout=None) as client:  # No timeout
            response = client.post(endpoint, headers=headers, json=body)
            response.raise_for_status()
            return self._parse_response(response.json())

    def complete(self, prompt: str, system: str = "", temperature: Optional[float] = None) -> str:
        """
        Convenience method for single-turn completion.
        Temperature is optional - if None, uses provider/model defaults.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, temperature)

    def embed(self, text: str, model: Optional[str] = None) -> list[float]:
        """
        Generate embeddings for text using the current provider.

        Args:
            text: Text to embed
            model: Optional model override (e.g., 'nomic-embed-text' for Ollama)

        Returns:
            List of floats representing the embedding vector
        """
        base = self.config["base_url"].rstrip("/")
        headers = self._get_headers()

        # Use a sensible embedding model default per provider
        embed_model = model
        if not embed_model:
            if self.provider == "ollama":
                embed_model = "nomic-embed-text"
            elif self.provider == "openrouter":
                embed_model = "openai/text-embedding-3-small"
            else:
                embed_model = self.config["model"]  # Try current model

        if self.provider == "ollama":
            # Ollama uses different endpoint
            endpoint = f"{base}/api/embeddings"
            body = {"model": embed_model, "prompt": text}
        else:
            # OpenAI-compatible (OpenRouter, LM Studio, OpenAI)
            endpoint = f"{base}/embeddings"
            body = {"model": embed_model, "input": text}

        with httpx.Client(timeout=None) as client:
            response = client.post(endpoint, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()

            # Parse response based on provider format
            if self.provider == "ollama":
                return data.get("embedding", [])
            else:
                # OpenAI format
                return data.get("data", [{}])[0].get("embedding", [])


# Singleton for easy import
_client: Optional[LLMClient] = None

def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client

def reset_llm_client():
    """Reset the LLM client singleton. Call when provider changes."""
    global _client
    _client = None
