import os
import asyncio
import json
from groq import Groq
from openai import OpenAI
from huggingface_hub import InferenceClient
from typing import Dict, List, Optional
from pydantic import BaseModel


class ProviderConfig(BaseModel):
    model_name: str
    api_key: str
    max_retries: int = 5


class LLMInferenceClient:
    """Unified client for multiple LLM inference APIs with retry logic."""

    def __init__(self, providers: Dict[str, ProviderConfig]):
        """
        Initialize the client with configurations for multiple providers.

        Args:
            providers: Dictionary mapping provider names to their configs
                      e.g., {"hf": ProviderConfig(...), "groq": ProviderConfig(...)}
        """
        self.providers = providers
        self._clients = {}  # Cache for initialized clients

    def _get_config(self, provider: str) -> ProviderConfig:
        """Get configuration for a specific provider."""
        if provider not in self.providers:
            raise ValueError(
                f"Provider '{provider}' not configured. Available: {list(self.providers.keys())}"
            )
        return self.providers[provider]

    def _get_hf_client(self, provider: str) -> InferenceClient:
        """Get or create HuggingFace client (cached)."""
        if provider not in self._clients:
            config = self._get_config(provider)
            self._clients[provider] = InferenceClient(api_key=config.api_key)
        return self._clients[provider]

    def _get_groq_client(self, provider: str) -> Groq:
        """Get or create Groq client (cached)."""
        if provider not in self._clients:
            config = self._get_config(provider)
            self._clients[provider] = Groq(api_key=config.api_key)
        return self._clients[provider]

    def _get_deepinfra_client(self, provider: str) -> OpenAI:
        """Get or create DeepInfra client (cached)."""
        if provider not in self._clients:
            config = self._get_config(provider)
            self._clients[provider] = OpenAI(
                api_key=config.api_key,
                base_url="https://api.deepinfra.com/v1/openai",
            )
        return self._clients[provider]

    def set_model(self, provider: str, model_name: str):
        """
        Change the model for a specific provider on the fly.

        Args:
            provider: Provider name
            model_name: New model name to use
        """
        if provider not in self.providers:
            raise ValueError(
                f"Provider '{provider}' not configured. Available: {list(self.providers.keys())}"
            )
        self.providers[provider].model_name = model_name
        print(f"[✓] Updated {provider} model to: {model_name}")

    def _sync_hf_call(self, client, model_name, messages):
        """Synchronous HuggingFace API call to be run in thread pool."""
        return client.chat_completion(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )

    def _sync_groq_call(self, client, model_name, formatted_prompt):
        """Synchronous Groq API call to be run in thread pool."""
        return client.chat.completions.create(
            messages=[{"role": "user", "content": formatted_prompt}],
            model=model_name,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

    def _sync_deepinfra_call(self, client, model_name, formatted_prompt):
        """Synchronous DeepInfra API call to be run in thread pool."""
        return client.chat.completions.create(
            messages=[{"role": "user", "content": formatted_prompt}],
            model=model_name,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

    async def hf_inference(
        self, feedback_text: str, prompt_template: str, provider: str = "hf"
    ) -> str:
        """HuggingFace Inference API with exponential backoff - truly async."""
        config = self._get_config(provider)
        client = self._get_hf_client(provider)
        formatted_prompt = prompt_template.format(feedback=feedback_text)
        messages = [{"role": "user", "content": formatted_prompt}]

        retries = 0
        current_sleep = 5  # Initial sleep time for backoff

        while retries < config.max_retries:
            try:
                # Run the blocking call in a thread pool to not block the event loop
                response = await asyncio.to_thread(
                    self._sync_hf_call, client, config.model_name, messages
                )
                return response.choices[0].message.content

            except Exception as e:
                error_msg = str(e)
                print(f"\n[!] HF API Error: {error_msg}")

                # Check for common temporary errors
                if "429" in error_msg or "503" in error_msg or "504" in error_msg:
                    print(
                        f"    Rate limit or Model loading. Sleeping {current_sleep}s..."
                    )
                    await asyncio.sleep(current_sleep)  # Non-blocking sleep
                    current_sleep *= 2  # Exponential backoff
                    retries += 1
                else:
                    # If it's a weird error (like 400 Bad Request), log it and skip
                    print("    Unrecoverable error. Skipping this sample.")
                    return "{}"

        print("    Max retries reached. Skipping.")
        return "{}"

    async def groq_inference(
        self, feedback_text: str, prompt_template: str, provider: str = "groq"
    ) -> Optional[str]:
        """Groq API inference with retry logic - truly async."""
        config = self._get_config(provider)
        client = self._get_groq_client(provider)
        formatted_prompt = prompt_template.format(feedback=feedback_text)

        retries = 0
        while retries < config.max_retries:
            try:
                # Run the blocking call in a thread pool to not block the event loop
                chat_completion = await asyncio.to_thread(
                    self._sync_groq_call, client, config.model_name, formatted_prompt
                )
                return chat_completion.choices[0].message.content

            except Exception as e:
                error_msg = str(e)
                print(f"\n[!] Groq Error: {error_msg}")

                # Rate Limit (Groq usually resets every minute)
                if "429" in error_msg:
                    print("    Rate limit hit. Sleeping 60 seconds...")
                    await asyncio.sleep(60)  # Non-blocking sleep
                    retries += 1
                else:
                    return None  # Return None on hard error

        return None

    async def deepinfra_inference(
        self, feedback_text: str, prompt_template: str, provider: str = "deepinfra"
    ) -> Optional[str]:
        """DeepInfra API inference with retry logic - truly async."""
        config = self._get_config(provider)
        client = self._get_deepinfra_client(provider)
        formatted_prompt = prompt_template.format(feedback=feedback_text)

        retries = 0
        while retries < config.max_retries:
            try:
                # Run the blocking call in a thread pool to not block the event loop
                chat_completion = await asyncio.to_thread(
                    self._sync_deepinfra_call, client, config.model_name, formatted_prompt
                )
                return chat_completion.choices[0].message.content

            except Exception as e:
                error_msg = str(e)
                print(f"\n[!] DeepInfra Error: {error_msg}")

                # Rate Limit (DeepInfra usually resets every minute)
                if "429" in error_msg:
                    print("    Rate limit hit. Sleeping 60 seconds...")
                    await asyncio.sleep(60)  # Non-blocking sleep
                    retries += 1
                else:
                    return None  # Return None on hard error

        return None

    async def infer(
        self, feedback_text: str, prompt_template: str, provider: str
    ) -> Optional[str]:
        """
        Unified inference method that routes to the appropriate API.

        Args:
            feedback_text: The feedback text to process
            prompt_template: Template string with {feedback} placeholder
            provider: Provider name (must be configured during initialization)

        Returns:
            Response content string or None on error
        """
        # Check if provider is configured
        if provider not in self.providers:
            raise ValueError(
                f"Provider '{provider}' not configured. Available: {list(self.providers.keys())}"
            )

        provider_lower = provider.lower()

        # Route to appropriate method based on provider type
        if provider_lower in ["hf", "huggingface"]:
            return await self.hf_inference(feedback_text, prompt_template, provider)
        elif provider_lower == "groq":
            return await self.groq_inference(feedback_text, prompt_template, provider)
        elif provider_lower == "deepinfra":
            return await self.deepinfra_inference(
                feedback_text, prompt_template, provider
            )
        else:
            raise ValueError(
                f"Unknown provider type: {provider}. Use 'hf', 'groq', or 'deepinfra'"
            )
