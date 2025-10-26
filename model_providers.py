"""
Model Provider Abstraction Layer for NxtCloud MCP Gateway

This module provides a unified interface for different AI model providers
including OpenAI and AWS Bedrock, enabling seamless model switching and
consistent error handling across providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import os
import boto3
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrock


@dataclass
class ModelConfig:
    """Configuration for a specific model"""

    display_name: str
    model_identifier: str  # model_name for OpenAI, model_id for Bedrock
    max_tokens: int
    temperature_range: Tuple[float, float]
    supports_streaming: bool
    additional_params: Dict[str, Any] = field(default_factory=dict)


class ModelProviderError(Exception):
    """Base exception for model provider errors"""

    pass


class AuthenticationError(ModelProviderError):
    """Authentication failed"""

    pass


class ModelNotAvailableError(ModelProviderError):
    """Requested model is not available"""

    pass


class RateLimitError(ModelProviderError):
    """Rate limit exceeded"""

    pass


class NetworkError(ModelProviderError):
    """Network connectivity issues"""

    pass


class ModelProvider(ABC):
    """Abstract base class for AI model providers"""

    @abstractmethod
    def create_model(self, model_config: ModelConfig, api_key: str, **kwargs) -> Any:
        """Create and return a model instance"""
        pass

    @abstractmethod
    def validate_credentials(self, api_key: str) -> bool:
        """Validate provider credentials"""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name"""
        pass

    def handle_error(self, error: Exception) -> str:
        """Convert provider-specific errors to user-friendly messages"""
        provider_name = self.get_provider_name()

        if (
            "authentication" in str(error).lower()
            or "unauthorized" in str(error).lower()
        ):
            return f"❌ {provider_name} 인증에 실패했습니다. API 키를 확인해주세요."
        elif "rate limit" in str(error).lower() or "quota" in str(error).lower():
            return f"⏱️ {provider_name} 사용량 한도에 도달했습니다. 잠시 후 다시 시도해주세요."
        elif "network" in str(error).lower() or "connection" in str(error).lower():
            return (
                f"🌐 {provider_name} 연결에 문제가 있습니다. 네트워크를 확인해주세요."
            )
        else:
            return f"❌ {provider_name} 모델 사용 중 오류가 발생했습니다: {str(error)}"


class OpenAIProvider(ModelProvider):
    """OpenAI model provider implementation"""

    def create_model(
        self, model_config: ModelConfig, api_key: str, **kwargs
    ) -> ChatOpenAI:
        """Create OpenAI model instance"""
        try:
            return ChatOpenAI(
                api_key=api_key,
                model=model_config.model_identifier,
                max_tokens=model_config.max_tokens,
                temperature=kwargs.get("temperature", 0.1),
                **model_config.additional_params,
            )
        except Exception as e:
            raise AuthenticationError(f"Failed to create OpenAI model: {str(e)}")

    def validate_credentials(self, api_key: str) -> bool:
        """Validate OpenAI API key format"""
        return api_key and api_key.startswith("sk-") and len(api_key) > 20

    def get_provider_name(self) -> str:
        return "OpenAI"


class BedrockProvider(ModelProvider):
    """AWS Bedrock model provider implementation"""

    def create_model(
        self, model_config: ModelConfig, api_key: str, **kwargs
    ) -> ChatBedrock:
        """Create AWS Bedrock model instance"""
        try:
            # Set AWS Bearer Token for Bedrock API Key authentication
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key

            # Create Bedrock client with Cross Region Inference
            client = boto3.client(
                service_name="bedrock-runtime",
                region_name="us-east-1",  # Fixed region as per requirements
            )

            return ChatBedrock(
                client=client,
                model_id=model_config.model_identifier,
                model_kwargs={
                    "max_tokens": model_config.max_tokens,
                    "temperature": kwargs.get("temperature", 0.1),
                    **model_config.additional_params,
                },
            )
        except Exception as e:
            raise AuthenticationError(f"Failed to create Bedrock model: {str(e)}")

    def validate_credentials(self, api_key: str) -> bool:
        """Validate AWS Bedrock API key format"""
        if not api_key or len(api_key) < 20:
            return False

        # Basic format validation for Bedrock API keys
        cleaned_key = api_key.replace("-", "").replace("_", "")
        return cleaned_key.isalnum()

    def get_provider_name(self) -> str:
        return "AWS Bedrock"


# Model Registry - Configuration for all supported models
MODEL_REGISTRY = {
    "openai": {
        "provider_class": OpenAIProvider,
        "models": {
            "gpt-4o": ModelConfig(
                display_name="OpenAI GPT-4o",
                model_identifier="gpt-4o",
                max_tokens=16000,
                temperature_range=(0.0, 2.0),
                supports_streaming=True,
            ),
            "gpt-4o-mini": ModelConfig(
                display_name="OpenAI GPT-4o Mini",
                model_identifier="gpt-4o-mini",
                max_tokens=16000,
                temperature_range=(0.0, 2.0),
                supports_streaming=True,
            ),
        },
    },
    "bedrock": {
        "provider_class": BedrockProvider,
        "models": {
            "claude-3-5-haiku": ModelConfig(
                display_name="AWS Bedrock Claude 3.5 Haiku",
                model_identifier="us.anthropic.claude-3-5-haiku-20241022-v1:0",
                max_tokens=8192,
                temperature_range=(0.0, 1.0),
                supports_streaming=True,
                additional_params={"region": "us-east-1"},
            )
        },
    },
}


class ModelManager:
    """Manages multiple model providers and handles model creation"""

    def __init__(self):
        self.providers: Dict[str, Dict[str, Any]] = {}
        self.active_model = None

    def register_provider(self, provider_name: str, api_key: str) -> bool:
        """Register a model provider with credentials"""
        if provider_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown provider: {provider_name}")

        provider_config = MODEL_REGISTRY[provider_name]
        provider_class = provider_config["provider_class"]
        provider_instance = provider_class()

        if provider_instance.validate_credentials(api_key):
            self.providers[provider_name] = {
                "instance": provider_instance,
                "api_key": api_key,
                "models": provider_config["models"],
            }
            return True
        return False

    def get_available_models(self) -> List[Dict[str, str]]:
        """Get list of all available models from registered providers"""
        available_models = []

        for provider_name, provider_info in self.providers.items():
            for model_key, model_config in provider_info["models"].items():
                available_models.append(
                    {
                        "key": f"{provider_name}:{model_key}",
                        "display": model_config.display_name,
                        "provider": provider_name,
                        "model_key": model_key,
                    }
                )

        return available_models

    def create_model(self, model_key: str, **kwargs) -> Any:
        """Create a model instance from a model key (format: provider:model)"""
        if ":" not in model_key:
            raise ValueError(
                f"Invalid model key format: {model_key}. Expected 'provider:model'"
            )

        provider_name, model_name = model_key.split(":", 1)

        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not registered")

        provider_info = self.providers[provider_name]

        if model_name not in provider_info["models"]:
            raise ValueError(
                f"Model {model_name} not available for provider {provider_name}"
            )

        model_config = provider_info["models"][model_name]

        try:
            model_instance = provider_info["instance"].create_model(
                model_config=model_config, api_key=provider_info["api_key"], **kwargs
            )
            self.active_model = model_instance
            return model_instance
        except Exception as e:
            error_msg = provider_info["instance"].handle_error(e)
            raise ModelProviderError(error_msg)

    def is_provider_registered(self, provider_name: str) -> bool:
        """Check if a provider is registered"""
        return provider_name in self.providers

    def get_model_info(self, model_key: str) -> Optional[ModelConfig]:
        """Get model configuration information"""
        if ":" not in model_key:
            return None

        provider_name, model_name = model_key.split(":", 1)

        if provider_name not in self.providers:
            return None

        provider_info = self.providers[provider_name]
        return provider_info["models"].get(model_name)

    def cleanup_credentials(self):
        """Clean up sensitive data from memory"""
        for provider_info in self.providers.values():
            provider_info["api_key"] = ""

        # Clean up environment variables
        if "AWS_BEARER_TOKEN_BEDROCK" in os.environ:
            del os.environ["AWS_BEARER_TOKEN_BEDROCK"]
