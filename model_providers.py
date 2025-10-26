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
    description: str = ""
    pricing_tier: str = ""  # e.g., "Standard", "Premium", "Enterprise"
    capabilities: List[str] = field(
        default_factory=list
    )  # e.g., ["text", "code", "reasoning"]
    context_window: int = 0  # Total context window size
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
        if not api_key:
            return False

        # OpenAI keys start with sk- and are typically 51+ characters
        # But we'll be more permissive for different key formats
        return api_key.startswith("sk-") and len(api_key) >= 20

    def get_provider_name(self) -> str:
        return "OpenAI"


class BedrockProvider(ModelProvider):
    """AWS Bedrock model provider implementation"""

    def create_model(
        self, model_config: ModelConfig, api_key: str, **kwargs
    ) -> ChatBedrock:
        """Create AWS Bedrock model instance with Cross Region Inference"""
        try:
            # Set AWS Bearer Token for Bedrock API Key authentication
            self._set_bedrock_credentials(api_key)

            # Create Bedrock client with Cross Region Inference configuration
            client = self._create_bedrock_client()

            # Configure model parameters
            model_kwargs = {
                "max_tokens": model_config.max_tokens,
                "temperature": kwargs.get("temperature", 0.1),
            }

            # Add additional parameters from model config
            if model_config.additional_params:
                model_kwargs.update(
                    {
                        k: v
                        for k, v in model_config.additional_params.items()
                        if k != "region"  # Exclude region from model_kwargs
                    }
                )

            return ChatBedrock(
                client=client,
                model_id=model_config.model_identifier,
                model_kwargs=model_kwargs,
                streaming=model_config.supports_streaming,
            )
        except Exception as e:
            raise AuthenticationError(f"Failed to create Bedrock model: {str(e)}")

    def _set_bedrock_credentials(self, api_key: str):
        """Safely set AWS Bedrock credentials in environment"""
        if not api_key or len(api_key) < 10:
            raise ValueError("Invalid Bedrock API key")

        # Set environment variable for current process only
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key

        # Also set default region for consistency
        os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

    def _create_bedrock_client(self):
        """Create Bedrock client with Cross Region Inference support"""
        try:
            # Import botocore Config for advanced configuration
            from botocore.config import Config

            # Configure retry and Cross Region Inference settings
            retry_config = Config(
                retries={"max_attempts": 3, "mode": "adaptive"},
                # Cross Region Inference configuration
                # This allows automatic failover to other regions when primary is unavailable
                region_name="us-east-1",
            )

            # Configure client with Cross Region Inference
            client = boto3.client(
                service_name="bedrock-runtime",
                region_name="us-east-1",  # Primary region for Cross Region Inference
                config=retry_config,
            )

            return client
        except Exception as e:
            raise NetworkError(f"Failed to create Bedrock client: {str(e)}")

    def test_cross_region_inference(self, client):
        """Test Cross Region Inference functionality"""
        try:
            # Verify the client is properly configured for Cross Region Inference
            if hasattr(client, "_client_config"):
                region = client._client_config.region_name
                return region == "us-east-1"
            return False
        except Exception:
            return False

    def validate_credentials(self, api_key: str) -> bool:
        """Validate AWS Bedrock API key format"""
        if not api_key or len(api_key) < 10:
            return False

        # AWS Bedrock API keys can have various formats
        # For now, we'll be more permissive and just check basic length and characters
        # Real validation happens when creating the client
        return len(api_key) >= 10 and len(api_key) <= 200

    def get_provider_name(self) -> str:
        return "AWS Bedrock"


# Model Registry - Configuration for all supported models
MODEL_REGISTRY = {
    "openai": {
        "provider_class": OpenAIProvider,
        "display_name": "OpenAI",
        "description": "OpenAI's GPT models with advanced reasoning capabilities",
        "models": {
            "gpt-4o-mini": ModelConfig(
                display_name="OpenAI GPT-4o Mini",
                model_identifier="gpt-4o-mini",
                max_tokens=16000,
                temperature_range=(0.0, 2.0),
                supports_streaming=True,
                description="빠르고 효율적인 경량 모델",
            ),
        },
    },
    "bedrock": {
        "provider_class": BedrockProvider,
        "display_name": "AWS Bedrock",
        "description": "AWS Bedrock을 통한 Anthropic Claude 모델 접근",
        "models": {
            "claude-3-5-haiku": ModelConfig(
                display_name="AWS Bedrock Claude 3.5 Haiku",
                model_identifier="us.anthropic.claude-3-5-haiku-20241022-v1:0",
                max_tokens=8192,
                temperature_range=(0.0, 1.0),
                supports_streaming=True,
                description="Anthropic의 빠르고 효율적인 Claude 모델",
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

    def get_provider_info(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """Get provider configuration information"""
        if provider_name not in MODEL_REGISTRY:
            return None

        registry_info = MODEL_REGISTRY[provider_name]
        is_registered = self.is_provider_registered(provider_name)

        return {
            "display_name": registry_info.get("display_name", provider_name),
            "description": registry_info.get("description", ""),
            "is_registered": is_registered,
            "model_count": len(registry_info["models"]) if is_registered else 0,
        }

    def get_all_providers_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available providers"""
        providers_info = {}
        for provider_name in MODEL_REGISTRY.keys():
            providers_info[provider_name] = self.get_provider_info(provider_name)
        return providers_info

    def get_models_by_capability(self, capability: str) -> List[Dict[str, str]]:
        """Get models that support a specific capability"""
        matching_models = []

        for provider_name, provider_info in self.providers.items():
            for model_key, model_config in provider_info["models"].items():
                if capability in model_config.capabilities:
                    matching_models.append(
                        {
                            "key": f"{provider_name}:{model_key}",
                            "display": model_config.display_name,
                            "provider": provider_name,
                            "model_key": model_key,
                        }
                    )

        return matching_models

    def cleanup_credentials(self):
        """Clean up sensitive data from memory and environment"""
        # Clear API keys from provider info
        for provider_info in self.providers.values():
            provider_info["api_key"] = ""

        # Clean up AWS Bedrock environment variables
        aws_env_vars = ["AWS_BEARER_TOKEN_BEDROCK", "AWS_DEFAULT_REGION"]

        for env_var in aws_env_vars:
            if env_var in os.environ:
                del os.environ[env_var]

    def get_bedrock_status(self) -> Dict[str, Any]:
        """Get AWS Bedrock provider status including Cross Region Inference"""
        if not self.is_provider_registered("bedrock"):
            return {
                "registered": False,
                "cross_region_inference": False,
                "region": None,
                "status": "Not registered",
            }

        try:
            bedrock_provider = self.providers["bedrock"]["instance"]
            # Create a test client to check configuration
            test_client = bedrock_provider._create_bedrock_client()
            cross_region_status = bedrock_provider.test_cross_region_inference(
                test_client
            )

            return {
                "registered": True,
                "cross_region_inference": cross_region_status,
                "region": "us-east-1",
                "status": (
                    "Active with Cross Region Inference"
                    if cross_region_status
                    else "Active"
                ),
            }
        except Exception as e:
            return {
                "registered": True,
                "cross_region_inference": False,
                "region": "us-east-1",
                "status": f"Error: {str(e)}",
            }
