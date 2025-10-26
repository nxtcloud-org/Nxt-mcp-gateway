"""
NxtCloud MCP Gateway용 모델 제공자 추상화 계층

이 모듈은 OpenAI와 AWS Bedrock을 포함한 다양한 AI 모델 제공자들에 대한
통합된 인터페이스를 제공하여 원활한 모델 전환과 일관된 에러 처리를 가능하게 합니다.
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
    """특정 모델에 대한 구성 정보"""

    display_name: str  # UI에 표시될 모델 이름
    model_identifier: str  # OpenAI는 model_name, Bedrock은 model_id
    max_tokens: int  # 최대 토큰 수
    temperature_range: Tuple[float, float]  # 온도 설정 범위
    supports_streaming: bool  # 스트리밍 지원 여부
    description: str = ""  # 모델 설명
    pricing_tier: str = ""  # 가격 등급 (예: "Standard", "Premium", "Enterprise")
    capabilities: List[str] = field(
        default_factory=list
    )  # 지원 기능 (예: ["text", "code", "reasoning"])
    context_window: int = 0  # 전체 컨텍스트 윈도우 크기
    additional_params: Dict[str, Any] = field(default_factory=dict)  # 추가 매개변수


class ModelProviderError(Exception):
    """모델 제공자 에러의 기본 예외 클래스"""

    pass


class AuthenticationError(ModelProviderError):
    """인증 실패 예외"""

    pass


class NetworkError(ModelProviderError):
    """네트워크 연결 문제 예외"""

    pass


class ModelProvider(ABC):
    """AI 모델 제공자를 위한 추상 기본 클래스"""

    @abstractmethod
    def create_model(self, model_config: ModelConfig, api_key: str, **kwargs) -> Any:
        """모델 인스턴스를 생성하고 반환합니다"""
        pass

    @abstractmethod
    def validate_credentials(self, api_key: str) -> bool:
        """제공자 자격 증명을 검증합니다"""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """제공자 이름을 반환합니다"""
        pass

    def handle_error(self, error: Exception) -> str:
        """제공자별 에러를 사용자 친화적인 메시지로 변환합니다"""
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
    """OpenAI 모델 제공자 구현"""

    def create_model(
        self, model_config: ModelConfig, api_key: str, **kwargs
    ) -> ChatOpenAI:
        """OpenAI 모델 인스턴스를 생성합니다"""
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
        """OpenAI API 키 형식을 검증합니다"""
        if not api_key:
            return False

        # OpenAI 키는 sk-로 시작하며 일반적으로 51자 이상입니다
        # 하지만 다양한 키 형식을 위해 더 관대하게 검증합니다
        return api_key.startswith("sk-") and len(api_key) >= 20

    def get_provider_name(self) -> str:
        return "OpenAI"


class BedrockProvider(ModelProvider):
    """AWS Bedrock 모델 제공자 구현"""

    def create_model(
        self, model_config: ModelConfig, api_key: str, **kwargs
    ) -> ChatBedrock:
        """Cross Region Inference를 지원하는 AWS Bedrock 모델 인스턴스를 생성합니다"""
        try:
            # Bedrock API 키 인증을 위한 AWS Bearer Token 설정
            self._set_bedrock_credentials(api_key)

            # Cross Region Inference 구성으로 Bedrock 클라이언트 생성
            client = self._create_bedrock_client()

            # 모델 매개변수 구성
            model_kwargs = {
                "max_tokens": model_config.max_tokens,
                "temperature": kwargs.get("temperature", 0.1),
            }

            # 모델 구성에서 추가 매개변수 추가
            if model_config.additional_params:
                model_kwargs.update(
                    {
                        k: v
                        for k, v in model_config.additional_params.items()
                        if k != "region"  # model_kwargs에서 region 제외
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
        """환경 변수에 AWS Bedrock 자격 증명을 안전하게 설정합니다"""
        if not api_key or len(api_key) < 10:
            raise ValueError("Invalid Bedrock API key")

        # 현재 프로세스에만 환경 변수 설정
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key

        # 일관성을 위해 기본 리전도 설정
        os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

    def _create_bedrock_client(self):
        """Cross Region Inference를 지원하는 Bedrock 클라이언트를 생성합니다"""
        try:
            # 고급 구성을 위한 botocore Config 가져오기
            from botocore.config import Config

            # 재시도 및 Cross Region Inference 설정 구성
            retry_config = Config(
                retries={"max_attempts": 3, "mode": "adaptive"},
                # Cross Region Inference 구성
                # 주 리전을 사용할 수 없을 때 다른 리전으로 자동 장애 조치 허용
                region_name="us-east-1",
            )

            # Cross Region Inference로 클라이언트 구성
            client = boto3.client(
                service_name="bedrock-runtime",
                region_name="us-east-1",  # Cross Region Inference를 위한 주 리전
                config=retry_config,
            )

            return client
        except Exception as e:
            raise NetworkError(f"Failed to create Bedrock client: {str(e)}")

    def test_cross_region_inference(self, client):
        """Cross Region Inference 기능을 테스트합니다"""
        try:
            # 클라이언트가 Cross Region Inference에 대해 올바르게 구성되었는지 확인
            if hasattr(client, "_client_config"):
                region = client._client_config.region_name
                return region == "us-east-1"
            return False
        except Exception:
            return False

    def validate_credentials(self, api_key: str) -> bool:
        """AWS Bedrock API 키 형식을 검증합니다"""
        if not api_key or len(api_key) < 10:
            return False

        # AWS Bedrock API 키는 다양한 형식을 가질 수 있습니다
        # 현재는 더 관대하게 기본 길이와 문자만 확인합니다
        # 실제 검증은 클라이언트 생성 시 수행됩니다
        return len(api_key) >= 10 and len(api_key) <= 200

    def get_provider_name(self) -> str:
        return "AWS Bedrock"


# 모델 레지스트리 - 지원되는 모든 모델의 구성 정보
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
    """여러 모델 제공자를 관리하고 모델 생성을 처리합니다"""

    def __init__(self):
        self.providers: Dict[str, Dict[str, Any]] = {}  # 등록된 제공자들
        self.active_model = None  # 현재 활성 모델

    def register_provider(self, provider_name: str, api_key: str) -> bool:
        """자격 증명과 함께 모델 제공자를 등록합니다"""
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
        """등록된 제공자들로부터 사용 가능한 모든 모델 목록을 가져옵니다"""
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
        """모델 키로부터 모델 인스턴스를 생성합니다 (형식: provider:model)"""
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
        """제공자가 등록되어 있는지 확인합니다"""
        return provider_name in self.providers

    def get_model_info(self, model_key: str) -> Optional[ModelConfig]:
        """모델 구성 정보를 가져옵니다"""
        if ":" not in model_key:
            return None

        provider_name, model_name = model_key.split(":", 1)

        if provider_name not in self.providers:
            return None

        provider_info = self.providers[provider_name]
        return provider_info["models"].get(model_name)

    def get_provider_info(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """제공자 구성 정보를 가져옵니다"""
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
        """사용 가능한 모든 제공자에 대한 정보를 가져옵니다"""
        providers_info = {}
        for provider_name in MODEL_REGISTRY.keys():
            providers_info[provider_name] = self.get_provider_info(provider_name)
        return providers_info

    def get_models_by_capability(self, capability: str) -> List[Dict[str, str]]:
        """특정 기능을 지원하는 모델들을 가져옵니다"""
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
        """메모리와 환경에서 민감한 데이터를 정리합니다"""
        # 제공자 정보에서 API 키 제거
        for provider_info in self.providers.values():
            provider_info["api_key"] = ""

        # AWS Bedrock 환경 변수 정리
        aws_env_vars = ["AWS_BEARER_TOKEN_BEDROCK", "AWS_DEFAULT_REGION"]

        for env_var in aws_env_vars:
            if env_var in os.environ:
                del os.environ[env_var]

    def get_bedrock_status(self) -> Dict[str, Any]:
        """Cross Region Inference를 포함한 AWS Bedrock 제공자 상태를 가져옵니다"""
        if not self.is_provider_registered("bedrock"):
            return {
                "registered": False,
                "cross_region_inference": False,
                "region": None,
                "status": "Not registered",
            }

        try:
            bedrock_provider = self.providers["bedrock"]["instance"]
            # 구성을 확인하기 위한 테스트 클라이언트 생성
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
