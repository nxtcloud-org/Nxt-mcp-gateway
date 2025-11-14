# NxtCloud MCP Gateway

다중 AI 모델 제공자를 지원하는 MCP (Model Context Protocol) 게이트웨이 애플리케이션입니다.

## ✨ 주요 특징

- 🤖 **다중 AI 모델 지원**: OpenAI GPT-4o Mini & AWS Bedrock Claude 3.5 Haiku
- ☁️ **AWS Cross Region Inference**: 안정적인 Bedrock 서비스 제공
- 🔧 **MCP 서버 통합**: 동적 도구 추가/관리 (JSON 기반)
- 💬 **실시간 스트리밍**: 응답 및 도구 호출 시각화
- 🛠️ **내장 MCP 서버**: 시간 계산기, 헬스 계산기
- 🔄 **원활한 모델 전환**: 대화 중 모델 변경 가능

## 🏗️ 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI Layer                       │
├─────────────────────────────────────────────────────────────┤
│  Model Settings Tab  │  Chat Tab  │  MCP Tools Tab         │
├─────────────────────────────────────────────────────────────┤
│                  Model Management Layer                     │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │ OpenAI Provider │    │    AWS Bedrock Provider        │ │
│  │ - GPT-4o Mini   │    │ - Claude 3.5 Haiku            │ │
│  │ - API Key Auth  │    │ - Cross Region Inference       │ │
│  └─────────────────┘    └─────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    LangGraph Agent Layer                    │
│              create_react_agent + MCP Tools                 │
├─────────────────────────────────────────────────────────────┤
│                      MCP Client Layer                       │
│                MultiServerMCPClient                         │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 빠른 시작

### 방법 1: Docker 사용 (권장) 🐳

```bash
# Docker Hub에서 이미지 다운로드 및 실행
docker run -p 8501:8501 glen15/nxt-mcp-gateway:latest
```

브라우저에서 `http://localhost:8501` 접속

### 방법 2: Docker Compose 사용

```bash
# 저장소 클론
git clone <repository-url>
cd nxt-mcp-gateway

# Docker Compose로 실행
docker-compose up -d
```

### 방법 3: 로컬 개발 환경

```bash
# 저장소 클론
git clone <repository-url>
cd nxt-mcp-gateway

# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 실행
streamlit run app.py
```

### 3. 설정

1. **모델 설정 탭**에서 원하는 AI 제공자의 API 키 입력:

   - ☁️ **AWS Bedrock**: Claude 3.5 Haiku 모델 사용
   - 🤖 **OpenAI**: GPT-4o Mini 모델 사용
   - 💡 **둘 중 하나만 설정해도 사용 가능**

2. **MCP 도구 탭**에서 "설정 적용하기" 클릭

3. **챗봇 탭**에서 대화 시작!

## 🔧 지원 모델

| 제공자      | 모델             | 설명                             |
| ----------- | ---------------- | -------------------------------- |
| AWS Bedrock | Claude 3.5 Haiku | Anthropic의 빠르고 효율적인 모델 |
| OpenAI      | GPT-4o Mini      | OpenAI의 경량 고성능 모델        |

## 📋 MCP 서버 관리

### 환경별 도구 관리 🐳💻

이 애플리케이션은 **컨테이너 환경**과 **로컬 환경**을 자동으로 구분하여 호환 가능한 MCP 도구만 로드합니다.

#### 환경 감지

- **컨테이너 환경** (🐳): `IS_CONTAINER=true` 환경 변수로 감지
- **로컬 환경** (💻): 기본값, 모든 도구 사용 가능

#### 도구 호환성

| 도구               | 컨테이너 | 로컬 | 기본 활성화 | 설명                                                |
| ------------------ | -------- | ---- | ----------- | --------------------------------------------------- |
| `get_current_time` | ✅       | ✅   | ✅          | 시간 관련 계산                                      |
| `playwright-mcp`   | ❌       | ✅   | ✅          | 브라우저 자동화 (컨테이너에서 브라우저 프로필 충돌) |
| `weather`          | ✅       | ✅   | ❌          | OpenWeatherMap 날씨 조회 (수동 추가 필요)           |

### 기본 활성화 서버

- `mcp_servers/time.py`: 시간 관련 계산 (자동 로드)

### 수동 추가 서버

- `mcp_servers/weather.py`: OpenWeatherMap 날씨 조회 (MCP 도구 탭에서 추가)

### 외부 서버 추가

MCP 도구 탭에서 JSON 형식으로 새 서버 추가:

**예시 1: NPM 패키지 서버**

```json
{
  "server-name": {
    "command": "npx",
    "args": ["-y", "package-name"],
    "transport": "stdio"
  }
}
```

**예시 2: Python 날씨 서버 (OpenWeatherMap)**

```json
{
  "weather": {
    "command": "python",
    "args": ["./mcp_servers/weather.py"],
    "transport": "stdio"
  }
}
```

### 새 MCP 도구에 메타데이터 추가하기

`app.py`의 `MCP_TOOLS_METADATA`에 도구 정보 등록:

```python
MCP_TOOLS_METADATA = {
    "your-tool-name": {
        "container_compatible": True,  # 컨테이너에서 사용 가능 여부
        "description": "도구 설명",
        "category": "카테고리",
        "note": "추가 정보 (선택)",
    },
}
```

#### 메타데이터 필드

- `container_compatible` (bool): 컨테이너 환경 호환 여부
  - `True`: 로컬/컨테이너 모두 사용 가능
  - `False`: 로컬 환경 전용 (컨테이너에서 자동 제외)
- `description` (str): 도구 설명 (UI에 표시)
- `category` (str): 도구 분류 (UI에 표시)
- `note` (str): 추가 정보/경고 (선택사항)

## 🔒 보안

- API 키는 세션 상태에만 저장 (파일 저장 안함)
- UI에서 API 키 마스킹 처리
- 세션 종료 시 자동 정리

## 📦 의존성

핵심 패키지 (9개):

- `streamlit`: 웹 인터페이스
- `langchain-openai`: OpenAI 통합
- `langchain-aws`: AWS Bedrock 통합
- `langgraph`: AI 에이전트 프레임워크
- `langchain-mcp-adapters`: MCP 클라이언트
- `boto3`: AWS SDK
- `mcp`: MCP 프로토콜
- `nest-asyncio`: 비동기 처리
- `pytz`: 시간대 처리

## 🐳 Docker 사용법

### 이미지 빌드

```bash
# 로컬에서 이미지 빌드
docker build -t glen15/nxt-mcp-gateway:latest .

# 또는 Docker Hub 업로드 스크립트 사용
./push-to-dockerhub.sh glen15
```

### 실행 옵션

```bash
# 기본 실행 (설정은 컨테이너 내부에만 저장, 컨테이너 삭제 시 사라짐)
docker run -p 8501:8501 glen15/nxt-mcp-gateway:latest

# 백그라운드 실행
docker run -d -p 8501:8501 --name nxt-mcp-gateway glen15/nxt-mcp-gateway:latest

# MCP 설정 파일 마운트 (권장: 설정이 호스트에 영구 저장됨)
docker run -p 8501:8501 -v $(pwd)/mcp_config.json:/app/mcp_config.json glen15/nxt-mcp-gateway:latest

# 백그라운드에서 실행 (권장)
docker run -d -p 8501:8501 \
  --name nxt-mcp-gateway \
  -v $(pwd)/mcp_config.json:/app/mcp_config.json \
  glen15/nxt-mcp-gateway:latest
```

⚠️ **중요**: Docker 실행 시 설정 파일을 볼륨으로 마운트하지 않으면:

- UI에서 변경한 MCP 서버 설정이 컨테이너 내부 파일에만 저장됩니다
- 컨테이너를 삭제하면 설정이 사라집니다
- 여러 컨테이너 인스턴스 간 설정이 공유되지 않습니다

✅ **권장**: 위의 볼륨 마운트 예제처럼 설정 파일을 마운트하여 사용하세요.

### Docker Hub 배포

```bash
# Docker Hub에 로그인
docker login

# 이미지 푸시
docker push glen15/nxt-mcp-gateway:latest
```

## 🛠️ 개발

### 프로젝트 구조

```
├── app.py                 # 메인 Streamlit 애플리케이션
├── model_providers.py     # 모델 제공자 추상화 계층
├── utils.py              # 유틸리티 함수
├── mcp_servers/          # 내장 MCP 서버들
├── requirements.txt      # Python 의존성
├── Dockerfile            # Docker 이미지 빌드 파일
├── docker-compose.yml    # Docker Compose 설정
├── build.sh             # Docker 빌드 스크립트
└── README.md            # 프로젝트 문서
```

### 새 모델 제공자 추가

1. `model_providers.py`에서 `ModelProvider` 클래스 상속
2. `MODEL_REGISTRY`에 새 제공자 등록
3. UI에서 자동으로 사용 가능

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.
