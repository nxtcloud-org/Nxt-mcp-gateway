# NxtCloud MCP Gateway

Streamlit 기반의 MCP (Model Context Protocol) 게이트웨이 애플리케이션입니다.

## MCP 아키텍처

이 프로젝트에서의 MCP 구성 요소:

### 🏠 **MCP Host**

- **NxtCloud MCP Gateway** (이 애플리케이션)
- Streamlit 기반 웹 인터페이스
- 사용자와 AI 모델 간의 상호작용 관리
- MCP 서버들을 통합하여 AI 에이전트에게 도구 제공

### 🔌 **MCP Client**

- **LangChain MCP Adapters** (`langchain-mcp-adapters`)
- MCP 서버들과의 통신을 담당
- 여러 MCP 서버를 동시에 관리하는 `MultiServerMCPClient` 사용
- LangGraph ReAct 에이전트에게 도구를 제공

### 🛠️ **MCP Servers**

- **내장 서버**: `mcp_servers/time.py`, `mcp_servers/fitness.py`
- **외부 서버**: 사용자가 추가하는 다양한 MCP 서버들
- 각 서버는 특정 기능을 제공하는 도구들을 포함
- stdio 또는 SSE 프로토콜을 통해 통신

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   MCP Host      │    │   MCP Client     │    │   MCP Servers   │
│                 │    │                  │    │                 │
│ NxtCloud        │◄──►│ LangChain        │◄──►│ time.py         │
│ MCP Gateway     │    │ MCP Adapters     │    │ fitness.py      │
│ (Streamlit)     │    │                  │    │ external...     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 기능

- 🤖 **AI 모델 지원**: OpenAI GPT-4o/GPT-4o-mini 모델
- 🔧 **MCP 서버 관리**: 동적 추가/삭제/설정 (JSON 기반)
- 💬 **실시간 채팅**: 스트리밍 인터페이스 및 도구 호출 시각화
- 🛠️ **내장 MCP 서버**: 시간 계산기, 헬스 계산기
- 📊 **세션 관리**: 대화 기록 및 스레드 관리
- 🔌 **MCP 클라이언트**: [LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters) 활용

## 설치 및 실행

1. 가상환경 생성 및 활성화

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. 의존성 설치

```bash
pip install -r requirements.txt
```

3. 환경변수 설정

```bash
cp .env.example .env
# .env 파일을 열어서 OPENAI_API_KEY를 설정하세요
```

4. 애플리케이션 실행

```bash
streamlit run app.py
```

## 환경변수

- `OPENAI_API_KEY`: OpenAI API 키 (필수)
- `TIMEOUT_SECONDS`: 요청 타임아웃 (기본값: 120초)
- `RECURSION_LIMIT`: 재귀 제한 (기본값: 100)
