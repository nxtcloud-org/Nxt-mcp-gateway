# NxtCloud MCP Gateway

Streamlit 기반의 MCP (Model Context Protocol) 게이트웨이 애플리케이션입니다.

## 기능

- 🤖 OpenAI GPT-4o/GPT-4o-mini 모델 지원
- 🔧 동적 MCP 서버 관리 (추가/삭제/설정)
- 💬 실시간 스트리밍 채팅 인터페이스
- 🛠️ 내장 MCP 서버: 시간 계산기, 헬스 계산기
- 📊 도구 호출 시각화 및 세션 관리

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
