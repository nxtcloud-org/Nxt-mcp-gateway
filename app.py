import streamlit as st
import asyncio
import nest_asyncio
import json
import os
import platform

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# nest_asyncio 적용: 이미 실행 중인 이벤트 루프 내에서 중첩 호출 허용
nest_asyncio.apply()

# 전역 이벤트 루프 생성 및 재사용 (한번 생성한 후 계속 사용)
if "event_loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    st.session_state.event_loop = loop
    asyncio.set_event_loop(loop)

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from utils import astream_graph, random_uuid
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from model_providers import ModelManager, ModelProviderError

# MCP 설정 파일 경로 설정
CONFIG_FILE_PATH = "mcp_config.json"


# JSON 설정 파일 로드 함수
def load_config_from_json():
    """
    config.json 파일에서 설정을 로드합니다.
    파일이 없는 경우 기본 설정으로 파일을 생성합니다.

    반환값:
        dict: 로드된 설정
    """
    default_config = {
        "get_current_time": {
            "command": "python",
            "args": ["./mcp_servers/time.py"],
            "transport": "stdio",
        },
        "fitness_calculator": {
            "command": "python",
            "args": ["./mcp_servers/fitness.py"],
            "transport": "stdio",
        },
    }

    try:
        if os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            # 파일이 없는 경우 기본 설정으로 파일 생성
            save_config_to_json(default_config)
            return default_config
    except Exception as e:
        st.error(f"설정 파일 로드 중 오류 발생: {str(e)}")
        return default_config


# JSON 설정 파일 저장 함수
def save_config_to_json(config):
    """
    설정을 config.json 파일에 저장합니다.

    매개변수:
        config (dict): 저장할 설정

    반환값:
        bool: 저장 성공 여부
    """
    try:
        with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"설정 파일 저장 중 오류 발생: {str(e)}")
        return False


# 페이지 설정
st.set_page_config(page_title="NxtCloud MCP Gateway", page_icon="🚀", layout="wide")


# 기존 페이지 타이틀 및 설명
st.title("💬 NxtCloud MCP Gateway")
st.markdown("✨ MCP 도구를 활용한 AI 에이전트 게이트웨이입니다.")

# 탭 생성
tab1, tab2, tab3 = st.tabs(["🤖 챗봇", "🤖 모델 설정", "🔧 MCP 도구"])

# 탭 컨테이너
chat_container = tab1
model_container = tab2
mcp_container = tab3

SYSTEM_PROMPT = """<ROLE>
You are a smart agent with an ability to use tools. 
You will be given a question and you will use the tools to answer the question.
Pick the most relevant tool to answer the question. 
If you are failed to answer the question, try different tools to get context.
Your answer should be helpful and appropriate to the context.
</ROLE>

----

<INSTRUCTIONS>
Step 1: Analyze the question
- Analyze user's question and final goal.
- If the user's question is consist of multiple sub-questions, split them into smaller sub-questions.

Step 2: Pick the most relevant tool
- Pick the most relevant tool to answer the question.
- If you are failed to answer the question, try different tools to get context.

Step 3: Answer the question
- Answer the question in the same language as the question.
- Use the tool's output as the primary source of information.
- If the tool provides rich formatting (emojis, markdown, personality), preserve and use it in your response.
- For simple data tools, you may summarize or present the information clearly.
- For personality-rich tools (like fitness calculator), include the full formatted output to preserve the experience.

Step 4: Provide the source of the answer(if applicable)
- If you've used the tool, provide the source of the answer.
- Valid sources are either a website(URL) or a document(PDF, etc).

Guidelines:
- The tool's output is more important than your own knowledge.
- Preserve formatting, emojis, and personality when the tool provides them.
- For technical/data tools, present information clearly and professionally.
- For entertainment/personality tools, maintain their character and style.
- Answer in the same language as the question.
- Be helpful and contextually appropriate.
</INSTRUCTIONS>

----

<OUTPUT_FORMAT>
(Appropriate response based on tool output - preserve personality for character tools, be clear for data tools)

**Source**(if applicable)
- (source1: valid URL)
- (source2: valid URL)
- ...
</OUTPUT_FORMAT>
"""

# OUTPUT_TOKEN_INFO는 이제 ModelManager에서 관리되므로 제거
# 모델별 토큰 정보는 model_providers.py의 ModelConfig에서 관리됨

# 시스템 설정
TIMEOUT_SECONDS = 120
RECURSION_LIMIT = 100

# 세션 상태 초기화
if "session_initialized" not in st.session_state:
    st.session_state.session_initialized = False  # 세션 초기화 상태 플래그
    st.session_state.agent = None  # ReAct 에이전트 객체 저장 공간
    st.session_state.history = []  # 대화 기록 저장 리스트
    st.session_state.mcp_client = None  # MCP 클라이언트 객체 저장 공간
    st.session_state.selected_model = (
        "openai:gpt-4o-mini"  # 기본 모델 선택 (provider:model 형식)
    )
    st.session_state.model_manager = ModelManager()  # 모델 매니저 인스턴스

if "thread_id" not in st.session_state:
    st.session_state.thread_id = random_uuid()


# --- 함수 정의 부분 ---


async def cleanup_mcp_client():
    """
    기존 MCP 클라이언트를 안전하게 종료합니다.

    기존 클라이언트가 있는 경우 정상적으로 리소스를 해제합니다.
    """
    if "mcp_client" in st.session_state and st.session_state.mcp_client is not None:
        try:
            # 새로운 API에서는 별도의 종료 메서드가 필요하지 않음
            st.session_state.mcp_client = None
        except Exception as e:
            import traceback

            # st.warning(f"MCP 클라이언트 종료 중 오류: {str(e)}")
            # st.warning(traceback.format_exc())


def print_message():
    """
    채팅 기록을 화면에 출력합니다.

    사용자와 어시스턴트의 메시지를 구분하여 화면에 표시하고,
    도구 호출 정보는 어시스턴트 메시지 컨테이너 내에 표시합니다.
    """
    i = 0
    while i < len(st.session_state.history):
        message = st.session_state.history[i]

        if message["role"] == "user":
            st.chat_message("user", avatar="🧑‍💻").markdown(message["content"])
            i += 1
        elif message["role"] == "assistant":
            # 어시스턴트 메시지 컨테이너 생성
            with st.chat_message("assistant", avatar="🤖"):
                # 어시스턴트 메시지 내용 표시
                st.markdown(message["content"])

                # 다음 메시지가 도구 호출 정보인지 확인
                if (
                    i + 1 < len(st.session_state.history)
                    and st.session_state.history[i + 1]["role"] == "assistant_tool"
                ):
                    # 도구 호출 정보를 동일한 컨테이너 내에 expander로 표시
                    with st.expander("🔧 도구 호출 정보", expanded=False):
                        st.markdown(st.session_state.history[i + 1]["content"])
                    i += 2  # 두 메시지를 함께 처리했으므로 2 증가
                else:
                    i += 1  # 일반 메시지만 처리했으므로 1 증가
        else:
            # assistant_tool 메시지는 위에서 처리되므로 건너뜀
            i += 1


def get_streaming_callback(text_placeholder, tool_placeholder):
    """
    스트리밍 콜백 함수를 생성합니다.

    이 함수는 LLM에서 생성되는 응답을 실시간으로 화면에 표시하기 위한 콜백 함수를 생성합니다.
    텍스트 응답과 도구 호출 정보를 각각 다른 영역에 표시합니다.

    매개변수:
        text_placeholder: 텍스트 응답을 표시할 Streamlit 컴포넌트
        tool_placeholder: 도구 호출 정보를 표시할 Streamlit 컴포넌트

    반환값:
        callback_func: 스트리밍 콜백 함수
        accumulated_text: 누적된 텍스트 응답을 저장하는 리스트
        accumulated_tool: 누적된 도구 호출 정보를 저장하는 리스트
    """
    accumulated_text = []
    accumulated_tool = []

    def callback_func(message: dict):
        nonlocal accumulated_text, accumulated_tool
        message_content = message.get("content", None)

        if isinstance(message_content, AIMessageChunk):
            content = message_content.content
            # 콘텐츠가 리스트 형태인 경우 (Claude 모델 등에서 주로 발생)
            if isinstance(content, list) and len(content) > 0:
                message_chunk = content[0]
                # 텍스트 타입인 경우 처리
                if message_chunk["type"] == "text":
                    accumulated_text.append(message_chunk["text"])
                    text_placeholder.markdown("".join(accumulated_text))
                # 도구 사용 타입인 경우 처리
                elif message_chunk["type"] == "tool_use":
                    if "partial_json" in message_chunk:
                        accumulated_tool.append(message_chunk["partial_json"])
                    else:
                        tool_call_chunks = message_content.tool_call_chunks
                        tool_call_chunk = tool_call_chunks[0]
                        accumulated_tool.append(
                            "\n```json\n" + str(tool_call_chunk) + "\n```\n"
                        )
                    with tool_placeholder.expander("🔧 도구 호출 정보", expanded=True):
                        st.markdown("".join(accumulated_tool))
            # tool_calls 속성이 있는 경우 처리 (OpenAI 모델 등에서 주로 발생)
            elif (
                hasattr(message_content, "tool_calls")
                and message_content.tool_calls
                and len(message_content.tool_calls[0]["name"]) > 0
            ):
                tool_call_info = message_content.tool_calls[0]
                accumulated_tool.append("\n```json\n" + str(tool_call_info) + "\n```\n")
                with tool_placeholder.expander("🔧 도구 호출 정보", expanded=True):
                    st.markdown("".join(accumulated_tool))
            # 단순 문자열인 경우 처리
            elif isinstance(content, str):
                accumulated_text.append(content)
                text_placeholder.markdown("".join(accumulated_text))
            # 유효하지 않은 도구 호출 정보가 있는 경우 처리
            elif (
                hasattr(message_content, "invalid_tool_calls")
                and message_content.invalid_tool_calls
            ):
                tool_call_info = message_content.invalid_tool_calls[0]
                accumulated_tool.append("\n```json\n" + str(tool_call_info) + "\n```\n")
                with tool_placeholder.expander(
                    "🔧 도구 호출 정보 (유효하지 않음)", expanded=True
                ):
                    st.markdown("".join(accumulated_tool))
            # tool_call_chunks 속성이 있는 경우 처리
            elif (
                hasattr(message_content, "tool_call_chunks")
                and message_content.tool_call_chunks
            ):
                tool_call_chunk = message_content.tool_call_chunks[0]
                accumulated_tool.append(
                    "\n```json\n" + str(tool_call_chunk) + "\n```\n"
                )
                with tool_placeholder.expander("🔧 도구 호출 정보", expanded=True):
                    st.markdown("".join(accumulated_tool))
            # additional_kwargs에 tool_calls가 있는 경우 처리 (다양한 모델 호환성 지원)
            elif (
                hasattr(message_content, "additional_kwargs")
                and "tool_calls" in message_content.additional_kwargs
            ):
                tool_call_info = message_content.additional_kwargs["tool_calls"][0]
                accumulated_tool.append("\n```json\n" + str(tool_call_info) + "\n```\n")
                with tool_placeholder.expander("🔧 도구 호출 정보", expanded=True):
                    st.markdown("".join(accumulated_tool))
        # 도구 메시지인 경우 처리 (도구의 응답)
        elif isinstance(message_content, ToolMessage):
            accumulated_tool.append(
                "\n```json\n" + str(message_content.content) + "\n```\n"
            )
            with tool_placeholder.expander("🔧 도구 호출 정보", expanded=True):
                st.markdown("".join(accumulated_tool))
        return None

    return callback_func, accumulated_text, accumulated_tool


async def process_query(query, text_placeholder, tool_placeholder, timeout_seconds=60):
    """
    사용자 질문을 처리하고 응답을 생성합니다.

    이 함수는 사용자의 질문을 에이전트에 전달하고, 응답을 실시간으로 스트리밍하여 표시합니다.
    지정된 시간 내에 응답이 완료되지 않으면 타임아웃 오류를 반환합니다.

    매개변수:
        query: 사용자가 입력한 질문 텍스트
        text_placeholder: 텍스트 응답을 표시할 Streamlit 컴포넌트
        tool_placeholder: 도구 호출 정보를 표시할 Streamlit 컴포넌트
        timeout_seconds: 응답 생성 제한 시간(초)

    반환값:
        response: 에이전트의 응답 객체
        final_text: 최종 텍스트 응답
        final_tool: 최종 도구 호출 정보
    """
    try:
        if st.session_state.agent:
            streaming_callback, accumulated_text_obj, accumulated_tool_obj = (
                get_streaming_callback(text_placeholder, tool_placeholder)
            )
            try:
                response = await asyncio.wait_for(
                    astream_graph(
                        st.session_state.agent,
                        {"messages": [HumanMessage(content=query)]},
                        callback=streaming_callback,
                        config=RunnableConfig(
                            recursion_limit=RECURSION_LIMIT,
                            thread_id=st.session_state.thread_id,
                        ),
                    ),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                error_msg = f"⏱️ 요청 시간이 {timeout_seconds}초를 초과했습니다. 나중에 다시 시도해 주세요."
                return {"error": error_msg}, error_msg, ""

            final_text = "".join(accumulated_text_obj)
            final_tool = "".join(accumulated_tool_obj)
            return response, final_text, final_tool
        else:
            return (
                {"error": "🚫 에이전트가 초기화되지 않았습니다."},
                "🚫 에이전트가 초기화되지 않았습니다.",
                "",
            )
    except Exception as e:
        import traceback

        error_msg = f"❌ 쿼리 처리 중 오류 발생: {str(e)}\n{traceback.format_exc()}"
        return {"error": error_msg}, error_msg, ""


async def initialize_session(mcp_config=None):
    """
    MCP 세션과 에이전트를 초기화합니다.

    매개변수:
        mcp_config: MCP 도구 설정 정보(JSON). None인 경우 기본 설정 사용

    반환값:
        bool: 초기화 성공 여부
    """
    with st.spinner("🔄 MCP 서버 및 AI 모델 초기화 중..."):
        # 먼저 기존 클라이언트를 안전하게 정리
        await cleanup_mcp_client()

        if mcp_config is None:
            # config.json 파일에서 설정 로드
            mcp_config = load_config_from_json()

        try:
            # 1. 선택된 모델 검증
            selected_model_key = st.session_state.selected_model

            if ":" not in selected_model_key:
                st.error("❌ 잘못된 모델 형식입니다. 제공자를 선택해주세요.")
                return False

            provider_name = selected_model_key.split(":")[0]

            # 제공자가 등록되어 있는지 확인
            if not st.session_state.model_manager.is_provider_registered(provider_name):
                provider_display = (
                    "OpenAI" if provider_name == "openai" else "AWS Bedrock"
                )
                st.error(
                    f"❌ {provider_display} 제공자가 등록되지 않았습니다. 모델 설정 탭에서 API 키를 설정해주세요."
                )
                return False

            # 2. MCP 클라이언트 초기화
            st.info("🔗 MCP 서버에 연결 중...")
            client = MultiServerMCPClient(mcp_config)
            tools = await client.get_tools()
            st.session_state.tool_count = len(tools)
            st.session_state.mcp_client = client

            # 3. 모델 인스턴스 생성
            st.info(f"🤖 {selected_model_key} 모델 초기화 중...")
            try:
                model = st.session_state.model_manager.create_model(
                    model_key=selected_model_key, temperature=0.1
                )
            except ModelProviderError as e:
                st.error(str(e))
                return False
            except Exception as e:
                # 제공자별 구체적인 에러 메시지
                if provider_name == "bedrock":
                    if "credentials" in str(e).lower():
                        st.error(
                            "❌ AWS Bedrock 인증에 실패했습니다. API 키를 확인하고 다시 시도해주세요."
                        )
                    elif "region" in str(e).lower():
                        st.error(
                            "❌ AWS 리전 설정에 문제가 있습니다. us-east-1 리전을 사용하는지 확인해주세요."
                        )
                    else:
                        st.error(f"❌ AWS Bedrock 모델 생성 중 오류: {str(e)}")
                elif provider_name == "openai":
                    if "api_key" in str(e).lower() or "unauthorized" in str(e).lower():
                        st.error(
                            "❌ OpenAI API 키가 유효하지 않습니다. 키를 확인하고 다시 시도해주세요."
                        )
                    else:
                        st.error(f"❌ OpenAI 모델 생성 중 오류: {str(e)}")
                else:
                    st.error(f"❌ 모델 생성 중 오류 발생: {str(e)}")
                return False

            # 4. LangGraph 에이전트 생성
            st.info("🔧 AI 에이전트 구성 중...")
            try:
                agent = create_react_agent(
                    model,
                    tools,
                    checkpointer=MemorySaver(),
                    prompt=SYSTEM_PROMPT,
                )
                st.session_state.agent = agent
                st.session_state.session_initialized = True

                # 성공 메시지
                model_info = st.session_state.model_manager.get_model_info(
                    selected_model_key
                )
                if model_info:
                    st.success(
                        f"✅ {model_info.display_name} 모델이 성공적으로 초기화되었습니다!"
                    )
                else:
                    st.success("✅ AI 모델이 성공적으로 초기화되었습니다!")

                return True

            except Exception as e:
                st.error(f"❌ AI 에이전트 생성 중 오류 발생: {str(e)}")
                return False

        except Exception as e:
            st.error(f"❌ 초기화 중 예상치 못한 오류 발생: {str(e)}")
            return False


# --- 모델 설정 탭 ---
with model_container:
    st.subheader("🤖 AI 모델 설정")

    # 안내문 추가
    st.info(
        "💡 **안내:** 아래 두 제공자 중 하나만 설정해도 사용할 수 있습니다. 둘 다 설정하면 모델을 자유롭게 전환할 수 있습니다."
    )

    # 세션 상태 초기화
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""
    if "bedrock_api_key" not in st.session_state:
        st.session_state.bedrock_api_key = ""

    # AWS Bedrock API 키 설정 섹션 (위로 이동)
    st.markdown("### ☁️ AWS Bedrock API 키 설정")

    bedrock_api_key_input = st.text_input(
        "AWS Bedrock API 키",
        value="",
        type="password",
        help="AWS Bedrock API 키를 입력하세요. Cross Region Inference를 위해 us-east-1 리전을 사용합니다.",
        placeholder="bedrock-api-key-...",
        key="bedrock_api_key_input",
    )

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button(
            "☁️ Bedrock 키 적용", key="apply_bedrock_key", use_container_width=True
        ):
            if bedrock_api_key_input.strip():
                if st.session_state.model_manager.register_provider(
                    "bedrock", bedrock_api_key_input.strip()
                ):
                    st.session_state.bedrock_api_key = bedrock_api_key_input.strip()
                    st.success("✅ AWS Bedrock API 키가 적용되었습니다.")
                    st.rerun()
                else:
                    st.error("❌ 유효하지 않은 AWS Bedrock API 키입니다.")
            else:
                st.error("❌ API 키를 입력해주세요.")

    # Bedrock 상태 표시
    if st.session_state.model_manager.is_provider_registered("bedrock"):
        masked_key = (
            st.session_state.bedrock_api_key[:7]
            + "..."
            + st.session_state.bedrock_api_key[-4:]
            if len(st.session_state.bedrock_api_key) > 11
            else "설정됨"
        )
        st.success(f"✅ AWS Bedrock API 키가 설정되어 있습니다. ({masked_key})")
        st.info("🌍 Cross Region Inference 활성화 (us-east-1 리전)")
    else:
        st.warning("⚠️ AWS Bedrock API 키를 입력해주세요.")

    st.divider()

    # OpenAI API 키 설정 섹션 (아래로 이동)
    st.markdown("### 🤖 OpenAI API 키 설정")

    openai_api_key_input = st.text_input(
        "OpenAI API 키",
        value="",
        type="password",
        help="OpenAI API 키를 입력하세요. sk-로 시작하는 키입니다.",
        placeholder="sk-proj-...",
        key="openai_api_key_input",
    )

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button(
            "🤖 OpenAI 키 적용", key="apply_openai_key", use_container_width=True
        ):
            if openai_api_key_input.strip():
                if st.session_state.model_manager.register_provider(
                    "openai", openai_api_key_input.strip()
                ):
                    st.session_state.openai_api_key = openai_api_key_input.strip()
                    st.success("✅ OpenAI API 키가 적용되었습니다.")
                    st.rerun()
                else:
                    st.error("❌ 유효하지 않은 OpenAI API 키입니다.")
            else:
                st.error("❌ API 키를 입력해주세요.")

    # OpenAI 상태 표시
    if st.session_state.model_manager.is_provider_registered("openai"):
        masked_key = (
            st.session_state.openai_api_key[:7]
            + "..."
            + st.session_state.openai_api_key[-4:]
            if len(st.session_state.openai_api_key) > 11
            else "설정됨"
        )
        st.success(f"✅ OpenAI API 키가 설정되어 있습니다. ({masked_key})")
    else:
        st.warning("⚠️ OpenAI API 키를 입력해주세요.")

    st.divider()

    # 통합 모델 선택 섹션
    st.markdown("### 🧠 모델 선택")

    available_models = st.session_state.model_manager.get_available_models()

    if available_models:
        # 모델 선택 드롭다운
        model_options = [model["key"] for model in available_models]

        # 현재 선택된 모델이 사용 가능한지 확인
        current_selection = st.session_state.selected_model
        if current_selection not in model_options and model_options:
            current_selection = model_options[0]
            st.session_state.selected_model = current_selection

        def format_model_display(model_key):
            model_info = next(
                (m for m in available_models if m["key"] == model_key), None
            )
            if model_info:
                provider_badge = "🤖" if model_info["provider"] == "openai" else "☁️"
                return f"{provider_badge} {model_info['display']}"
            return model_key

        previous_model = st.session_state.selected_model
        selected_model = st.selectbox(
            "사용할 모델 선택",
            options=model_options,
            index=(
                model_options.index(current_selection)
                if current_selection in model_options
                else 0
            ),
            format_func=format_model_display,
            help="등록된 제공자의 모델을 선택하세요.",
            key="model_selector",
        )

        st.session_state.selected_model = selected_model

        # 모델이 변경되었을 때 세션 초기화 필요 알림
        if previous_model != selected_model and st.session_state.session_initialized:
            st.warning(
                "⚠️ 모델이 변경되었습니다. MCP 도구 탭에서 '설정 적용하기' 버튼을 눌러 변경사항을 적용하세요."
            )

        st.divider()

        # 선택된 모델 정보 표시
        st.subheader("📊 현재 모델 정보")
        model_config = st.session_state.model_manager.get_model_info(selected_model)

        if model_config:
            provider_name = selected_model.split(":")[0]
            provider_info = st.session_state.model_manager.get_provider_info(
                provider_name
            )

            st.write(f"🧠 **선택된 모델:** {model_config.display_name}")
            st.write(f"🏢 **제공자:** {provider_info['display_name']}")
            if model_config.description:
                st.info(f"📝 {model_config.description}")
    else:
        st.warning("⚠️ 사용 가능한 모델이 없습니다. 위에서 API 키를 설정해주세요.")

        # 제공자 상태 요약 표시
        st.markdown("### 📋 제공자 상태")
        providers_info = st.session_state.model_manager.get_all_providers_info()

        for provider_name, info in providers_info.items():
            status_icon = "✅" if info["is_registered"] else "❌"
            st.write(
                f"{status_icon} **{info['display_name']}**: {'등록됨' if info['is_registered'] else '미등록'}"
            )
            if info["description"]:
                st.caption(f"   {info['description']}")

# --- MCP 도구 설정 탭 ---
with mcp_container:
    # 설정 적용하기 버튼을 최상단으로 이동
    if st.button(
        "⚙️ 설정 적용하기",
        key="apply_button",
        type="primary",
        use_container_width=True,
    ):
        # 적용 중 메시지 표시
        apply_status = st.empty()
        with apply_status.container():
            st.warning("🔄 변경사항을 적용하고 있습니다. 잠시만 기다려주세요...")
            progress_bar = st.progress(0)

            # 설정 저장
            st.session_state.mcp_config_text = json.dumps(
                st.session_state.pending_mcp_config, indent=2, ensure_ascii=False
            )

            # config.json 파일에 설정 저장
            save_result = save_config_to_json(st.session_state.pending_mcp_config)
            if not save_result:
                st.error("❌ 설정 파일 저장에 실패했습니다.")

            progress_bar.progress(15)

            # 세션 초기화 준비
            st.session_state.session_initialized = False
            st.session_state.agent = None

            # 진행 상태 업데이트
            progress_bar.progress(30)

            # 초기화 실행
            success = st.session_state.event_loop.run_until_complete(
                initialize_session(st.session_state.pending_mcp_config)
            )

            # 진행 상태 업데이트
            progress_bar.progress(100)

            if success:
                st.success("✅ 새로운 설정이 적용되었습니다.")
                # 도구 추가 expander 접기
                if "mcp_tools_expander" in st.session_state:
                    st.session_state.mcp_tools_expander = False
            else:
                st.error("❌ 설정 적용에 실패하였습니다.")

        # 페이지 새로고침
        st.rerun()

    st.divider()

    # MCP 도구 수 정보 표시
    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric("🛠️ 등록된 MCP 도구", st.session_state.get("tool_count", 0))
    with col2:
        st.metric(
            "✅ 초기화 상태",
            "완료" if st.session_state.get("session_initialized", False) else "미완료",
        )

    st.divider()

    # 현재 적용된 MCP 서버 리스트
    st.markdown("### 📋 현재 적용된 MCP 서버")

    # pending config가 없으면 기존 mcp_config_text 기반으로 생성
    if "pending_mcp_config" not in st.session_state:
        try:
            loaded_config = load_config_from_json()
            st.session_state.pending_mcp_config = loaded_config
        except Exception as e:
            st.error(f"초기 pending config 설정 실패: {e}")
            st.session_state.pending_mcp_config = {}

    try:
        pending_config = st.session_state.pending_mcp_config
        if pending_config:
            for i, (tool_name, tool_config) in enumerate(pending_config.items()):
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**{tool_name}**")
                        if "command" in tool_config:
                            st.caption(
                                f"Command: {tool_config['command']} {' '.join(tool_config.get('args', [])[:2])}..."
                            )
                        elif "url" in tool_config:
                            st.caption(f"URL: {tool_config['url']}")
                    with col2:
                        # 고유한 키 생성: 도구 이름과 인덱스 조합
                        if st.button(
                            "🗑️", key=f"delete_server_{tool_name}_{i}", help="삭제"
                        ):
                            del st.session_state.pending_mcp_config[tool_name]
                            st.success(f"{tool_name} 서버가 삭제되었습니다.")
                            st.rerun()

                    if i < len(pending_config) - 1:
                        st.divider()
        else:
            st.info("등록된 MCP 서버가 없습니다.")
    except Exception as e:
        st.error("MCP 서버 목록을 불러오는 중 오류가 발생했습니다.")

    st.divider()

    # MCP 서버 추가 섹션
    st.markdown("### ➕ 새 MCP 서버 추가")
    st.markdown("💡 중괄호 숫자를 잘 확인하고 JSON 형식을 체크해주세요")

    # 기본 예시 JSON
    default_json = {
        "desktop-commander": {
            "command": "npx",
            "args": [
                "-y",
                "@smithery/cli@latest",
                "run",
                "@wonderwhy-er/desktop-commander",
                "--key",
                "8f1bc671-fe10-43cd-8da1-b76a057f3c0a",
            ],
            "transport": "stdio",
        }
    }

    default_text = json.dumps(default_json, indent=2, ensure_ascii=False)

    new_tool_json = st.text_area(
        "MCP 서버 설정 (JSON)",
        default_text,
        height=300,
        help="JSON 형식으로 MCP 서버 설정을 입력하세요",
    )

    # 추가하기 버튼
    if st.button(
        "➕ MCP 서버 추가",
        type="primary",
        key="add_mcp_server_button",
        use_container_width=True,
    ):
        try:
            # 입력값 검증
            if not new_tool_json.strip().startswith(
                "{"
            ) or not new_tool_json.strip().endswith("}"):
                st.error("JSON은 중괄호({})로 시작하고 끝나야 합니다.")
                st.markdown('올바른 형식: `{ "도구이름": { ... } }`')
            else:
                # JSON 파싱
                parsed_tool = json.loads(new_tool_json)

                # mcpServers 형식인지 확인하고 처리
                if "mcpServers" in parsed_tool:
                    # mcpServers 안의 내용을 최상위로 이동
                    parsed_tool = parsed_tool["mcpServers"]
                    st.info("'mcpServers' 형식이 감지되었습니다. 자동으로 변환합니다.")

                # 입력된 도구 수 확인
                if len(parsed_tool) == 0:
                    st.error("최소 하나 이상의 도구를 입력해주세요.")
                else:
                    # 모든 도구에 대해 처리
                    success_tools = []
                    for tool_name, tool_config in parsed_tool.items():
                        # URL 필드 확인 및 transport 설정
                        if "url" in tool_config:
                            # URL이 있는 경우 transport를 "sse"로 설정
                            tool_config["transport"] = "sse"
                            st.info(
                                f"'{tool_name}' 도구에 URL이 감지되어 transport를 'sse'로 설정했습니다."
                            )
                        elif "transport" not in tool_config:
                            # URL이 없고 transport도 없는 경우 기본값 "stdio" 설정
                            tool_config["transport"] = "stdio"

                        # 필수 필드 확인
                        if "command" not in tool_config and "url" not in tool_config:
                            st.error(
                                f"'{tool_name}' 도구 설정에는 'command' 또는 'url' 필드가 필요합니다."
                            )
                        elif "command" in tool_config and "args" not in tool_config:
                            st.error(
                                f"'{tool_name}' 도구 설정에는 'args' 필드가 필요합니다."
                            )
                        elif "command" in tool_config and not isinstance(
                            tool_config["args"], list
                        ):
                            st.error(
                                f"'{tool_name}' 도구의 'args' 필드는 반드시 배열([]) 형식이어야 합니다."
                            )
                        else:
                            # pending_mcp_config에 도구 추가
                            st.session_state.pending_mcp_config[tool_name] = tool_config
                            success_tools.append(tool_name)

                    # 성공 메시지
                    if success_tools:
                        if len(success_tools) == 1:
                            st.success(
                                f"{success_tools[0]} 도구가 추가되었습니다. 적용하려면 '설정 적용하기' 버튼을 눌러주세요."
                            )
                        else:
                            tool_names = ", ".join(success_tools)
                            st.success(
                                f"총 {len(success_tools)}개 도구({tool_names})가 추가되었습니다. 적용하려면 '설정 적용하기' 버튼을 눌러주세요."
                            )
                        # 추가되면 expander를 접어줌
                        st.session_state.mcp_tools_expander = False
                        st.rerun()
        except json.JSONDecodeError as e:
            st.error(f"JSON 파싱 에러: {e}")
            st.markdown(
                f"""
                **수정 방법**:
                1. JSON 형식이 올바른지 확인하세요.
                2. 모든 키는 큰따옴표(")로 감싸야 합니다.
                3. 문자열 값도 큰따옴표(")로 감싸야 합니다.
                4. 문자열 내에서 큰따옴표를 사용할 경우 이스케이프(\\")해야 합니다.
                """
            )
        except Exception as e:
            st.error(f"오류 발생: {e}")

    st.divider()

    # 기본 서버 복원 버튼
    if st.button(
        "🔄 기본 서버 복원 (시간, 헬스계산기)",
        key="restore_default_mcp_tools",
        use_container_width=True,
    ):
        # 기본 설정 정의
        default_tools = {
            "get_current_time": {
                "command": "python",
                "args": ["./mcp_servers/time.py"],
                "transport": "stdio",
            },
            "fitness_calculator": {
                "command": "python",
                "args": ["./mcp_servers/fitness.py"],
                "transport": "stdio",
            },
        }

        # 기존에 없는 기본 도구만 추가
        added_tools = []
        for tool_name, tool_config in default_tools.items():
            if tool_name not in st.session_state.pending_mcp_config:
                st.session_state.pending_mcp_config[tool_name] = tool_config
                added_tools.append(tool_name)

        if added_tools:
            tool_names = ", ".join(added_tools)
            st.success(f"기본 서버 {tool_names}가 복원되었습니다.")
            st.rerun()
        else:
            st.info("모든 기본 서버가 이미 등록되어 있습니다.")

    st.divider()  # 구분선 추가


# --- 챗봇 탭 ---
with chat_container:
    # 상단 버튼 영역
    col1, col2 = st.columns([3, 1])

    with col1:
        # --- 제공자 및 세션 상태 확인 ---
        available_models = st.session_state.model_manager.get_available_models()

        if not available_models:
            st.warning(
                "⚠️ 사용 가능한 모델이 없습니다. '모델 설정' 탭에서 API 키를 설정해주세요."
            )
        elif not st.session_state.session_initialized:
            st.info(
                "MCP 서버와 에이전트가 초기화되지 않았습니다. 'MCP 도구' 탭에서 '설정 적용하기' 버튼을 클릭하여 초기화해주세요."
            )

    with col2:
        # 대화 초기화 버튼
        if st.button(
            "🔄 대화 초기화", key="reset_chat_history", use_container_width=True
        ):
            # thread_id 초기화
            st.session_state.thread_id = random_uuid()
            # 대화 히스토리 초기화
            st.session_state.history = []
            # 알림 메시지
            st.success("✅ 대화가 초기화되었습니다.")
            # 페이지 새로고침
            st.rerun()

    st.divider()

    # --- 대화 기록 출력 ---
    print_message()

# --- 화면 하단 고정: 사용자 입력 및 처리 ---
user_query = st.chat_input("💬 질문을 입력하세요")
if user_query:
    # 사용 가능한 모델 확인
    available_models = st.session_state.model_manager.get_available_models()
    if not available_models:
        st.warning(
            "⚠️ 사용 가능한 모델이 없습니다. '모델 설정' 탭에서 API 키를 설정해주세요."
        )
    elif st.session_state.session_initialized:
        # 챗봇 탭이 활성화되어 있을 때만 채팅 메시지 표시
        with chat_container:
            st.chat_message("user", avatar="🧑‍💻").markdown(user_query)
            with st.chat_message("assistant", avatar="🤖"):
                tool_placeholder = st.empty()
                text_placeholder = st.empty()
                resp, final_text, final_tool = (
                    st.session_state.event_loop.run_until_complete(
                        process_query(
                            user_query,
                            text_placeholder,
                            tool_placeholder,
                            TIMEOUT_SECONDS,
                        )
                    )
                )
            if "error" in resp:
                st.error(resp["error"])
            else:
                st.session_state.history.append({"role": "user", "content": user_query})
                st.session_state.history.append(
                    {"role": "assistant", "content": final_text}
                )
                if final_tool.strip():
                    st.session_state.history.append(
                        {"role": "assistant_tool", "content": final_tool}
                    )
                st.rerun()
    else:
        st.warning(
            "⚠️ MCP 서버와 에이전트가 초기화되지 않았습니다. 'MCP 도구' 탭에서 '설정 적용하기' 버튼을 클릭하여 초기화해주세요."
        )
