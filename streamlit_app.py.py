import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="🦙 LLaMA Chat", page_icon="🦙")

# 기본 세팅
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "너는 신속하고 정확한 한국어 AI 도우미야."}]

st.title("🦙 LLaMA Chat (사용자 API 키 입력)")

# 사용자 API 키 입력창
st.subheader("API 키 입력")
st.info("LLaMA 호환 서버의 OpenAI API 키를 입력 하세요. (예: vLLM, Ollama, llama.cpp --api)")
st.session_state.api_key = st.text_input("API 키", type="password", placeholder="sk-...", key="api_key_input")

base_url = st.text_input("Base URL", value="http://localhost:8000/v1", placeholder="http://...")
model_name = st.text_input("Model 이름", value="llama-3.1-8b-instruct")

# 키가 입력되지 않은 경우 안내
if not st.session_state.api_key:
    st.warning("먼저 API 키를 입력하세요.")
    st.stop()

# OpenAI 클라이언트 생성
client = OpenAI(base_url=base_url, api_key=st.session_state.api_key)

# 대화 내역 출력
for m in st.session_state.messages[1:]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 사용자 입력
prompt = st.chat_input("메시지를 입력하세요...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        out = ""

        # 스트리밍 생성
        try:
            stream = client.chat.completions.create(
                model=model_name,
                messages=st.session_state.messages,
                stream=True,
                temperature=0.3,
                max_tokens=512,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                token = delta.content or ""
                out += token
                placeholder.markdown(out)
        except Exception as e:
            st.error(f"오류 발생: {e}")
            st.stop()

    st.session_state.messages.append({"role": "assistant", "content": out})