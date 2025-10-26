import streamlit as st

st.set_page_config(page_title="News Impact Rater", page_icon="📈")

st.title("News Impact Rater")
st.caption("LLaMA 또는 EXAONE 중 하나를 선택하여 사용할 수 있습니다.")

# -----------------------------
# 1. 모델 선택 섹션
# -----------------------------
st.subheader("모델 선택")
col1, col2 = st.columns(2)

use_llama = col1.checkbox("🦙 LLaMA", value=False)
use_exaone = col2.checkbox("🤖 EXAONE", value=False)

# 체크박스 중복 방지
if use_llama and use_exaone:
    st.warning("하나만 선택하세요. (LLaMA 또는 EXAONE 중 하나)")
    st.stop()

# -----------------------------
# 2. 모델 기본 설정
# -----------------------------
if use_llama:
    default_model = "meta-llama/Llama-3.1-8b-instruct"
    default_base = "http://localhost:8000/v1"
    st.success("LLaMA 모드가 선택되었습니다.")
elif use_exaone:
    default_model = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
    default_base = "https://api-inference.huggingface.co/models/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
    st.success("EXAONE 모드가 선택되었습니다.")
else:
    default_model = ""
    default_base = ""
    st.info("모델을 선택하면 기본 설정이 자동으로 채워집니다.")

# -----------------------------
# 3. 사용자 입력 (API 정보)
# -----------------------------
st.subheader("API 설정")

base_url = st.text_input("API Base URL", value=default_base, placeholder="예: https://api.openai.com/v1")
api_key = st.text_input("API Key / Token", type="password", placeholder="hf_xxx 또는 sk-xxx")
model_id = st.text_input("모델 이름", value=default_model, placeholder="예: meta-llama/Llama-3.1-8b-instruct")

# -----------------------------
# 4. 테스트 프롬프트
# -----------------------------
st.subheader("테스트 프롬프트")
prompt = st.text_area("뉴스 본문", placeholder="이곳에 뉴스 텍스트를 입력하세요.")

if st.button("테스트 실행"):
    if not (use_llama or use_exaone):
        st.error("먼저 모델을 선택하세요.")
    elif not api_key:
        st.error("API Key를 입력하세요.")
    elif not base_url or not model_id:
        st.error("API 설정을 확인하세요.")
    elif not prompt.strip():
        st.error("프롬프트를 입력하세요.")
    else:
        st.success(f"✅ 설정 완료!\n- 모델: {model_id}\n- API URL: {base_url}")
        st.info("이후 단계에서 뉴스 영향력 평가 기능과 연동됩니다.")

