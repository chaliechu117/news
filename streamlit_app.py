# app.py
import streamlit as st
import requests, re, json, math
from bs4 import BeautifulSoup
from huggingface_hub import InferenceClient

st.set_page_config(page_title="뉴스 호재/악재 분류기 • Llama-2-7b-chat", page_icon="🦙")

# -----------------------------
# UI: 사이드바 설정
# -----------------------------
st.sidebar.header("모델 설정")
hf_token = st.sidebar.text_input("Hugging Face Access Token", type="password", placeholder="hf_xxx")
model = st.sidebar.text_input("Model Repo", value="meta-llama/Llama-2-7b-chat-hf")
temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.2, 0.05)
max_new_tokens = st.sidebar.slider("max_new_tokens", 64, 1024, 256, 32)
top_p = st.sidebar.slider("top_p", 0.1, 1.0, 0.9, 0.05)

st.title("🦙 Llama-2-7b-chat 뉴스 호재/악재 분류기")

# -----------------------------
# 입력 방식 선택: URL 스크랩 vs 직접 입력
# -----------------------------
mode = st.radio("입력 방식을 선택하세요", ["뉴스 URL로 스크랩", "텍스트 직접 붙여넣기"], horizontal=True)

def fetch_article(url: str) -> str:
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        html = resp.text
        soup = BeautifulSoup(html, "html.parser")

        # meta/og/ld+json에서 제목·설명 후보
        og_desc = soup.find("meta", property="og:description")
        og_title = soup.find("meta", property="og:title")

        # 본문 후보: article 태그, id/class에 article|content 포함, p 태그 다모으기
        candidates = []
        for sel in ["article", "[id*='article']", "[class*='article']", "[id*='content']", "[class*='content']"]:
            for node in soup.select(sel):
                text = " ".join(p.get_text(" ", strip=True) for p in node.find_all(["p","h2","li"]))
                if len(text) > 400:
                    candidates.append(text)

        # 일반 p 태그 백업 플랜
        if not candidates:
            pts = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
            if len(pts) > 400:
                candidates.append(pts)

        bodies = []
        if og_title and og_title.get("content"):
            bodies.append(og_title["content"])
        if og_desc and og_desc.get("content"):
            bodies.append(og_desc["content"])
        if candidates:
            bodies.append(max(candidates, key=len))

        text = "\n\n".join(bodies).strip()
        # 광고/잡음 간단 제거
        text = re.sub(r"\s+", " ", text)
        return text
    except Exception as e:
        st.error(f"스크랩 오류: {e}")
        return ""

news_text = ""
if mode == "뉴스 URL로 스크랩":
    url = st.text_input("뉴스 URL", placeholder="https://...")
    if st.button("스크랩 실행", type="primary") and url:
        news_text = fetch_article(url)
        if not news_text:
            st.warning("본문을 추출하지 못했습니다. 텍스트를 직접 붙여넣어 주세요.")
else:
    news_text = st.text_area("뉴스 원문을 붙여넣으세요", height=240)

if news_text:
    with st.expander("입력 텍스트 확인/편집"):
        news_text = st.text_area("텍스트", value=news_text, height=240)

# -----------------------------
# 프롬프트 템플릿 (Llama-2 Chat 스타일)
# 출력은 JSON만 허용
# -----------------------------
def build_prompt(article: str) -> str:
    system = (
        "You are a precise financial news classifier. "
        "Classify whether the news is bullish (호재), bearish (악재), or neutral for the main involved asset(s). "
        "Return only valid JSON in UTF-8 with keys: "
        "label (one of: bullish, bearish, neutral), "
        "confidence (0~1 float), "
        "summary_ko (<=80자 한국어 요약), "
        "reasons_ko (한국어 근거 2~4개 리스트), "
        "tickers (관련 종목/티커 리스트, 모르겠으면 빈 리스트)."
    )
    user = (
        "아래 뉴스의 투자 관점 임팩트를 분류해줘. 모호하면 neutral로.\n\n"
        f"뉴스 원문:\n{article[:8000]}\n\n"
        "출력은 오직 JSON만:\n"
        "{"
        "\"label\":\"bullish|bearish|neutral\","
        "\"confidence\":0.0,"
        "\"summary_ko\":\"...\","
        "\"reasons_ko\":[\"...\"],"
        "\"tickers\":[\"...\"]"
        "}"
    )
    # Llama-2 chat 형식
    return f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n{user} [/INST]"

def extract_json(text: str):
    try:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return None
        obj = json.loads(m.group(0))
        return obj
    except Exception:
        return None

# -----------------------------
# 분류 실행
# -----------------------------
def classify(article: str):
    if not hf_token:
        st.warning("먼저 Hugging Face 토큰을 입력하세요.")
        return None, None

    client = InferenceClient(model=model, token=hf_token)
    prompt = build_prompt(article)

    # Inference API: text_generation (Llama-2-7b-chat-hf)
    # 참고: 모델/엔드포인트에 따라 'stop_sequences'가 필요할 수 있음
    try:
        output = client.text_generation(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=1.05,
            return_full_text=False,
        )
        parsed = extract_json(output)
        return output, parsed
    except Exception as e:
        st.error(f"Inference 오류: {e}")
        return None, None

run = st.button("분류 실행", type="primary", disabled=(not bool(news_text)))
if run and news_text:
    with st.spinner("분류 중..."):
        raw, parsed = classify(news_text)

    if parsed is None:
        st.error("JSON 파싱에 실패했습니다. raw 출력 확인 후 프롬프트를 조정하세요.")
        if raw:
            st.code(raw)
    else:
        label = str(parsed.get("label", "neutral")).lower()
        confidence = float(parsed.get("confidence", 0.0))
        summary_ko = parsed.get("summary_ko", "")
        reasons_ko = parsed.get("reasons_ko", [])
        tickers = parsed.get("tickers", [])

        # 라벨 색상
        color = {"bullish": "#0ea5e9", "bearish": "#ef4444", "neutral": "#9ca3af"}.get(label, "#9ca3af")

        st.markdown(
            f"""
            <div style="display:flex;gap:10px;align-items:center;">
              <div style="padding:4px 10px;border-radius:999px;background:{color};color:white;font-weight:600;">
                {label.upper()}
              </div>
              <div style="opacity:0.8;">confidence: {confidence:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if summary_ko:
            st.markdown(f"요약: {summary_ko}")
        if reasons_ko:
            st.markdown("근거")
            for i, r in enumerate(reasons_ko, 1):
                st.markdown(f"- {i}. {r}")
        if tickers:
            st.markdown(f"관련 종목/티커: {', '.join(tickers)}")

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "JSON 결과 다운로드",
                data=json.dumps(parsed, ensure_ascii=False, indent=2),
                file_name="classification.json",
                mime="application/json",
            )
        with col2:
            st.download_button(
                "Raw 출력 다운로드",
                data=raw if raw else "",
                file_name="raw_output.txt",
                mime="text/plain",
                disabled=(raw is None)
            )

        with st.expander("원문 프롬프트 보기"):
            st.code(build_prompt(news_text))

# -----------------------------
# 사용 팁
# -----------------------------
st.caption(
    "팁: Llama-2-7b-chat은 라이선스 수락이 필요합니다. Hugging Face 계정에서 해당 모델 접근 권한을 받아야 Inference API 호출이 성공합니다. "
    "또한 뉴스 출처/포맷에 따라 스크랩 성능이 달라질 수 있어요."
)
