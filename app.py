# app.py
# -*- coding: utf-8 -*-
import json
import time
from collections import defaultdict

import streamlit as st
import google.generativeai as genai

# --------------------------
# 0. Gemini 설정 (키는 secrets에서만 읽기)
# --------------------------
# Streamlit Cloud / 로컬 .streamlit/secrets.toml 에서
# GEMINI_API_KEY 를 넣어둘 거야.
API_KEY = st.secrets.get("GEMINI_API_KEY")
if not API_KEY:
    st.stop()  # 키 없으면 바로 중단
genai.configure(api_key=API_KEY)

MODEL_ID = "gemini-2.0-flash-001"
model = genai.GenerativeModel(MODEL_ID)

# --------------------------
# 1. 검수 프롬프트 (네 로직 기반, 단일 텍스트 버전)
# --------------------------

def create_review_prompt_for_text(korean_text: str) -> str:
    """
    네가 원래 쓰던 create_review_prompt를
    '번역문 한글 하나만 검수'하는 버전으로 단순화.
    plain_korean / markdown_korean 둘 다 같은 텍스트로 사용.
    """
    translation_text = korean_text
    translation_md = korean_text

    prompt = f"""
    You are a machine-like **Data Verifier**. Your ONLY job is to find **objective, factual errors**. You are strictly forbidden from judging style, meaning, or making subjective suggestions. Your output MUST BE a single, valid JSON object.

    **Definition of "Objective Error":**
    You must only report the following:
    1.  **Typos:** Clearly misspelled words (e.g., "recieve" -> "receive", "이점들을를" -> "이점들을").
    2.  **Grammatical Errors:** Incorrect particles, endings, or spacing (e.g., "사과을" -> "사과를").
    3.  **Content Mismatch:** Verifiable differences between plain text and markdown versions (e.g., a word is missing in the markdown).

    **CRITICAL RULES OF ENGAGEMENT:**
    1.  **ABSOLUTELY NO STYLISTIC FEEDBACK:** Do not suggest alternative wording (e.g., "근사한" vs "멋있는"). Do not comment on what sounds "more natural" or "more appropriate". This is a critical failure.
    2.  **SILENCE ON PERFECTION:** If no objective errors are found, the report field MUST be an empty string (`""`). Do not write "오류 없음".
    3.  **RESPECT STYLING CONVENTIONS:** It is CORRECT for *italicized English* to be represented with `'single quotes'` or `《double angle brackets》` in Korean. This is NOT an error.
    4.  **GROUND YOUR FINDINGS:** When reporting an error, you MUST quote the problematic text.
    5.  **NO IDENTICAL CORRECTIONS:** A suggested correction must be different from the original text.

    ---
    **EXAMPLES OF CORRECT AND INCORRECT EXECUTION:**

    **Example 1: Correct - Typo in Korean text**
    - `plain_korean`: "이점들을를 확인할 수 있습니다."
    - **Your Correct JSON Output:**
    ```json
    {{
        "suspicion_score": 3,
        "content_typo_report": "",
        "translated_typo_report": "- '이점들을를'에서 오타 발견. '이점들을'로 수정해야 함.",
        "markdown_report": ""
    }}
    ```

    **Example 2: Correct - No errors found**
    - `plain_korean`: "아울러, 이것은 테스트입니다."
    - **Your Correct JSON Output:**
    ```json
    {{
        "suspicion_score": 1,
        "content_typo_report": "",
        "translated_typo_report": "",
        "markdown_report": ""
    }}
    ```

    **Example 3: INCORRECT - Making a stylistic suggestion (DO NOT DO THIS)**
    - `plain_korean`: "아울러, 이것은 테스트입니다."
    - **Your INCORRECT (Forbidden) Output:**
      `"translated_typo_report": "- '아울러'는 문맥상 부적절합니다. '오히려'로 수정하는 것이 좋습니다."`
    - **This is a violation of Rule #1. The word '아울러' is not a typo or a grammatical error.**

    ---

    **ANALYSIS WORKFLOW:**
    Now, apply these strict rules and examples to the following data.

    **Data to Review:**
    - `plain_english`: ""
    - `markdown_english`: ""
    - `plain_korean`: "{translation_text}"
    - `markdown_korean`: "{translation_md}"
    """
    return prompt


def analyze_text_with_gemini(prompt: str, max_retries: int = 3) -> dict:
    """Gemini를 JSON 모드로 호출"""
    for attempt in range(max_retries):
        try:
            generation_config = {
                "response_mime_type": "application/json",
                "temperature": 0.0,
            }
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            return json.loads(response.text)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                return {
                    "suspicion_score": 5,
                    "content_typo_report": f"API 호출 실패: {str(e)}",
                    "translated_typo_report": "",
                    "markdown_report": ""
                }


def validate_and_clean_analysis(result: dict) -> dict:
    """네가 짠 필터링 로직을 단일 텍스트 버전으로 정리"""
    if not isinstance(result, dict):
        return {
            "suspicion_score": 5,
            "content_typo_report": "AI 응답이 유효한 JSON 형식이 아님",
            "translated_typo_report": "",
            "markdown_report": ""
        }

    score = result.get('suspicion_score')
    reports = {
        "content_typo_report": result.get('content_typo_report', ''),
        "translated_typo_report": result.get('translated_typo_report', ''),
        "markdown_report": result.get('markdown_report', '')
    }

    # 스타일/문체 제안 금지 키워드
    forbidden_keywords = [
        "문맥상", "부적절", "어색", "더 자연스럽", "더 적절", "수정하는 것이 좋",
        "제안", "바꾸는 것", "의미를 명확히"
    ]
    for key, text in reports.items():
        if any(kw in text for kw in forbidden_keywords):
            reports[key] = ""

    # "오류 없음" 같은 멘트 제거
    forbidden_phrases = ["오류 없음", "정상", "문제 없음", "수정할 필요 없음"]
    for key, text in reports.items():
        if any(ph in text for ph in forbidden_phrases):
            reports[key] = ""

    final_content = reports["content_typo_report"]
    final_translated = reports["translated_typo_report"]
    final_markdown = reports["markdown_report"]

    if not final_content and not final_translated and not final_markdown:
        score = 1
    elif (final_content or final_translated or final_markdown) and score == 1:
        score = 3

    return {
        "suspicion_score": score,
        "content_typo_report": final_content,
        "translated_typo_report": final_translated,
        "markdown_report": final_markdown,
    }


def review_text(korean_text: str) -> dict:
    """Streamlit에서 호출할 최종 함수"""
    prompt = create_review_prompt_for_text(korean_text)
    raw = analyze_text_with_gemini(prompt)
    cleaned = validate_and_clean_analysis(raw)
    return {
        "score": cleaned.get("suspicion_score"),
        "content_typo_report": cleaned.get("content_typo_report", ""),
        "translated_typo_report": cleaned.get("translated_typo_report", ""),
        "markdown_report": cleaned.get("markdown_report", ""),
        "raw": raw,  # 디버깅용
    }

# --------------------------
# 2. Streamlit UI
# --------------------------

st.set_page_config(
    page_title="AI 검수기 (Gemini)",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI 한국어 텍스트 검수기 (Gemini 기반)")
st.caption("오탈자 / 조사 / 띄어쓰기 / 형식 오류에만 집중하는 검수기 (스타일 제안 금지).")

tab_main, tab_about, tab_debug = st.tabs(["✏️ 텍스트 검수", "ℹ️ 설명", "🐞 디버그(개발자용)"])

with tab_main:
    st.subheader("검수할 텍스트를 입력하세요")
    default_text = "이것은 테스트 문장 입니다. 그는는 학교에 갔다."
    text = st.text_area(
        "입력 텍스트",
        value=default_text,
        height=220,
        help="번역문/교과서/모의고사 지문 등 검수하고 싶은 한국어 텍스트를 넣어주세요.",
    )

    if st.button("검수 실행", type="primary"):
        if not text.strip():
            st.warning("먼저 텍스트를 입력해주세요.")
        else:
            with st.spinner("AI가 검수 중입니다..."):
                result = review_text(text)

            score = result.get("score")
            content_report = result.get("content_typo_report") or ""
            translated_report = result.get("translated_typo_report") or ""
            markdown_report = result.get("markdown_report") or ""

            st.success("검수가 완료되었습니다!")

            if score is not None:
                st.metric("의심 점수 (1~5)", f"{score:.2f}")

            st.markdown("### 🔍 리포트")

            with st.expander("🇰🇷 번역문/한글 텍스트 리포트 (translated_typo_report)", expanded=True):
                if translated_report.strip():
                    st.markdown(translated_report)
                else:
                    st.info("보고할 오류가 없습니다.")

            with st.expander("📄 CONTENT 원문 리포트 (content_typo_report)"):
                if content_report.strip():
                    st.markdown(content_report)
                else:
                    st.info("보고할 오류가 없습니다.")

            with st.expander("📝 마크다운 변환 리포트 (markdown_report)"):
                if markdown_report.strip():
                    st.markdown(markdown_report)
                else:
                    st.info("보고할 오류가 없습니다.")

with tab_about:
    st.markdown("""
### 이 앱은?

- 네가 만든 **검수 규칙(프롬프트 + 필터링)**을
- **Gemini API + Streamlit**으로 감싼 텍스트 검수기입니다.
- 현재 버전은 “단일 텍스트”만 검수합니다.  
  (나중에 시트명/탭명 입력해서 돌리는 배치 버전도 여기서 이어서 만들 수 있어요)

### 동작

1. 사용자가 한국어 텍스트를 입력
2. `review_text()`가 Gemini를 JSON 모드로 호출
3. 결과에서 스타일/문체 제안은 모두 필터링
4. 의심 점수 + 리포트 3종을 화면에 표시
""")

with tab_debug:
    st.markdown("여기는 나중에 raw JSON을 보는 용도로 쓸 수 있습니다. (현재는 입력 후 콘솔 등으로 확인)")
