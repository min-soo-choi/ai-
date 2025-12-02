# app.py
# -*- coding: utf-8 -*-
import json
import time
from collections import defaultdict
import os
import streamlit as st
import google.generativeai as genai
from sheet_review import run_sheet_review


# --------------------------
# 0. Gemini 설정 (키는 secrets에서만 읽기)
# --------------------------
# Streamlit Cloud / 로컬 .streamlit/secrets.toml 에서
# GEMINI_API_KEY 를 넣어둘 거야.
API_KEY = st.secrets.get("GEMINI_API_KEY")
if not API_KEY:
    st.error("GEMINI_API_KEY가 secrets에 설정되어 있지 않습니다.")
    st.stop()

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-001")

# --------------------------
# 1. 검수 프롬프트 (네 로직 기반, 단일 텍스트 버전)
# --------------------------

def create_review_prompt_for_text(korean_text: str) -> str:
    """
    한국어 텍스트(문장/문단) 하나만 검수하는 프롬프트.
    - 의미/스타일은 건드리지 않고
    - 오탈자 / 조사·어미 / 띄어쓰기 / 기본 문장부호만 본다.
    """
    prompt = f"""
You are a machine-like **Korean text proofreader**.
Your ONLY job is to detect **objective, verifiable errors** in the following Korean text.
You are strictly forbidden from suggesting stylistic improvements, rephrasing, or commenting on "more natural" expressions.

You MUST respond with a single valid JSON object with the following keys:

- "suspicion_score": integer (1~5)
- "content_typo_report": string
- "translated_typo_report": string
- "markdown_report": string

If there is nothing to report, each report field MUST be an empty string "" (do NOT write things like "오류 없음", "문제 없음", etc.).

---

## 1. What counts as an error?

You must ONLY report these types of errors in Korean:

1. **Obvious typos (오탈자)**  
   - 잘못된 철자, 중복 글자, 명백한 입력 실수  
   - 예) "이점들을를" → "이점들을"

2. **Particles / endings (조사, 어미) errors**  
   - 주격/목적격/보격/부사격 조사 잘못 사용  
   - 동사/형용사 어미가 문법적으로 분명히 잘못된 경우  
   - 예) "사과을" → "사과를"

3. **Spacing (띄어쓰기) errors**  
   - 띄어쓰기/붙여쓰기 규범이 명백히 잘못된 경우  
   - 예) "책을읽고" → "책을 읽고"

4. **Basic punctuation (기본 문장부호) errors**  
   - 마침표/쉼표/물음표 등 필수 문장부호가 빠져
     문장이 비문이 되거나 구조가 심각하게 모호한 경우만.
   - 단순한 스타일 차이는 오류가 아니다.
   
5.  **MORPHEME SPLIT ERRORS(형태소 오류):** 무조건 오류로 판단한다. 
    예: "묻 는", "먹 는", "잡 아" → 모두 오타.

You must NOT:
- 단어 선택이 "더 자연스럽다/부자연스럽다"는 식의 의견을 말하지 마라.
- 의미를 바꾸는 재서술을 하지 마라.
- 단순 어휘 교체 제안을 하지 마라 (예: "근사한" 대신 "멋있는" 추천 금지).

---

## 2. Output format

Return EXACTLY ONE JSON object, with no additional text, no Markdown, no code fences.

For example:

{{
  "suspicion_score": 3,
  "content_typo_report": "",
  "translated_typo_report": "- '이점들을를'에서 오타 발견. '이점들을'로 수정해야 함.",
  "markdown_report": ""
}}

### Rules for suspicion_score
- 1: 보고할 만한 오류가 없을 때 (모든 리포트 필드가 "")
- 2~3: 소수의 명확한 오류가 있을 때
- 4~5: 다수의 오류 또는 전반적으로 품질이 의심될 때

### Rules for reports
- 각 리포트에는 반드시 **문제가 된 부분을 직접 인용**하고, 제안 수정안을 함께 제시한다.
- 한 줄에 하나의 오류를 `- `로 시작하는 bullet 형식으로 작성한다.
  - 예) "- '사과을'에서 목적격 조사 오류. '사과를'로 수정해야 함."

If there is NO objective error at all:
- "suspicion_score": 1
- "content_typo_report": ""
- "translated_typo_report": ""
- "markdown_report": ""

---

## 3. Text to review

Now apply all the rules above to the following Korean text:

- plain_korean: "{korean_text}"
- markdown_korean: "{korean_text}"
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
                generation_config=generation_config,
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
                    "markdown_report": "",
                }


def validate_and_clean_analysis(result: dict) -> dict:
    """AI 응답에서 문체 제안 등을 필터링하고 점수를 보정"""
    if not isinstance(result, dict):
        return {
            "suspicion_score": 5,
            "content_typo_report": "AI 응답이 유효한 JSON 형식이 아님",
            "translated_typo_report": "",
            "markdown_report": "",
        }

    score = result.get("suspicion_score")
    reports = {
        "content_typo_report": result.get("content_typo_report", "") or "",
        "translated_typo_report": result.get("translated_typo_report", "") or "",
        "markdown_report": result.get("markdown_report", "") or "",
    }

    # 스타일/문체 제안 금지 키워드
    forbidden_keywords = [
        "문맥상",
        "부적절",
        "어색",
        "더 자연스럽",
        "더 적절",
        "수정하는 것이 좋",
        "제안",
        "바꾸는 것",
        "의미를 명확히",
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

    # score 기본값 보정
    if score is None:
        score = 1

    # 리포트가 모두 비어 있으면 1점 강제
    if not final_content and not final_translated and not final_markdown:
        score = 1
    # 리포트가 있는데 1점이면 3점으로 보정
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

st.title("📚 AI 텍스트 검수기 (Gemini 기반)")
st.caption("오탈자 / 조사 / 띄어쓰기 / 형식 오류 + 영어 원지문 검수에만 집중하는 검수기 (스타일 제안 금지).")

tab_main, tab_sheet, tab_about, tab_debug = st.tabs(
    ["✏️ 텍스트 검수", "📄 시트 검수", "ℹ️ 설명", "🐞 디버그"]
)

with tab_sheet:
    st.subheader("📄 Google Sheets 시트 검수")

    spreadsheet_name = st.text_input(
        "스프레드시트 이름",
        value="[DATA] Paragraph DB (교과서 / 참고서 / 모의고사)",  # 네가 자주 쓰는 이름으로 기본값 설정
    )

    worksheet_name = st.text_input(
        "탭 이름(워크시트 이름)",
        value="22개정 / 최종데이터",
    )

    if st.button("이 시트 검수 실행", type="primary"):
        if not spreadsheet_name.strip() or not worksheet_name.strip():
            st.warning("스프레드시트 이름과 탭 이름을 모두 입력해주세요.")
        else:
            with st.spinner("시트 검수 중입니다... (행이 많으면 시간이 걸려요)"):
                try:
                    summary = run_sheet_review(spreadsheet_name, worksheet_name)
                except Exception as e:
                    st.error(f"실행 중 오류가 발생했습니다: {e}")
                else:
                    st.success("시트 검수가 완료되었습니다!")
                    st.metric("전체 행 수", summary.get("total_rows", 0))
                    st.metric("검수 대상 행 수 (STATUS=1. AI검수요청)", summary.get("target_rows", 0))
                    st.metric("실제 처리된 행 수", summary.get("processed_rows", 0))
                    st.info("Google Sheets에서 SCORE / *_REPORT / STATUS 컬럼을 확인해주세요.")


with tab_main:
    st.subheader("검수할 텍스트를 입력하세요")
    default_text = "이것은 테스트 문장 입니다. 그는는 학교에 갔다."
    text = st.text_area(
        "입력 텍스트",
        value=default_text,
        height=220,
        help="한국어 텍스트를 넣어주세요.",
    )

    if st.button("검수 실행", type="primary"):
        if not text.strip():
            st.warning("먼저 텍스트를 입력해주세요.")
        else:
            with st.spinner("AI가 검수 중입니다..."):
                result = review_text(text)

            score = result.get("score")
            content_report = result.get("content_typo_report")

            st.success("검수가 완료되었습니다!")

            if score is not None:
                st.metric("의심 점수 (1~5)", f"{score:.2f}")

            st.markdown("### 🔍 리포트")

            with st.expander("📄 입력 텍스트 검수 결과 리포트 (content_typo_report)"):
                if content_report.strip():
                    st.markdown(content_report)
                else:
                    st.info("보고할 오류가 없습니다.")


with tab_about:
    st.markdown("""
## 이 앱은?

- 텍스트 검수에 대한 통합 버전을 만들기 위한 기초 streamlit입니다.

### 텍스트 검수

- 한글 텍스트를 기입하면 AI를 통해 검수를 진행합니다.
- 아직 테스트 중으로 정확하게 잡아내진 못할 수 있으니 **주의**해주세요!

### 시트 검수 (영어 AI 검수)

- Gemini를 활용한 영어 원지문 AI 검수기입니다.
- **Gemini API + Streamlit**으로 감싼 텍스트 검수기입니다.
- 현재 버전은 "시트명 + 탭명" 기입을 기반으로 자동하는 것이 주 용도입니다.

### 동작

1. 사용자는 시트 검수 탭으로 이동해주세요.
2. 스프레드시트 이름과 탭명을 기입해주세요.
3. 기입 완료 후 **이 시트 검수 실행**을 눌러주세요.
4. 요청 행이 많으면 시간이 조금 걸릴 수 있어요.
5. 실행이 완료되면 **시트**로 이동해서 결과를 확인해주세요.


### score 정의

- 1: 오류 없음  
- 2~3: 소수의 명확한 오류  
- 4~5: 다수의 오류 또는 품질 매우 의심 
""")
with tab_debug:
    st.markdown("여기는 나중에 raw JSON을 보는 용도로 쓸 수 있습니다. (현재는 입력 후 콘솔 등으로 확인)")
