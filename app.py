# app.py
# -*- coding: utf-8 -*-
import json
import time
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

# --------------------------
# 공통: Gemini 호출 / 결과 정제
# --------------------------
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
                # 지수 백오프
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

    # 스타일/문체 제안 금지 키워드 (한국어 쪽)
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


# --------------------------
# 1-A. 한국어 단일 텍스트 검수 프롬프트 + 래퍼
# --------------------------
def create_korean_review_prompt_for_text(korean_text: str) -> str:
    """
    한국어 텍스트(문장/문단) 하나만 검수하는 프롬프트.
    - 오탈자 / 조사·어미 / 띄어쓰기 / 기본 문장부호 / 형태소 분리 / 반복 오타
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

For this task:
- Use "translated_typo_report" to report errors in the Korean text.
- "content_typo_report" should remain empty ("") unless you are explicitly asked to check English.

If there is nothing to report, each report field MUST be an empty string "" (do NOT write things like "오류 없음", "문제 없음", etc.).

---

## 1. What counts as an error in Korean?

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
   - 따옴표/쌍따옴표가 한쪽만 있거나 짝이 안 맞는 경우는 항상 오류.
   - 예)
     - 잘못된 예: 나는 말한다."
     - 올바른 예: "나는 말한다."

5. **Morpheme Split Errors (형태소 분리 오류)**  
   - 동사, 형용사, 어미, 조사 등 하나의 형태소로 결합되어야 하는 항목이 
     부적절하게 분리된 경우는 무조건 오류.
   - 예:
     - "묻 는" → "묻는"
     - "먹 는" → "먹는"
     - "잡 아" → "잡아"
     - "된 다" → "된다"
     - "간 다" → "간다"
   - 단, 한국어 맞춤법에서 두 형태 모두 허용되는 띄어쓰기(예: "해 보다"/"해보다")는 제외.

6. **Repetition Typos (반복 오타)**  
   - 유효한 한국어 단어를 이루지 못하는 음절/글자 반복은 항상 오타.
   - 예:
     - "된다따따." → "된다."
     - "합니다아아" → "합니다."
     - "간다다다" → "간다."
     
7. **마침표 ↔ 쉼표 오용 (MUST ALWAYS FLAG)**  

한국어에서도 다음은 모두 **명백한 문장부호 오류**임:

### 1) 마침표가 들어가야 하는데 쉼표를 사용한 경우  
예:  
- "나는 오늘 학교에 갔다, 그리고 집에 왔다."  
→ "나는 오늘 학교에 갔다. 그리고 집에 왔다."

### 2) 쉼표가 들어가야 하는데 마침표를 사용한 경우  
예:  
- "나는 밥을 먹었다. 그리고 물을 마셨다."  
(이건 자연스럽지만)  
- "나는 밥을 먹었다. 그리고"  
→ 문장 구조가 불완전 → 오류

### 3) 쉼표로 두 문장을 억지로 연결한 경우 (Comma splice)  
예:  
- "비가 온다, 나는 우산을 쓴다."  
→ "비가 온다. 나는 우산을 쓴다."

### 4) 문장 끝에 쉼표가 있는 경우  
예:  
- "나는 간다," → "나는 간다."

### 5) 연결 어미 앞에서 잘못된 구두점  
예:  
- "나는 간다. 그리고 학교에 간다."  
→ ‘그리고’ 앞에서는 마침표 대신 쉼표가 더 적절한 문장 구조 → 오류로 처리

---

## 2. Output format

Return EXACTLY ONE JSON object, with no additional text, no Markdown, no code fences.

For example:

{{
  "suspicion_score": 3,
  "content_typo_report": "",
  "translated_typo_report": "- '사과을'에서 목적격 조사 오류. '사과를'로 수정해야 함.",
  "markdown_report": ""
}}

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


def review_korean_text(korean_text: str) -> dict:
    """한국어 텍스트 검수 래퍼"""
    prompt = create_korean_review_prompt_for_text(korean_text)
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
# 1-B. 영어 단일 텍스트 검수 프롬프트 + 래퍼
# --------------------------
def create_english_review_prompt_for_text(english_text: str) -> str:
    """
    영어 단일 문장/문단을 검수하는 완전 강화 프롬프트.
    - 스펠링 / AI↔Al 오타 / 대문자 규칙 / 쉼표↔마침표 오용 / 기본 문장부호 / 공백 오류 / 중복 단어
    - 모든 리포트는 한국어로 작성해야 한다.
    """
    prompt = f"""
You are a machine-like **English text proofreader**.
Your ONLY job is to detect **objective, verifiable errors** in the following English text.
You are strictly forbidden from suggesting stylistic improvements, rewriting, rephrasing, or judging naturalness.

Your output MUST be a single valid JSON object:

- "suspicion_score": integer (1~5)
- "content_typo_report": string
- "translated_typo_report": string (always "")
- "markdown_report": string (always "")

All explanation MUST be in **Korean**, never English.

If no errors exist, all *_report fields MUST be empty strings "".

---

# 1. 반드시 감지해야 할 영어 오류 규칙 (ABSOLUTE REQUIREMENTS)

## (A) **Spelling / Typo Errors (MUST detect ALL)**

You MUST treat a token as a spelling error if:

1. It is very similar to a valid English word  
   (1–2 letters missing, added, swapped, or wrong), AND  
2. It is NOT a proper noun, acronym, technical token, filename, or code.

Examples (patterns, NOT an exhaustive list):

1. recieve → receive  
2. enviroment → environment  
3. understaning → understanding  
4. langauge → language  
5. problme → problem  
6. definately → definitely  
7. seperated → separated  
8. occured → occurred  
9. adress → address  
10. wierd → weird  
11. becuase → because  
12. comming → coming  
13. teh → the  
14. sytem → system  

MUST ALWAYS FLAG lower-case “i” used for the pronoun “I”.

❗ 예시 문장 분석:
- "This is a simple understaning of the AI model."  
  → MUST detect understaning → understanding

---

## (B) **AI 문맥에서 Al → AI (MUST ALWAYS FLAG)**

If the sentence clearly refers to artificial intelligence (model, system, learning, LLM, agent, chatbot):

- “Al” (A + lowercase L) MUST be treated as a typo of “AI”.

Examples:
- Al model → AI model  
- modern Al technology → modern AI technology  
- Al system learns → AI system learns  

---

## (C) **Capitalization Errors (MUST detect)**

You MUST flag:
1. Sentence starting with lowercase  
   - “this is…” → “This is…”
2. Pronoun “I” in lowercase  
   - “i do not” → “I do not”
3. Proper nouns without capitalization  
   - “london” → “London”
   - “korea” → “Korea”

---

## (D) **Basic punctuation errors (MUST detect)**

You MUST detect:

1. Missing period at the end of a full sentence  
2. Missing comma after introductory elements  
3. Broken quotation marks  
4. Two sentences joined without punctuation  
5. Double punctuation (“..”), wrong punctuation marks (!?, ?!, ,.)  

---

## (E) **Period ↔ Comma Misplacement (MUST detect ALL cases)**

You MUST flag:

### 1) 쉼표가 마침표 자리에서 사용됨
- "He is here, This is wrong."  
  → Should be two sentences.

### 2) 마침표가 쉼표 자리에서 사용됨  
- "He slept. and I worked."  
  → Should be “He slept, and I worked.”

### 3) **Comma splice** (MUST flag always)
- “I finished the task, It was easy.”  
  → MUST treat as an objective grammar error.

### 4) Sentence-ending comma
- "He is here,"  
  → Should be “He is here.”

### 5) Incorrect punctuation before conjunction  
- “I ate lunch. and I left.”  
  → Must be a comma, not a period.

---

## (F) **Spacing / duplication errors (MUST detect)**

- “re turn” → “return”  
- “mod el” → “model”  
- “the the” → “the”  
- “AI  model” (double space) → “AI model”  

---

## (G) **Markdown mismatch**  
Always flag if markdown text differs from plain text.

---

# 2. Output Format Rules (VERY IMPORTANT)

- All reports MUST be written in Korean.
- Each bullet MUST follow the format:

“- 'wrong' → 'correct': 'wrong'은(는) ~ 오류이며, 'correct'로 수정해야 합니다.”

- suspicion_score =  
  - 1 → 오류 없음  
  - 2~3 → 경미한 오류  
  - 4~5 → 다수 또는 심각한 오류  

---

# 3. Text to review

plain_english: "{english_text}"
markdown_english: "{english_text}"

---

# 4. Self-check requirement (MUST FOLLOW)

If the input contains ANY of the following:

- understaning  
- langauge  
- problme  
- Al model  
- i do not  
- He slept. and I worked.  
- This is wrong, This is wrong.

You MUST ALWAYS flag them as objective errors.

"""
    return prompt





def review_english_text(english_text: str) -> dict:
    """영어 텍스트 검수 래퍼"""
    prompt = create_english_review_prompt_for_text(english_text)
    raw = analyze_text_with_gemini(prompt)
    cleaned = validate_and_clean_analysis(raw)
    return {
        "score": cleaned.get("suspicion_score"),
        "영어 문장 검수 결과": cleaned.get("content_typo_report", ""),
        "markdown_report": cleaned.get("markdown_report", ""),
        "raw": raw,  # 디버깅용
    }
    
def summarize_json_diff(raw: dict | None, final: dict | None) -> str:
    """
    raw와 final JSON(dict)을 비교해서
    - 값이 달라진 key만 bullet로 뽑아주는 간단 diff 요약.
    """
    if not isinstance(raw, dict):
        raw = {}
    if not isinstance(final, dict):
        final = {}

    lines = []
    all_keys = sorted(set(raw.keys()) | set(final.keys()))

    for key in all_keys:
        rv = raw.get(key, "<없음>")
        fv = final.get(key, "<없음>")
        if rv == fv:
            continue

        # 보기 좋게 문자열로 캐스팅
        rv_str = json.dumps(rv, ensure_ascii=False) if isinstance(rv, (dict, list)) else str(rv)
        fv_str = json.dumps(fv, ensure_ascii=False) if isinstance(fv, (dict, list)) else str(fv)

        lines.append(
            f"- **{key}**\n"
            f"  - raw: `{rv_str}`\n"
            f"  - final: `{fv_str}`"
        )

    if not lines:
        return "차이가 없습니다. (raw와 final이 동일합니다.)"

    return "\n".join(lines)

def extract_korean_suggestions_from_raw(raw: dict) -> list[str]:
    """
    raw JSON 전체에서 오류 설명을 추출하여 bullet list로 변환한다.
    포함 대상:
    - translated_typo_report
    - content_typo_report
    - markdown_report (한국어 오류 관련 내용이 있을 때만)
    """

    if not isinstance(raw, dict):
        return []

    collected = []

    # 1️⃣ 한국어 오류가 들어가는 주요 보고 필드들
    fields = [
        raw.get("translated_typo_report", ""),
        raw.get("content_typo_report", ""),
        raw.get("markdown_report", ""),
    ]

    for block in fields:
        if not block:
            continue
        
        # 각 필드 내 줄 단위 추출
        for line in block.split("\n"):
            line = line.strip()
            if not line:
                continue

            # bullet 없는 라인도 bullet 형태로 정규화
            if not line.startswith("- "):
                line = f"- {line}"

            collected.append(line)

    return collected



def extract_english_suggestions_from_raw(raw: dict) -> list[str]:
    """
    raw JSON 전체에서 '영어 원문'에 대한 오류 설명을 추출하여
    bullet 리스트로 변환한다.

    포함 대상 필드:
    - content_typo_report: 영어 원문(English) 관련 오류 설명 (한국어로 기술)
    - translated_typo_report: 예외적으로 영어 관련 내용이 들어갈 수도 있어 보조로 포함
    - markdown_report: 마크다운 변환 과정에서 발생한 영어 텍스트 오류가 있을 수 있음

    반환 형식:
    - 각 요소는 반드시 '- '로 시작하는 한 줄짜리 문자열
    """
    if not isinstance(raw, dict):
        return []

    collected: list[str] = []

    # 1️⃣ 영어 원문 쪽 오류가 담길 수 있는 필드들
    fields = [
        raw.get("content_typo_report", ""),
        raw.get("translated_typo_report", ""),
        raw.get("markdown_report", ""),
    ]

    for block in fields:
        if not block:
            continue

        # 각 필드를 줄 단위로 분해 후 정리
        for line in block.split("\n"):
            line = line.strip()
            if not line:
                continue

            # 이미 "- "로 시작하지 않으면 bullet로 감싸기
            if not line.startswith("- "):
                line = f"- {line}"

            collected.append(line)

    return collected





# --------------------------
# 2. Streamlit UI
# --------------------------
st.set_page_config(
    page_title="AI 검수기 (Gemini)",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI 텍스트 검수기 (Gemini 기반)")
st.caption("한국어/영어 단일 텍스트 + Google Sheets 기반 검수기 (오탈자/형식 위주, 스타일 제안 금지).")

tab_ko, tab_en, tab_sheet, tab_about, tab_debug = st.tabs(
    ["✏️ 한국어 검수", "✏️ 영어 검수", "📄 시트 검수", "ℹ️ 설명", "🐞 디버그"]
)

# --- 한국어 검수 탭 ---
with tab_ko:
    st.subheader("한국어 텍스트 검수")
    default_ko = "이것은 테스트 문장 입니다, 그는.는 학교에 갔다,"
    text_ko = st.text_area(
        "한국어 텍스트 입력",
        value=default_ko,
        height=220,
    )

    if st.button("한국어 검수 실행", type="primary"):
        if not text_ko.strip():
            st.warning("먼저 한국어 텍스트를 입력해주세요.")
        else:
            with st.spinner("AI가 한국어 텍스트를 검수 중입니다..."):
                result = review_korean_text(text_ko)

            # ✅ 최신 결과를 세션에 저장
            st.session_state["ko_result"] = result

    # ✅ 세션에 결과가 있으면 항상 아래를 보여줌
    if "ko_result" in st.session_state:
        result = st.session_state["ko_result"]
        score = result.get("score", 1)

        # 🔹 raw 전체 JSON (모델이 준 원본)
        raw_json = result.get("raw", {}) or {}

        # 🔹 final: 한국어 단일 텍스트에 필요한 필드만
        final_json = {
            "의심 점수": result.get("score", 1),
            "한국어 검수 결과": result.get("translated_typo_report", ""),
        }

        # 🔹 raw도 비교 키만 슬림하게 잘라서 보기 좋게
        raw_view = {
            "의심 점수": raw_json.get("suspicion_score"),
            "한국어 검수 결과": raw_json.get("translated_typo_report", ""),
        }

        st.success("한국어 검수가 완료되었습니다!")
        st.metric("의심 점수 (1~5)", f"{score:.2f}")

        st.markdown("### 🔍 결과 비교 (Raw vs Final)")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### ✅ Final JSON (한국어 입력 기준 최소 필드)")
            st.json(final_json)

        with col2:
            st.markdown("#### 🧪 Raw JSON (동일 필드만 발췌)")
            st.json(raw_view)

        # 🔍 Diff 요약
        st.markdown("#### 🔍 Raw vs Final 차이 요약")
        diff_md = summarize_json_diff(raw_view, final_json)
        st.markdown(diff_md)

        raw = result.get("raw", {})
        
        # 🛠 최종 수정 제안 사항
        st.markdown("### 🛠 최종 수정 제안 사항")
        suggestions = extract_korean_suggestions_from_raw(raw)

        if not suggestions:
            st.info("보고할 수정 사항이 없습니다.")
        else:
            for s in suggestions:
                st.markdown(f"- {s}")




# --- 영어 검수 탭 ---
with tab_en:
    st.subheader("영어 텍스트 검수")
    default_en = "This is a simple understaning of the Al model."
    text_en = st.text_area(
        "English text input",
        value=default_en,
        height=220,
    )

    if st.button("영어 검수 실행", type="primary"):
        if not text_en.strip():
            st.warning("먼저 영어 텍스트를 입력해주세요.")
        else:
            with st.spinner("AI가 영어 텍스트를 검수 중입니다..."):
                result = review_english_text(text_en)

            st.session_state["en_result"] = result

    if "en_result" in st.session_state:
        result = st.session_state["en_result"]
        score = result.get("score", 1)

        raw_json = result.get("raw", {}) or {}

        # 🔹 final: 영어 단일 텍스트에 필요한 필드만
        final_json = {
            "의심 점수": result.get("score", 1),
            "영어 검수 결과": result.get("content_typo_report", ""),
        }

        # 🔹 raw도 동일 키만 추려서 보기 좋게
        raw_view = {
            "의심 점수": raw_json.get("suspicion_score"),
            "영어 검수 결과": raw_json.get("content_typo_report", ""),
        }

        st.success("영어 검수가 완료되었습니다!")
        st.metric("Suspicion score (1~5)", f"{score:.2f}")

        st.markdown("### 🔍 결과 비교 (Raw vs Final)")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### ✅ Final JSON (영어 입력 기준 최소 필드)")
            st.json(final_json)

        with col2:
            st.markdown("#### 🧪 Raw JSON (동일 필드만 발췌)")
            st.json(raw_view)

        st.markdown("#### 🔍 Raw vs Final 차이 요약")
        diff_md = summarize_json_diff(raw_view, final_json)
        st.markdown(diff_md)
        
        raw = result.get("raw", {})
        
         # 🛠 최종 수정 제안 사항
        st.markdown("### 🛠 최종 수정 제안 사항 (영어 원문 기준)")
        suggestions = extract_english_suggestions_from_raw(raw)

        if not suggestions:
            st.info("보고할 수정 사항이 없습니다.")
        else:
            for s in suggestions:
                st.markdown(f"- {s}")




# --- 시트 검수 탭 ---
with tab_sheet:
    st.subheader("📄 Google Sheets 시트 검수")

    spreadsheet_name = st.text_input(
        "스프레드시트 이름",
        value="[DATA] Paragraph DB (교과서 / 참고서 / 모의고사)",
    )

    worksheet_name = st.text_input(
        "탭 이름(워크시트 이름)",
        value="22개정 / 최종데이터",
    )

    # 🔹 1) 실행 버튼
    run_clicked = st.button("이 시트 검수 실행", type="primary")

    # 🔹 2) 버튼 눌렀을 때만 실제 검수 실행 + progress 표시
    if run_clicked:
        if not spreadsheet_name.strip() or not worksheet_name.strip():
            st.warning("스프레드시트 이름과 탭 이름을 모두 입력해주세요.")
        else:
            # 진행도 UI
            progress_bar = st.progress(0.0)
            progress_text = st.empty()

            def progress_callback(done: int, total: int):
                ratio = done / total if total > 0 else 0
                remaining = total - done
                progress_bar.progress(ratio)
                progress_text.text(
                    f"진행도: {done} / {total} 행 처리 완료 (남은 행: {remaining})"
                )

            with st.spinner("시트 검수 중입니다... (행이 많으면 시간이 걸려요)"):
                try:
                    summary = run_sheet_review(
                        spreadsheet_name,
                        worksheet_name,
                        collect_raw=True,
                        progress_callback=progress_callback,
                    )
                except Exception as e:
                    st.error(f"실행 중 오류가 발생했습니다: {e}")
                else:
                    # 진행바 100%
                    progress_bar.progress(1.0)
                    progress_text.text("진행도: 모든 대상 행 처리 완료 ✅")

                    # ✅ 결과를 SessionState에 저장
                    st.session_state["sheet_summary"] = summary
                    st.session_state["raw_results"] = summary.get("raw_results", [])

    # 🔹 3) 여기부터는 "버튼을 누른 적이 있다면" 저장된 결과를 항상 다시 보여준다.
    summary = st.session_state.get("sheet_summary")
    raw_results = st.session_state.get("raw_results", [])

    if summary:
        total_rows = summary.get("total_rows", 0)
        target_rows = summary.get("target_rows", 0)
        processed_rows = summary.get("processed_rows", 0)
        remaining_rows = max(target_rows - processed_rows, 0)

        st.success("시트 검수가 완료되었습니다! (마지막 실행 기준)")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("전체 행 수", total_rows)
        with col_m2:
            st.metric("검수 대상 행 수", target_rows)
        with col_m3:
            st.metric("실제 처리된 행 수", processed_rows)
        with col_m4:
            st.metric("남은 행 수", remaining_rows)

        st.info("Google Sheets에서 SCORE / *_REPORT / STATUS 컬럼을 확인해주세요.")

    st.markdown("### 🐞 디버그: 특정 행의 Raw / Final JSON & Diff")

    if not raw_results:
        st.info("아직 raw 결과가 없습니다. 먼저 시트 검수를 실행해주세요.")
    else:
        # 시트 실제 행 번호 리스트 (2,3,4,...)
        row_numbers = [item["sheet_row_index"] for item in raw_results]

        # 🔹 selectbox는 단순히 session_state에 '후보'를 저장
        st.selectbox(
            "Raw/Final JSON을 보고 싶은 행 번호를 선택하세요:",
            options=row_numbers,
            key="selected_row_candidate",
            format_func=lambda x: f"행 {x}번",
        )

        # 🔹 이 버튼을 눌렀을 때만 실제로 반영
        if st.button("이 행의 JSON 보기"):
            st.session_state["selected_row"] = st.session_state["selected_row_candidate"]

        selected_row = st.session_state.get("selected_row")

        if selected_row is not None:
            selected_item = next(
                (item for item in raw_results if item["sheet_row_index"] == selected_row),
                None,
            )

            if selected_item:
                col_final, col_raw = st.columns(2)

                with col_final:
                    st.markdown(f"#### ✅ Final JSON (행 {selected_row})")
                    st.json(selected_item.get("final"))

                with col_raw:
                    st.markdown(f"#### 🧪 Raw JSON (행 {selected_row})")
                    st.json(selected_item.get("raw"))

                # Diff 요약
                st.markdown("#### 🔍 Raw vs Final 차이 요약")
                diff_md = summarize_json_diff(
                    selected_item.get("raw"),
                    selected_item.get("final"),
                )
                st.markdown(diff_md)
            else:
                st.info("선택한 행의 Raw/Final 데이터가 없습니다.")



# --- 설명 탭 ---
with tab_about:
    st.markdown("""
## 이 앱은?

- 한국어/영어 **단일 텍스트 검수기** + **Google Sheets 기반 배치 검수기**입니다.
- 스타일/어투/자연스러움은 건드리지 않고, **오탈자 / 조사 / 띄어쓰기 / 기본 문장부호 / 단순 스펠링 오류**에만 집중합니다.

### 탭 설명

- **✏️ 한국어 검수**: 한국어 문장/문단 하나를 넣으면,
  - 형태소 분리 오류(예: `된 다`, `묻 는`)
  - 반복 오타(예: `된다따따.`)
  - 조사/어미/띄어쓰기 오류
  - 따옴표 짝 불일치
  - 마침표, 쉼표 검수
  등을 중심으로 검수합니다.
  
  ** 12/4 업데이트 내용**
  - 모델이 실제 검수한 결과와, 필터링 되어서 나오는 결과를 비교할 수 있게 됐어요.
  - 간혹 과하게 검수가 된 경우도 있으니 참고해주세요.

- **✏️ 영어 검수**: 영어 문장/문단 하나를 넣으면,
  - 스펠링 typo (예: `understaning` → `understanding`)
  - 중복 단어 (`the the`)
  - 잘못된 띄어쓰기 (`re turn` → `return`)
  - AI 문맥에서 `Al` → `AI` 오타
  - 마침표, 쉼표 검수
  등을 중심으로 검수합니다.
  
   ** 12/4 업데이트 내용**
  - 모델이 실제 검수한 결과와, 필터링 되어서 나오는 결과를 '비교'할 수 있게 됐어요.
  - 간혹 '과하게 검수'가 된 경우도 있으니 참고해주세요.

- **📄 시트 검수**: Google Sheets에 있는
  - 영어 원문 / 마크다운
  - 한국어 번역 / 마크다운
  을 row 단위로 읽어서, 시트에 SCORE / *_REPORT / STATUS를 채워넣습니다.
  
  **[12/4] 업데이트 내용**
  - 시트에서 검수하고 있는 진행 상태를 볼 수 있어요.
  - 실행된 행을 Select box에서 골라서 검수 내역을 확인할 수 있어요.
  - 모델이 준 결과와 필터링된 결과를 비교할 수 있어요.
""")


# --- 디버그 탭 ---
with tab_debug:
    st.markdown("여기는 추후에 로그, 디버그용 정보를 추가로 표시할 수 있는 영역입니다.")
    
    