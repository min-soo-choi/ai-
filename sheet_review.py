# sheet_review.py
# -*- coding: utf-8 -*-
import json
import time

import streamlit as st
import pandas as pd
import gspread
import google.generativeai as genai
from google.oauth2.service_account import Credentials

# ---------------------------------------------------
# 1. Gemini / Google Sheets 클라이언트 설정
# ---------------------------------------------------

# Gemini 키 (secrets에서 읽기)
API_KEY = st.secrets.get("GEMINI_API_KEY")
if not API_KEY:
    st.error("GEMINI_API_KEY가 secrets에 설정되어 있지 않습니다.")
    st.stop()

genai.configure(api_key=API_KEY)
MODEL_ID = "gemini-2.0-flash-001"
model = genai.GenerativeModel(MODEL_ID)

# 서비스 계정 정보 (JSON 전체를 secrets에 넣어둠)
service_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_JSON"])

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

creds = Credentials.from_service_account_info(service_info, scopes=SCOPES)
gs_client = gspread.authorize(creds)

# ---------------------------------------------------
# 2. 시트 컬럼 이름 (네 기존 스키마 그대로)
# ---------------------------------------------------

STATUS_COL = "STATUS"
ORIGINAL_TEXT_COL = "content"
ORIGINAL_MD_COL = "content_markdown"
TRANSLATION_TEXT_COL = "content_translated"
TRANSLATION_MD_COL = "content_markdown_translated"

SUSPICION_SCORE_COL = "SCORE"
CONTENT_TYPO_REPORT_COL = "CONTENT_TYPO_REPORT"
TRANSLATED_COL = "TRANSLATED_TYPO_REPORT"
MARKDOWN_REPORT_COL = "MARKDOWN_REPORT"


# ---------------------------------------------------
# 3. 프롬프트 / 모델 호출 / 결과 정제
# ---------------------------------------------------

def create_review_prompt(row: dict) -> str:
    original_text = row.get(ORIGINAL_TEXT_COL, "")
    original_md = row.get(ORIGINAL_MD_COL, "")
    translation_text = row.get(TRANSLATION_TEXT_COL, "")
    translation_md = row.get(TRANSLATION_MD_COL, "")

    prompt = f"""
    You are a machine-like **Data Verifier**. Your ONLY job is to find **objective, factual errors** 
    in both the English source text and the Korean translated text. 
    You are strictly forbidden from judging style, meaning, or making subjective suggestions. 
    Your output MUST BE a single, valid JSON object.

    Your JSON MUST have exactly the following keys:
    - "suspicion_score": an integer between 1 and 5 (1 = almost certainly no error, 5 = very likely serious errors)
    - "content_typo_report": a string (may be empty "")
    - "translated_typo_report": a string (may be empty "")
    - "markdown_report": a string (may be empty "")

    Use the fields as follows:
    - **content_typo_report**: objective errors in the **English source text** (plain_english / markdown_english).
    - **translated_typo_report**: objective errors in the **Korean translated text** (plain_korean / markdown_korean).
    - **markdown_report**: pure **markdown vs plain-text mismatches** (missing words, extra words, broken formatting) 
      for either English or Korean.

    ---

    ## 1. What counts as an objective error?

    You must ONLY report the following error types.

    ### 1-A. For English (plain_english / markdown_english)

    1. **Spelling / Typos (VERY IMPORTANT)**
       - Any obviously misspelled English word MUST be treated as an error,
         not only specific examples.
       - Treat a token as a spelling typo if:
         - It is very similar to a common English word (1–2 letters missing, added, swapped, or wrong),
           AND
         - It is not clearly a proper noun, acronym, variable name, or chemical formula.
       - Examples (these are patterns, NOT an exhaustive list):
         - "recieve"  → "receive"
         - "enviroment" → "environment"
         - "understaning" → "understanding"
         - "langauge" → "language"
         - "teh" → "the"
         - "problme" → "problem"

       - Counter-examples (DO NOT mark these as spelling errors):
         - Proper nouns or product names: "OpenAI", "ChatGPT", "PyTorch"
         - Technical tokens / code / formulas: "int64", "Al2O3", "NaCl"

    2. **Obvious spacing / duplication errors**
       - Accidental extra spaces inside a word, or duplicated words.
       - Examples:
         - "re turn" → "return"
         - "the the" → "the"

       3. **AI vs Al typo in AI-related context**
       - In contexts clearly about artificial intelligence (e.g. "model", "system", "tool",
         "chatbot", "LLM", "agent", "neural network"), the token "Al"
         (capital A + lowercase L) is almost always a typo for "AI".
       - In such contexts, you MUST treat "Al" as a spelling error and correct it to "AI".
       - Examples:
         - "Al model"   → "AI model"
         - "modern Al technology" → "modern AI technology"
         - "Al chatbot" → "AI chatbot"

    4. **Plain vs Markdown content mismatch (English)**
       - A word or phrase is missing in markdown, duplicated, or obviously wrong compared to the plain version.
       - Example:
         - plain_english: "He went to school yesterday."
         - markdown_english: "He went school yesterday."
         → Missing "to" is an objective mismatch.

    ### 1-B. For Korean (plain_korean / markdown_korean)

    1.  **Typos (오탈자)**  
        - 잘못된 철자, 중복 글자, 명백한 입력 실수  
        - 예: "이점들을를" → "이점들을"

    2.  **Grammatical Errors (조사, 어미)**  
        - 주격/목적격/보격/부사격 조사 잘못 사용  
        - 동사/형용사 어미가 문법적으로 분명히 잘못된 경우  
        - 예: "사과을" → "사과를"

    3.  **Spacing (띄어쓰기) errors**  
        - 띄어쓰기/붙여쓰기 규범이 명백히 잘못된 경우  
        - 예: "책을읽고" → "책을 읽고"

    4.  **Basic punctuation (기본 문장부호) errors**  
        - 마침표/쉼표/물음표 등 필수 문장부호가 빠져
          문장이 비문이 되거나 구조가 심각하게 모호한 경우.
        - 따옴표/쌍따옴표가 한쪽만 있거나 짝이 안 맞는 경우는 **항상 오류**이다.
        - 문단 첫 번째 문장에서는 문장 부호 누락 여부를 특히 주의해서 확인한다.
        - 예:
          - 잘못된 예: 나는 말한다."
          - 올바른 예: "나는 말한다."

    5.  **Morpheme Split Errors (형태소 분리 오류)**  
        - 동사, 형용사, 어미, 조사 등 하나의 형태소로 결합되어야 하는 항목이 
          부적절하게 분리된 경우는 **무조건 오류**로 판단한다. 
        - 예:
          - "묻 는" → "묻는"
          - "먹 는" → "먹는"
          - "잡 아" → "잡아"
          - "된 다" → "된다"
          - "간 다" → "간다"
        - 단, 한국어 맞춤법에서 두 형태 모두 허용되는 띄어쓰기(예: "해 보다"/"해보다")는 제외한다.

    6.  **Repetition Typos (반복 오타)**  
        - 유효한 한국어 단어를 이루지 못하는 음절/글자 반복은 **항상 오타**로 판단한다.
        - 예:
          - "된다따따." → "된다."
          - "합니다아아" → "합니다."
          - "간다다다" → "간다."

    7.  **Plain vs Markdown content mismatch (Korean)**  
        - plain_korean과 markdown_korean 사이에 단어가 빠지거나, 잘못 추가되거나, 
          명백히 다른 내용이 있을 때만 보고한다.

    ---

    ## 2. CRITICAL RULES OF ENGAGEMENT (for BOTH English and Korean)

    1.  **ABSOLUTELY NO STYLISTIC FEEDBACK:** 
        Do NOT suggest alternative wording (e.g., "근사한" vs "멋있는", 
        "big" vs "large"). Do not comment on what sounds "more natural" or "more appropriate". 
        This is a critical failure.

    2.  **SILENCE ON PERFECTION:** 
        If no objective errors are found for a given field, its report MUST be an empty string (`""`). 
        Do not write phrases like "오류 없음", "문제 없음", "정상", "no issues", etc.

    3.  **RESPECT STYLING CONVENTIONS:** 
        It is CORRECT for *italicized English* to be represented with 'single quotes' or 《double angle brackets》 
        in Korean. This is NOT an error.

    4.  **GROUND YOUR FINDINGS:** 
        When reporting an error, you MUST quote the problematic text and provide the corrected form.

    5.  **NO IDENTICAL CORRECTIONS:** 
        A suggested correction must be different from the original text.

    ---

    ## 3. EXAMPLES OF CORRECT EXECUTION

    **Example: English typo**
    - plain_english: "We can easily understaning the data."
    - Correct JSON (excerpt):
    {{
        "content_typo_report": "- 'understaning' is a spelling mistake. It must be 'understanding'.",
        ...
    }}

    **Example: AI vs Al typo**
    - plain_english: "Our Al model learns from data."
    - Correct JSON (excerpt):
    {{
        "content_typo_report": "- In 'Al model', 'Al' is a typo in an AI context. It must be 'AI model'.",
        ...
    }}

    **Example: Korean repetition typo**
    - plain_korean: "된다따따."
    - Correct JSON (excerpt):
    {{
        "translated_typo_report": "- '된다따따.'에서 불필요한 반복 '따따'가 있음. '된다.'로 수정해야 함.",
        ...
    }}

    **Example: Korean morpheme split**
    - plain_korean: "그렇게 된 다."
    - Correct JSON (excerpt):
    {{
        "translated_typo_report": "- '된 다'에서 형태소 분리 오류. '된다'로 붙여 써야 함.",
        ...
    }}

    **Example: Unbalanced quotes**
    - plain_korean: 나는 말한다."
    - Correct JSON (excerpt):
    {{
        "translated_typo_report": "- 따옴표가 한쪽만 있음. '\"나는 말한다.\"'처럼 시작과 끝을 모두 써야 함.",
        ...
    }}

    ---

    ## 4. ANALYSIS WORKFLOW

    Now, apply these strict rules and examples to the following data.

    **Data to Review:**
    - `plain_english`: "{original_text}"
    - `markdown_english`: "{original_md}"
    - `plain_korean`: "{translation_text}"
    - `markdown_korean`: "{translation_md}"
    """
    return prompt


def analyze_text_with_gemini(prompt: str, max_retries: int = 5) -> dict:
    for attempt in range(max_retries):
        try:
            generation_config = {
                "response_mime_type": "application/json",
                "temperature": 0.0,
            }
            response = model.generate_content(prompt, generation_config=generation_config)
            return json.loads(response.text)
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"Gemini 호출 오류 (시도 {attempt+1}/{max_retries}): {e} → {wait_time}초 후 재시도")
                time.sleep(wait_time)
            else:
                print("최대 재시도 횟수 초과.")
                return {
                    "suspicion_score": 5,
                    "content_typo_report": f"API 호출 실패: {str(e)}",
                    "translated_typo_report": "",
                    "markdown_report": "",
                }


def validate_and_clean_analysis(result: dict) -> dict:
    if not isinstance(result, dict):
        return {
            "suspicion_score": 5,
            "content_typo_report": "AI 응답이 유효한 JSON 형식이 아님",
            "translated_typo_report": "",
            "markdown_report": "",
        }

    score = result.get("suspicion_score")
    reports = {
        "content_typo_report": result.get("content_typo_report", ""),
        "translated_typo_report": result.get("translated_typo_report", ""),
        "markdown_report": result.get("markdown_report", ""),
    }

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


# ---------------------------------------------------
# 4. 공개 함수: 시트 전체를 돌리고 요약 리턴
# ---------------------------------------------------

def run_sheet_review(spreadsheet_name: str,
                     worksheet_name: str,
                     collect_raw: bool = False, 
                     progress_callback=None,) -> dict:
    """
    - 주어진 스프레드시트 / 워크시트에서
    - STATUS == '1. AI검수요청' 인 행만 골라서
    - SCORE / *_REPORT / STATUS를 채워넣는다.

    반환값: {
      "total_rows": ...,
      "target_rows": ...,
      "processed_rows": ...,
      "raw_results": [  # collect_raw=True일 때만
          {
            "sheet_row_index": int,
            "raw": {...},      # validate 전 원본 JSON
            "final": {...},    # validate_and_clean_analysis 이후
          },
          ...
      ]
    }
    """
    try:
        spreadsheet = gs_client.open(spreadsheet_name)
    except gspread.exceptions.SpreadsheetNotFound:
        raise ValueError(f"스프레드시트를 찾을 수 없습니다: {spreadsheet_name}")

    try:
        worksheet = spreadsheet.worksheet(worksheet_name)
    except gspread.exceptions.WorksheetNotFound:
        raise ValueError(f"워크시트를 찾을 수 없습니다: {worksheet_name}")

    all_data = worksheet.get_all_records()
    df = pd.DataFrame(all_data)
    df["sheet_row_index"] = df.index + 2  # 1행은 헤더라서 +2

    targets = df[df[STATUS_COL] == "1. AI검수요청"].copy()
    if targets.empty:
        return {
            "total_rows": len(df),
            "target_rows": 0,
            "processed_rows": 0,
            "raw_results": [],
        }

    results = []
    raw_results = []  # 🔹 디버그용

    total_targets = len(targets)


    for i, (_, row) in enumerate(targets.iterrows(), start=1):
        row_dict = row.to_dict()
        row_idx = row["sheet_row_index"]
        print(f"행 {row_idx} 검수 중... ({i}/{total_targets})")
        
        if progress_callback is not None:
            # progress_callback(처리한 개수, 전체 개수)
            progress_callback(i, total_targets)

        prompt = create_review_prompt(row_dict)
        raw = analyze_text_with_gemini(prompt)
        final = validate_and_clean_analysis(raw)

        results.append(
            {
                "sheet_row_index": row_idx,
                SUSPICION_SCORE_COL: final.get("suspicion_score"),
                CONTENT_TYPO_REPORT_COL: final.get("content_typo_report"),
                TRANSLATED_COL: final.get("translated_typo_report"),
                MARKDOWN_REPORT_COL: final.get("markdown_report"),
                STATUS_COL: "2. AI검수완료",
            }
        )
        
         # 스트림릿에서 볼 raw 디버그용
        if collect_raw:
            raw_results.append(
                {
                    "sheet_row_index": row_idx,
                    "raw": raw,
                    "final": final,
                }
            )

        time.sleep(0.5)  # API 과다 호출 방지용 (필요시 조정)

    # === 시트에 결과 반영 ===
    headers = worksheet.row_values(1)
    score_col_idx = headers.index(SUSPICION_SCORE_COL) + 1
    content_col_idx = headers.index(CONTENT_TYPO_REPORT_COL) + 1
    translated_col_idx = headers.index(TRANSLATED_COL) + 1
    markdown_col_idx = headers.index(MARKDOWN_REPORT_COL) + 1
    status_col_idx = headers.index(STATUS_COL) + 1

    def sanitize(v):
        return "" if v is None else str(v)

    update_cells = []
    for r in results:
        ridx = r["sheet_row_index"]
        update_cells.append(gspread.Cell(ridx, score_col_idx, sanitize(r[SUSPICION_SCORE_COL])))
        update_cells.append(gspread.Cell(ridx, content_col_idx, sanitize(r[CONTENT_TYPO_REPORT_COL])))
        update_cells.append(gspread.Cell(ridx, translated_col_idx, sanitize(r[TRANSLATED_COL])))
        update_cells.append(gspread.Cell(ridx, markdown_col_idx, sanitize(r[MARKDOWN_REPORT_COL])))
        update_cells.append(gspread.Cell(ridx, status_col_idx, sanitize(r[STATUS_COL])))

    if update_cells:
        worksheet.update_cells(update_cells)

    return {
        "total_rows": len(df),
        "target_rows": len(targets),
        "processed_rows": len(results),
        "raw_results": raw_results,   # 🔹 여기 추가

    }
