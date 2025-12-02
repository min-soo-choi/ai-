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
    6. **Morpheme Split Errors:** If a verb, adjective, ending, or particle that must remain a single morpheme is incorrectly split, it must always be treated as an error. 
   Examples: "묻 는", "먹 는", "잡 아"
   However, spacing cases where both forms are officially acceptable in Korean orthography 
   (e.g., "해 보다" / "해보다") must be excluded from error detection.
    ---

    **ANALYSIS WORKFLOW:**
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

def run_sheet_review(spreadsheet_name: str, worksheet_name: str) -> dict:
    """
    - 주어진 스프레드시트 / 워크시트에서
    - STATUS == '1. AI검수요청' 인 행만 골라서
    - SCORE / *_REPORT / STATUS를 채워넣는다.

    반환값: {"total_rows": ..., "target_rows": ..., "processed_rows": ...}
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
        }

    results = []

    for _, row in targets.iterrows():
        row_dict = row.to_dict()
        row_idx = row["sheet_row_index"]
        print(f"행 {row_idx} 검수 중...")

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
    }
