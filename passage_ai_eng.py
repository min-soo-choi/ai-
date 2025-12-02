# -*- coding: utf-8 -*-
"""
Vertex AI 인증(서비스계정) → Gemini API Key 방식으로 전환한 버전
- 동일 모델: gemini-2.0-flash-001
- 동일 동작: temperature=0, JSON 응답 강제(response_mime_type)
- Google Sheets 연동은 그대로 유지

실행 전 준비:
1) pip install google-generativeai gspread google-auth pandas
2) 환경변수로 API 키 설정 (예: mac/linux)
   export GEMINI_API_KEY="YOUR_API_KEY"
   (Windows PowerShell)
   setx GEMINI_API_KEY "YOUR_API_KEY"
"""

import os
import json
import time
import pandas as pd
import gspread
import google.generativeai as genai
from google.oauth2.service_account import Credentials
from config import get_gemini_api_key

api_key = get_gemini_api_key()

# --- 1. 설정 (사용자 환경에 맞게 유지) ---

# Google Sheets 정보
SPREADSHEET_NAME = '[DATA] Paragraph DB (참고서)'
WORKSHEET_NAME = '최종데이터'

# 검수 및 결과 컬럼 이름
STATUS_COL = 'STATUS'
ORIGINAL_TEXT_COL = 'content'
ORIGINAL_MD_COL = 'content_markdown'
TRANSLATION_TEXT_COL = 'content_translated'
TRANSLATION_MD_COL = 'content_markdown_translated'

SUSPICION_SCORE_COL = 'SCORE'
CONTENT_TYPO_REPORT_COL = 'CONTENT_TYPO_REPORT'
TRANSLATED_COL = 'TRANSLATED_TYPO_REPORT'
MARKDOWN_REPORT_COL = 'MARKDOWN_REPORT'

# 모델 설정 (동일)
MODEL_ID = 'gemini-2.0-flash-001'

# 서비스 계정 키 (Google Sheets 용)
script_dir = os.path.dirname(os.path.abspath(__file__))
SERVICE_ACCOUNT_FILE_NAME = 'expertupdate-f1983b6ca93e.json'  # 필요 시 파일명 교체
SERVICE_ACCOUNT_FILE = os.path.join(script_dir, SERVICE_ACCOUNT_FILE_NAME)

# --- 2. 인증 및 초기화 ---
def setup_services():
    """Google Sheets 인증 및 Gemini API 초기화(API Key)"""
    try:
        # Google Sheets 인증
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scopes)
        gs_client = gspread.authorize(creds)

        # Gemini API Key 구성
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise RuntimeError('환경변수 GEMINI_API_KEY가 설정되어 있지 않습니다.')
        genai.configure(api_key=api_key)

        print('✅ Google Sheets 인증 & Gemini API Key 구성 완료')
        return gs_client
    except Exception as e:
        print(f"❗️ 인증 실패: {e}")
        return None

# --- 3. 프롬프트 생성 (원본과 동일 규칙) ---
def create_review_prompt(row):
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
    3.  **RESPECT STYLING CONVENTIONS:** It is CORRECT for *italicized English* to be represented with '\'single quotes\'' or `《double angle brackets》` in Korean. This is NOT an error.
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
    - `plain_english`: "{original_text}"
    - `markdown_english`: "{original_md}"
    - `plain_korean`: "{translation_text}"
    - `markdown_korean`: "{translation_md}"
    """
    return prompt

# --- 4. Gemini API 호출 (API Key) ---
def analyze_text_with_gemini_api(prompt: str, max_retries: int = 5):
    """temperature=0, JSON 응답 강제, 재시도 로직 포함(Gemini API Key)"""
    generation_config = {
        "response_mime_type": "application/json",
        "temperature": 0.0,
    }
    model = genai.GenerativeModel(model_name=MODEL_ID)

    last_error = None
    for attempt in range(max_retries):
        try:
            resp = model.generate_content(prompt, generation_config=generation_config)
            # 응답 텍스트 추출 (SDK 버전에 따라 .text 또는 candidates 경로)
            text = getattr(resp, 'text', None)
            if not text:
                # fallback: candidates → content → parts → text
                try:
                    text = resp.candidates[0].content.parts[0].text
                except Exception:
                    text = None
            if not text:
                raise ValueError('빈 응답 수신')
            return json.loads(text)
        except Exception as e:
            last_error = e
            print(f"❗️ Gemini API 호출 오류 (시도 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 10 * (attempt + 1)
                print(f"⏳ 잠시 후 재시도합니다... ({wait_time}초)")
                time.sleep(wait_time)
            else:
                print("❗️ 최대 재시도 횟수를 초과했습니다.")
                return {
                    "suspicion_score": 5,
                    "content_typo_report": f"API 호출에 최종 실패했습니다: {str(last_error)}",
                    "translated_typo_report": "",
                    "markdown_report": ""
                }

# --- 5. 결과 검증 (주관적 표현 필터링) ---
def validate_and_clean_analysis(result):
    if not isinstance(result, dict):
        return { "suspicion_score": 5, "content_typo_report": "AI 응답이 유효한 JSON 형식이 아님", "translated_typo_report": "", "markdown_report": "" }

    score = result.get('suspicion_score')
    reports = {
        "content_typo_report": result.get('content_typo_report', ''),
        "translated_typo_report": result.get('translated_typo_report', ''),
        "markdown_report": result.get('markdown_report', '')
    }

    forbidden_keywords = [
        "문맥상", "부적절", "어색", "더 자연스럽", "더 적절", "수정하는 것이 좋", "제안", "바꾸는 것", "의미를 명확히"
    ]

    for key, report_text in reports.items():
        if any(keyword in report_text for keyword in forbidden_keywords):
            reports[key] = ""

    forbidden_phrases = ["오류 없음", "정상", "문제 없음", "수정할 필요 없음"]
    for key, report_text in reports.items():
        if any(phrase in report_text for phrase in forbidden_phrases):
            reports[key] = ""

    final_content_report = reports["content_typo_report"]
    final_translated_report = reports["translated_typo_report"]
    final_markdown_report = reports["markdown_report"]

    if not final_content_report and not final_translated_report and not final_markdown_report:
        score = 1
    elif (final_content_report or final_translated_report or final_markdown_report) and score == 1:
        score = 3

    return {
        "suspicion_score": score,
        "content_typo_report": final_content_report,
        "translated_typo_report": final_translated_report,
        "markdown_report": final_markdown_report
    }

# --- 6. 메인 실행 로직 ---
def main():
    print("🚀 지문 검수 프로세스를 시작합니다... (Gemini API Key 모드)")

    gs_client = setup_services()
    if not gs_client:
        return

    try:
        spreadsheet = gs_client.open(SPREADSHEET_NAME)
        worksheet = spreadsheet.worksheet(WORKSHEET_NAME)
        all_data = worksheet.get_all_records()
        df = pd.DataFrame(all_data)
        df['sheet_row_index'] = df.index + 2

        review_targets_df = df[df[STATUS_COL] == '1. AI검수요청'].copy()

        if review_targets_df.empty:
            print("✅ 검수 요청된 항목이 없습니다.")
            return

        print(f"🔍 총 {len(review_targets_df)}개의 항목에 대한 검수를 시작합니다.")

        results = []
        for index, row in review_targets_df.iterrows():
            print(f"🔄 {row['sheet_row_index']}번 행 검수 중...")
            prompt = create_review_prompt(row)
            raw_analysis_result = analyze_text_with_gemini_api(prompt)
            final_analysis_result = validate_and_clean_analysis(raw_analysis_result)

            score = final_analysis_result.get('suspicion_score')
            content_report = final_analysis_result.get('content_typo_report')
            translated_report = final_analysis_result.get('translated_typo_report')
            markdown_report = final_analysis_result.get('markdown_report')

            results.append({
                'sheet_row_index': row['sheet_row_index'],
                SUSPICION_SCORE_COL: score,
                CONTENT_TYPO_REPORT_COL: content_report,
                TRANSLATED_COL: translated_report,
                MARKDOWN_REPORT_COL: markdown_report,
                STATUS_COL: '2.AI검수완료'
            })
            time.sleep(1)

        print("\n✅ 모든 항목의 분석이 완료되었습니다. 결과를 스프레드시트에 업데이트합니다.")

        update_cells = []
        headers = worksheet.row_values(1)

        score_col_idx = headers.index(SUSPICION_SCORE_COL) + 1
        content_col_idx = headers.index(CONTENT_TYPO_REPORT_COL) + 1
        translated_col_idx = headers.index(TRANSLATED_COL) + 1
        markdown_col_idx = headers.index(MARKDOWN_REPORT_COL) + 1
        status_col_idx = headers.index(STATUS_COL) + 1

        for result in results:
            row_idx = result['sheet_row_index']

            def sanitize(value):
                return str(value) if value is not None else ""

            update_cells.append(gspread.Cell(row_idx, score_col_idx, sanitize(result[SUSPICION_SCORE_COL])))
            update_cells.append(gspread.Cell(row_idx, content_col_idx, sanitize(result[CONTENT_TYPO_REPORT_COL])))
            update_cells.append(gspread.Cell(row_idx, translated_col_idx, sanitize(result[TRANSLATED_COL])))
            update_cells.append(gspread.Cell(row_idx, markdown_col_idx, sanitize(result[MARKDOWN_REPORT_COL])))
            update_cells.append(gspread.Cell(row_idx, status_col_idx, sanitize(result[STATUS_COL])))

        if update_cells:
            worksheet.update_cells(update_cells)

        print("🎉 작업이 성공적으로 완료되었습니다!")

    except gspread.exceptions.SpreadsheetNotFound:
        print(f"❗️ 스프레드시트를 찾을 수 없습니다. 이름: '{SPREADSHEET_NAME}'")
    except gspread.exceptions.WorksheetNotFound:
        print(f"❗️ 워크시트를 찾을 수 없습니다. 이름: '{WORKSHEET_NAME}'")
    except Exception as e:
        print(f"❗️ 예상치 못한 오류 발생: {e}")


if __name__ == '__main__':
    main()
