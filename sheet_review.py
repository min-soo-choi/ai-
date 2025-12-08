# sheet_review.py
# -*- coding: utf-8 -*-
import json
import time
import re
from typing import Dict, Any, List

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
raw = st.secrets["GCP_SERVICE_ACCOUNT_JSON"]

if isinstance(raw, dict):
    service_info = dict(raw)
elif isinstance(raw, str):
    service_info = json.loads(raw)
else:
    st.error("GCP_SERVICE_ACCOUNT_JSON 형식이 올바르지 않습니다.")
    st.stop()

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

creds = Credentials.from_service_account_info(service_info, scopes=SCOPES)
gs_client = gspread.authorize(creds)


# ---------------------------------------------------
# 2. 시트 컬럼 이름 (기존 스키마)
# ---------------------------------------------------

STATUS_COL = "STATUS"
ORIGINAL_TEXT_COL = "content"                   # 영어 원문
ORIGINAL_MD_COL = "content_markdown"
TRANSLATION_TEXT_COL = "content_translated"    # 한국어 번역
TRANSLATION_MD_COL = "content_markdown_translated"

SUSPICION_SCORE_COL = "SCORE"
CONTENT_TYPO_REPORT_COL = "CONTENT_TYPO_REPORT"      # 영어 검수 결과
TRANSLATED_COL = "TRANSLATED_TYPO_REPORT"            # 한국어 검수 결과
MARKDOWN_REPORT_COL = "MARKDOWN_REPORT"              # 마크다운 관련 오류


# ---------------------------------------------------
# 3. 공통 유틸: 리포트 후처리 / 문장부호 강제 / hallucination 필터
# ---------------------------------------------------

def dedup_korean_bullet_lines(report: str) -> str:
    """
    한국어 bullet 리포트에서 의미가 겹치는 줄을 정리한다.
    - 완전히 동일한 줄은 하나만 남김
    - '불필요한 마침표'류에서 원문이 부분 문자열 관계이면 더 긴 쪽만 유지
      (예: '했고.' vs '역할을 했고.' -> 후자만 남김)
    """
    if not report:
        return ""

    lines = [l.strip() for l in report.splitlines() if l.strip()]
    if not lines:
        return ""

    pattern = re.compile(r"^- '(.+?)' → '(.+?)':\s*(.+)$", re.UNICODE)

    # 1차: 완전 중복 제거
    unique_lines = []
    seen = set()
    for l in lines:
        if l not in seen:
            unique_lines.append(l)
            seen.add(l)

    # 2차: 불필요한 마침표 관련 중복 제거
    entries = []
    for idx, l in enumerate(unique_lines):
        m = pattern.match(l)
        if not m:
            entries.append({"idx": idx, "raw": l, "orig": None, "fixed": None, "msg": ""})
            continue
        orig, fixed, msg = m.group(1), m.group(2), m.group(3)
        entries.append({"idx": idx, "raw": l, "orig": orig, "fixed": fixed, "msg": msg})

    to_drop = set()
    for i, e1 in enumerate(entries):
        if not e1["orig"] or "불필요한 마침표" not in e1["msg"]:
            continue
        for j, e2 in enumerate(entries):
            if i == j or not e2["orig"] or "불필요한 마침표" not in e2["msg"]:
                continue

            o1, o2 = e1["orig"], e2["orig"]
            # 더 짧은 것이 더 긴 것의 부분 문자열이면 짧은 것 제거
            if o1 in o2 and len(o1) < len(o2):
                to_drop.add(e1["idx"])
            elif o2 in o1 and len(o2) < len(o1):
                to_drop.add(e2["idx"])

    final_lines = [
        l for idx, l in enumerate(unique_lines) if idx not in to_drop
    ]

    return "\n".join(final_lines)


# 종결부호 뒤 공백은 정상 → 보고서에서 제거
def drop_false_punctuation_space_errors(text: str, report: str) -> str:
    """
    '- ...: Spacing error' 류 중에서
    '문장부호(.?! 등) + 공백 + 새 문장 시작' 형태는 정상으로 보고 제거.
    """
    if not report:
        return report

    fixed = []
    for line in report.splitlines():
        if "Spacing error" in line or "공백 오류" in line:
            # 원문 추출
            m = re.search(r"'(.+?)' →", line)
            if m:
                original = m.group(1)
                # 종결부호 뒤 공백 여부 검사
                if re.search(r"[.!?]\s+[가-힣A-Za-z]", original):
                    # 이건 정상 구조 → 버린다
                    continue
        fixed.append(line)
    return "\n".join(fixed)


def drop_false_korean_period_errors(report: str) -> str:
    """
    한국어 리포트에서, '원문' 부분에 이미 종결부호가 있는데
    '마침표가 없습니다' 류로 잘못 보고한 줄을 제거한다.
    """
    if not report:
        return ""

    cleaned_lines = []
    pattern = re.compile(r"^- '(.+?)' → '(.+?)':", re.UNICODE)
    bad_phrases = [
        "마침표가 없습니다",
        "마침표가 빠져",
        "마침표가 필요",
        "마침표를 찍어야",
        "문장 끝에 마침표가 없",
    ]

    for line in report.splitlines():
        s = line.strip()
        if not s:
            continue

        # 마침표 관련 멘트가 아니면 그대로 통과
        if not any(p in s for p in bad_phrases):
            cleaned_lines.append(s)
            continue

        m = pattern.match(s)
        if not m:
            cleaned_lines.append(s)
            continue

        original = m.group(1).rstrip()
        if not original:
            cleaned_lines.append(s)
            continue

        last = original[-1]
        ok = False
        if last in ".?!":
            ok = True
        elif len(original) >= 2 and last in ['"', "'", "”", "’", "」", "』", "》", "〉", ")", "]"] and original[-2] in ".?!":
            ok = True

        # 이미 종결부호가 있으면 → 이 줄은 가짜 오류로 보고 제거
        if ok:
            continue
        else:
            cleaned_lines.append(s)

    return "\n".join(cleaned_lines)


def drop_lines_not_in_source(source_text: str, report: str) -> str:
    """
    report 안 '- '원문' → '수정안':' 패턴에서 '원문'이
    1) 실제 source_text에 완전 동일하게 존재하는 경우만 유지
    2) 띄어쓰기 normalize 후에도 존재하지 않으면 제거
    3) 부분 문자열만 일치할 경우도 제거
    """
    if not report:
        return ""

    cleaned = []
    pattern = re.compile(r"^- '(.+?)' → '(.+?)':", re.UNICODE)

    # normalize
    normalized_src = (
        source_text.replace(" ", "")
        .replace("\n", "")
        .replace("\u200b", "")
        .strip()
    )

    for line in report.splitlines():
        s = line.strip()
        if not s:
            continue

        m = pattern.match(s)
        if not m:
            cleaned.append(s)
            continue

        original = m.group(1)

        # 완전 동일 매칭만 허용
        if original in source_text:
            cleaned.append(s)
            continue

        # 띄어쓰기 제거 후 비교
        if original.replace(" ", "") in normalized_src:
            cleaned.append(s)
            continue

        # 그 외는 drop
        continue

    return "\n".join(cleaned)


def drop_escape_false_positives(report: str) -> str:
    """
    \"\\\"\", \"\\'\", '\"/\"' 등 escape/포맷팅 전용 토큰 때문에
    발생하는 잘못된 따옴표/문장부호 오류를 제거.
    """
    if not report:
        return report

    false_patterns = [
        r'\\\"',   # \"
        r'\\\'',   # \'
        r'\"/\"',
        r'\"',
        r'/\"',
        r'\"/',
    ]

    cleaned = []
    for line in report.splitlines():
        for p in false_patterns:
            if re.search(p, line):
                # escape 문자열로 인한 오판 → 제거
                break
        else:
            cleaned.append(line)

    return "\n".join(cleaned)


def ensure_final_punctuation_error(text: str, report: str) -> str:
    """
    문단 마지막 문장의 끝에 종결부호(. ? !)가 없으면
    report에 해당 오류를 강제로 한 줄 추가한다. (한국어 쪽에서 주로 사용)
    """
    if not text or not text.strip():
        return report or ""

    s = text.rstrip()
    if not s:
        return report or ""

    last = s[-1]

    end_ok = False
    if last in ".?!":
        end_ok = True
    elif last in ['"', "'", "”", "’", "」", "』", "》", "〉", ")", "]"] and len(s) >= 2 and s[-2] in ".?!":
        end_ok = True

    if end_ok:
        return report or ""

    # 이미 비슷한 내용이 있으면 중복으로 추가하지 않음
    if report and ("마침표" in report or "문장부호" in report):
        return report

    line = "- 문단 마지막 문장 끝에 마침표(또는 물음표, 느낌표)가 빠져 있으므로 적절한 문장부호를 추가해야 합니다."
    if report:
        return report.rstrip() + "\n" + line
    else:
        return line


def ensure_sentence_end_punctuation(text: str, report: str) -> str:
    """
    문장 단위로 잘라서 종결부호(. ? !) 없는 문장이 있으면
    한 줄로 요약해서 보고.
    """
    if not text or not text.strip():
        return report or ""

    # 기본 문장 분리
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    missing = []
    for i, s in enumerate(sentences):
        s = s.strip()
        if not s:
            continue

        ok = (
            s[-1] in ".?!"
            or (len(s) >= 2 and s[-1] in ['"', "'", "”", "’", "」", "』", "》", "〉", ")", "]"] and s[-2] in ".?!")
        )

        # 종결부호 없는 문장 수집
        if not ok:
            missing.append(s)

    if not missing:
        return report or ""

    line = "- 문장 끝에 종결부호(., ?, !)가 누락된 문장이 있습니다."

    if report:
        return report.rstrip() + "\n" + line
    else:
        return line


def clean_self_equal_corrections(report: str) -> str:
    """
    '- '원문' → '수정안': ...' 형식에서
    원문과 수정안이 완전히 같은 줄은 제거한다.
    (주로 영어 쪽 content_typo_report에 사용)
    """
    if not report:
        return ""

    cleaned_lines = []
    pattern = re.compile(r"^- '(.+?)' → '(.+?)':", re.UNICODE)

    for line in report.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue

        m = pattern.match(line_stripped)
        if not m:
            cleaned_lines.append(line_stripped)
            continue

        orig = m.group(1).strip()
        fixed = m.group(2).strip()

        if orig == fixed:
            # '원문' == '수정안'인 라인은 의미 없으므로 제거
            continue

        cleaned_lines.append(line_stripped)

    return "\n".join(cleaned_lines)


def drop_false_period_errors(english_text: str, report: str) -> str:
    """
    영어 원문 끝에 실제로 . ? ! 이 있으면
    리포트에서 '마침표 없음'류 문장을 제거.
    (거짓 양성 줄이기용)
    """
    if not report:
        return ""

    stripped = (english_text or "").rstrip()
    last_char = stripped[-1] if stripped else ""

    if last_char in [".", "?", "!"]:
        bad_phrases = [
            "마침표가 없습니다",
            "마침표가 빠져",
            "마침표가 필요",
            "마침표를 찍어야",
        ]
        cleaned_lines = []
        for line in report.splitlines():
            if any(p in line for p in bad_phrases):
                continue
            cleaned_lines.append(line.strip())
        return "\n".join(cleaned_lines)

    return report


def split_report_by_source(report: str, plain_text: str, md_text: str) -> tuple[str, str]:
    """
    하나의 리포트를 '원문이 plain에서 온 것' / '원문이 markdown에서 온 것'으로 나눈다.
    - "- '원문' → '수정안': ..." 패턴 기준으로 '원문'을 보고 소속을 결정
    - 원문이 plain에도 있고 md에도 있으면, 우선 plain 쪽으로 보낸다.
    """
    if not report:
        return "", ""

    plain_lines: List[str] = []
    md_lines: List[str] = []

    pattern = re.compile(r"^- '(.+?)' → '(.+?)':.*", re.UNICODE)

    for line in report.splitlines():
        s = line.strip()
        if not s:
            continue

        m = pattern.match(s)
        if not m:
            # 패턴이 아니면 일단 plain 쪽에 넣어둔다
            plain_lines.append(s)
            continue

        original = m.group(1)

        in_plain = original in (plain_text or "")
        in_md = original in (md_text or "")

        if in_plain and not in_md:
            plain_lines.append(s)
        elif in_md and not in_plain:
            md_lines.append(s)
        else:
            # 둘 다 포함되거나 둘 다 안 포함되면 우선 plain으로
            plain_lines.append(s)

    return "\n".join(plain_lines), "\n".join(md_lines)


# ---------------------------------------------------
# 3-A. 한국어 단어 내부 분리(형태소 분리) 후처리 전용 유틸
# ---------------------------------------------------

# ✅ 단어 내부 분리 오류 예외(정상 표현) 화이트리스트
INTERNAL_SPLIT_WHITELIST: set[str] = {
    "할 수",
    "수 있",
    "할 것",
    "것 이",
    "있 을",
    "할 뿐",
    "중요 한",
    "같 은 점",
    "다른 점",
    # 필요할 때 점점 추가해서 사용
}


def find_korean_internal_split_candidates(text: str) -> list[tuple[str, str]]:
    """
    '된 다', '묻 는' 같은 **단어 내부 분리**만 후보로 잡는다.
    - 앞/뒤 각각 1~2글자짜리 한글 + 공백 + 1~2글자짜리 한글
    - INTERNAL_SPLIT_WHITELIST에 있는 정상 표현은 제외
    """
    if not text:
        return []

    candidates: list[tuple[str, str]] = []

    # ([가-힣]{1,2}) + 공백 + ([가-힣]{1,2}) 패턴만 잡음
    pattern = re.compile(r"([가-힣]{1,2})\s+([가-힣]{1,2})")

    for m in pattern.finditer(text):
        left = m.group(1)
        right = m.group(2)
        span_text = f"{left} {right}"

        # 1) whitelist에 있으면 정상 띄어쓰기 → 건너뛰기
        if span_text in INTERNAL_SPLIT_WHITELIST:
            continue

        fixed = left + right
        candidates.append((span_text, fixed))

    return candidates


def build_internal_split_report(text: str) -> str:
    """
    find_korean_internal_split_candidates() 결과를
    리포트 형식의 문자열로 변환한다.
    """
    errors = find_korean_internal_split_candidates(text)
    if not errors:
        return ""

    lines: list[str] = []
    for orig, fixed in errors:
        lines.append(
            f"- '{orig}' → '{fixed}': '{orig}'는 단어 내부 공백이 잘못 들어간 형태소 분리 오류이며 "
            f"'{fixed}'로 붙여 써야 합니다."
        )
    return "\n".join(lines)


# ---------------------------------------------------
# 3. 프롬프트 정의 (영어 / 한국어 분리)
# ---------------------------------------------------

def create_english_review_prompt(text: str) -> str:
    """
    시트의 content(영어 원문)에 대해 검수하는 프롬프트.
    - 스펠링 / split-word / AI↔Al / 대문자 / 기본 문장 부호
    - 결과는 content_typo_report(한국어 설명)에만 쌓이게 유도
    """
    return f"""
You are a machine-like **English text proofreader**.
Your ONLY job is to detect **objective, verifiable errors** in the following English text.
You MUST NOT suggest stylistic changes, paraphrasing, natural-sounding alternatives, tone changes, or meaning changes.

Your response MUST be a single valid JSON object with keys:
- "suspicion_score": integer 1~5
- "content_typo_report": string (Korean 설명)
- "translated_typo_report": ""   ← 항상 빈 문자열
- "markdown_report": ""          ← 항상 빈 문자열

All explanations in content_typo_report MUST be written in **Korean**.

If there are no errors:
- suspicion_score = 1
- all reports = ""

------------------------------------------------------------
# IMPORTANT ANTI-HALLUCINATION RULE
------------------------------------------------------------
- In the pattern "- '원문' → '수정안': ...", the '원문' part MUST be a substring
  that actually appears in the input text `plain_english`.
- "원문" MUST always be copied from `plain_english` exactly as it appears.
- You MUST NOT invent new tokens or reuse example tokens that do not literally
  appear in the input text.

------------------------------------------------------------
# 1. RULES FOR ENGLISH OBJECTIVE ERRORS
------------------------------------------------------------

## (A) Split-Word Errors (항상 오타로 취급 — 매우 중요)
If an English word appears with an incorrect internal space,
AND removing the space yields a valid English word,
you MUST treat it as a spelling error.

ALWAYS flag patterns like:
- "wi th"  → "with"
- "dea th" → "death"
- "o f"    → "of"
- "amo ng" → "among"
- "cont inents" → "continents"

Report format (Korean):
"- 'wi th' → 'with': 'wi th'는 단어 내부 공백이 잘못된 오타이며 'with'로 수정해야 합니다."

## (B) Normal English spelling mistakes (MUST detect)
Any token similar to a valid English word (1–2 letters swapped/missing) MUST be flagged.

Examples (patterns, not exhaustive):
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

## (C) AI 문맥에서 "Al" → "AI" (항상 잡기)
If the surrounding sentence mentions:
model / system / tool / chatbot / LLM / agent / dataset / training / inference
then “Al” (A+소문자 l) MUST be interpreted as a typo for “AI”.

Examples:
- "Al model" → "AI model"
- "Al system" → "AI system"

## (D) Capitalization Errors
You MUST flag:
- Sentence starting with lowercase
- Pronoun “I” written as “i”
- Proper nouns not capitalized (london → London)

## (E) Duplicate / spacing errors
- "the the"
- "re turn" → "return"
- "mod el" → "model"

## (F) STRICT punctuation rule — avoid false positives
You MUST NOT report a punctuation error if the text already ends with ANY of:
- ".", "?", "!"
- ".*\"", ".*!\"", ".*?\""
- ".*’", ".*!’", ".*?’"

ONLY report a punctuation error if:
- the sentence has NO ending punctuation at all, OR
- a closing quotation mark is missing, OR
- punctuation is clearly malformed (e.g. ",.", ".,", "..", "!!", "??" in a wrong place)

------------------------------------------------------------
# 2. OUTPUT FORMAT
------------------------------------------------------------
You MUST output EXACTLY ONE JSON object (no extra text, no markdown).

Each error line example (in Korean):

"- 'understaning' → 'understanding': 'understaning'은 철자 오타이며 'understanding'으로 수정해야 합니다."

If there is NO objective error at all:
- "suspicion_score": 1
- "content_typo_report": ""
- "translated_typo_report": ""
- "markdown_report": ""

------------------------------------------------------------
# 3. TEXT TO REVIEW
plain_english: \"\"\"{text}\"\"\"
"""


def create_korean_review_prompt(text: str) -> str:
    """
    시트의 content_translated(한국어 번역)에 대해 검수하는 프롬프트.
    - 오탈자 / 조사·어미 / 띄어쓰기 / 형태소 분리 / 반복 / 문장부호
    - 결과는 translated_typo_report에만 쌓이게 유도
    """
    return f"""
당신은 기계적으로 동작하는 **Korean text proofreader**입니다.
당신의 유일한 임무는 아래 한국어 텍스트에서 **객관적이고 확인 가능한 오류만** 찾아내는 것입니다.

============================================================
🚨 가장 중요한 규칙 (원문 보존 — 절대 위반 금지)
============================================================
아래 질문 중 하나라도 “예”라면, 그 수정은 **보고하지 말고 완전히 무시**해야 합니다.

1) 수정하려는 부분이 plain_korean에 **그대로 존재하지 않는가?**
2) 단어 **순서를 변경**해야 하는가?
3) 의미가 달라질 수 있는 수정인가?
4) 새로운 단어를 **추가해야만** 수정이 가능한가?
5) 자연스럽게 들리도록 **다듬는 것**처럼 보이는가?
6) 문장을 사실상 **다시 쓰는 것처럼** 보이는가?

→ 하나라도 “예”라면, 해당 오류는 **절대 출력하지 않는다.**

============================================================
🚫 Hallucination 방지 규칙
============================================================
❌ 존재하지 않는 단어/구절 생성 금지  
❌ 프롬프트 설명부에 있는 단어를 ‘원문’으로 재사용 금지  
❌ 원문의 문장 구조·의도·톤·어순 변경 금지  

'- '원문' → '수정안': ...' 형식의 '원문'은  
반드시 plain_korean 안에 **문자 단위로 동일하게 존재**해야 합니다.

============================================================
📌 안전 예시 (더미 토큰 — 출력 금지)
============================================================
아래 예시는 허용되는 “수정 크기의 범위”만 설명하기 위한 것이며  
AAA/BBB/CCC 등은 실제 텍스트에 없는 **더미 토큰**입니다.  
출력에 등장하면 안 됩니다.

- 'AAA를를' → 'AAA를' : 조사 중복 수정(1~2글자)
- 'BBB 을' → 'BBB을' : 공백/조사 오용(극소수 변경)
- 'CCC 한 다' → 'CCC한다' : 단어 내부 공백(형태소 분리 오류)
- 'DDD다다다' → 'DDD다' : 반복 오타 정리

※ 위 예시는 단순 설명용이며 실제 출력에 포함되면 안 됩니다.

============================================================
# 1. 한국어에서 반드시 잡아야 하는 객관적 오류
============================================================

(A) 오탈자 / 철자 오류  
(B) 조사·어미 오류  
(C) 단어 내부 불필요한 공백  
(D) 반복 오타  
(E) 명백한 띄어쓰기 오류  
(F) 문장부호 오류  
   - 문장 끝에 종결부호 없음  
   - 따옴표 짝 불일치  
   - 명백히 잘못된 쉼표  
   - 문장 중간의 불필요한 마침표/쉼표  

[G] 문장부호 뒤 공백 규칙 (중요)
- 문장 끝에 마침표/물음표/느낌표가 있고, 그 뒤에서 새로운 문장이 시작될 경우,
  문장부호 뒤의 공백은 **정상이며 오타가 아니다.**
- 그러므로 "흘린다. 텔레비전"처럼 
  종결부호 + 공백 + 새로운 문장이 시작되는 구조는 절대로 오류로 판단하지 않는다.
- 단어 내부에서 불필요한 공백(예: '흘 린다', '한다 다')만 오류로 인정한다.

다음과 같은 패턴은 문법적 오류가 아니므로 절대로 오류로 보고하지 않는다.
- \"  ← JSON/markdown에서 쓰는 escape 문자
- \'  
- \"\"  
- /"  
- "/  
- */"  
- '"'
- markdown code block 기술에서 사용되는 `\"`, `\'`, `\(` 등

이들은 단순한 escape 또는 마크다운 포맷팅일 뿐이며,
따옴표 짝 불일치나 문장부호 오류로 간주해서는 안 된다.

특히 plain_korean 전체의 **마지막 문장 끝에 종결부호가 없으면 반드시 오류로 보고해야 합니다.**

============================================================
# 2. Output Format (JSON Only)
============================================================
다음 형식으로 bullet 단위 오류를 출력해야 합니다:

"- '원문' → '수정안': 오류 설명"

오류가 없다면:
- suspicion_score = 1
- content_typo_report = ""
- translated_typo_report = ""
- markdown_report = ""

============================================================
# 3. TEXT TO REVIEW
============================================================
plain_korean: "/"/"{text}"/"/"
"""


# ---------------------------------------------------
# 3-1. 공통: Gemini 호출 / 결과 정제
# ---------------------------------------------------

def analyze_text_with_gemini(prompt: str, max_retries: int = 5) -> dict:
    """Gemini를 JSON 모드로 호출 + 재시도 로직"""
    last_error: Exception | None = None

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
            last_error = e
            wait_time = 5 * (attempt + 1)
            print(f"Gemini 호출 오류 (시도 {attempt+1}/{max_retries}): {e}")

            if attempt < max_retries - 1:
                print(f"→ {wait_time}초 후 재시도")
                time.sleep(wait_time)

    print("최대 재시도 횟수 초과.")
    return {
        "suspicion_score": 5,
        "content_typo_report": f"API 호출 실패: {str(last_error)}",
        "translated_typo_report": "",
        "markdown_report": "",
    }


def validate_and_clean_analysis(result: dict) -> dict:
    """
    모델 응답의 기본 구조를 보정 + 스타일/문체성 멘트 필터링.
    """
    # 0) Gemini가 에러 JSON을 돌려준 경우 처리
    if isinstance(result, dict) and "ERROR" in result:
        err_obj = result.get("ERROR") or {}
        if isinstance(err_obj, dict):
            msg = err_obj.get("message") or str(err_obj)
        else:
            msg = str(err_obj)

        return {
            "suspicion_score": 5,
            "content_typo_report": f"Gemini API 내부 오류: {msg}",
            "translated_typo_report": "",
            "markdown_report": "",
        }

    # 1) 아예 dict가 아닐 때
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

    # 스타일/문체 제안 금지 키워드 필터
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

    # "오류 없음"류 멘트 제거
    forbidden_phrases = ["오류 없음", "정상", "문제 없음", "수정할 필요 없음"]
    for key, text in reports.items():
        if any(ph in text for ph in forbidden_phrases):
            reports[key] = ""

    # 영어 리포트에 대해서 self equal 정리 (시트 영어 쪽에서 사용)
    reports["content_typo_report"] = clean_self_equal_corrections(reports["content_typo_report"])

    # score 기본값 보정
    try:
        score = int(score)
    except Exception:
        score = 1

    if score < 1:
        score = 1
    if score > 5:
        score = 5

    if not reports["content_typo_report"] and not reports["translated_typo_report"] and not reports["markdown_report"]:
        score = 1
    elif (reports["content_typo_report"] or reports["translated_typo_report"] or reports["markdown_report"]) and score == 1:
        score = 3

    return {
        "suspicion_score": score,
        "content_typo_report": reports["content_typo_report"],
        "translated_typo_report": reports["translated_typo_report"],
        "markdown_report": reports["markdown_report"],
    }


# ---------------------------------------------------
# 3-2. 한 행(영어+한국어)을 통합 검수하는 헬퍼
# ---------------------------------------------------

def analyze_row_with_both_langs(row: Dict[str, Any]):
    """
    한 행(row)에 대해:
      - content / content_markdown (영어)
      - content_translated / content_markdown_translated (한국어)
    를 모두 합쳐서 한 번에 검수한다.
    """

    # 1) 원본 텍스트들 가져오기
    en_plain = (row.get(ORIGINAL_TEXT_COL) or "").strip()
    en_md = (row.get(ORIGINAL_MD_COL) or "").strip()
    ko_plain = (row.get(TRANSLATION_TEXT_COL) or "").strip()
    ko_md = (row.get(TRANSLATION_MD_COL) or "").strip()

    # 2) 실제로 모델에 보낼 통합 텍스트 (빈 건 제외하고 줄바꿈으로 이어 붙이기)
    en_text = "\n".join(t for t in [en_plain, en_md] if t)
    ko_text = "\n".join(t for t in [ko_plain, ko_md] if t)

    raw_en = final_en = None
    raw_ko = final_ko = None

    # --- 영어 쪽 ---
    if en_text:
        prompt_en = create_english_review_prompt(en_text)
        raw_en = analyze_text_with_gemini(prompt_en)
        final_en = validate_and_clean_analysis(raw_en)

        filtered_en = drop_lines_not_in_source(
            en_text,  # ✅ 통합 텍스트 기준으로 존재 여부 확인
            final_en.get("content_typo_report", "") or "",
        )
        filtered_en = drop_false_period_errors(en_text, filtered_en)
        filtered_en = ensure_sentence_end_punctuation(en_text, filtered_en)
        final_en["content_typo_report"] = filtered_en
    else:
        final_en = {
            "suspicion_score": 1,
            "content_typo_report": "",
            "translated_typo_report": "",
            "markdown_report": "",
        }

    # --- 한국어 쪽 ---
    if ko_text:
        prompt_ko = create_korean_review_prompt(ko_text)
        raw_ko = analyze_text_with_gemini(prompt_ko)
        final_ko = validate_and_clean_analysis(raw_ko)

        filtered_ko = drop_lines_not_in_source(
            ko_text,  # ✅ 통합 텍스트 기준
            final_ko.get("translated_typo_report", "") or "",
        )
        filtered_ko = drop_false_korean_period_errors(filtered_ko)
        filtered_ko = ensure_final_punctuation_error(ko_text, filtered_ko)
        filtered_ko = ensure_sentence_end_punctuation(ko_text, filtered_ko)
        filtered_ko = drop_escape_false_positives(filtered_ko)
        filtered_ko = dedup_korean_bullet_lines(filtered_ko)

        # 🔹 추가: 단어 내부 분리(형태소 분리) 전용 규칙 리포트 덧붙이기
        internal_report = build_internal_split_report(ko_text)
        if internal_report:
            if filtered_ko:
                filtered_ko = filtered_ko.rstrip() + "\n" + internal_report
            else:
                filtered_ko = internal_report

        filtered_ko = drop_false_punctuation_space_errors(ko_text, filtered_ko)
        final_ko["translated_typo_report"] = filtered_ko
    else:
        final_ko = {
            "suspicion_score": 1,
            "content_typo_report": "",
            "translated_typo_report": "",
            "markdown_report": "",
        }

    # --- plain / markdown 기준으로 리포트 분리 ---
    en_plain_report, en_md_report = split_report_by_source(
        final_en.get("content_typo_report", "") or "",
        en_plain,
        en_md,
    )
    ko_plain_report, ko_md_report = split_report_by_source(
        final_ko.get("translated_typo_report", "") or "",
        ko_plain,
        ko_md,
    )

    # plain 쪽은 기존 컬럼에 남기고
    final_en["content_typo_report"] = en_plain_report
    final_ko["translated_typo_report"] = ko_plain_report

    # markdown에서 나온 오류는 MARKDOWN_REPORT로 모으기
    markdown_report_parts: List[str] = []
    if en_md_report:
        markdown_report_parts.append(en_md_report)
    if ko_md_report:
        markdown_report_parts.append(ko_md_report)
    markdown_report = "\n".join(markdown_report_parts)

    # --- 통합 스코어 ---
    combined_final = {
        "suspicion_score": max(
            final_en.get("suspicion_score", 1),
            final_ko.get("suspicion_score", 1),
        ),
        "content_typo_report": final_en.get("content_typo_report", ""),
        "translated_typo_report": final_ko.get("translated_typo_report", ""),
        "markdown_report": markdown_report,
    }

    debug_bundle = {
        "english": {
            "text_plain": en_plain,
            "text_markdown": en_md,
            "text": en_text,  # 실제로 검수한 통합 텍스트
            "raw": raw_en,
            "final": final_en,
        },
        "korean": {
            "text_plain": ko_plain,
            "text_markdown": ko_md,
            "text": ko_text,
            "raw": raw_ko,
            "final": final_ko,
        },
    }

    return combined_final, debug_bundle


# ---------------------------------------------------
# 4. 공개 함수: 시트 전체를 돌리고 요약 리턴
# ---------------------------------------------------

def run_sheet_review(
    spreadsheet_name: str,
    worksheet_name: str,
    collect_raw: bool = False,
    progress_callback=None,
) -> dict:
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
            "english": {"text", "raw", "final"},
            "korean": {"text", "raw", "final"},
            "combined_final": {...},
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

    results: List[Dict[str, Any]] = []
    raw_results: List[Dict[str, Any]] = []

    total_targets = len(targets)

    for i, (_, row) in enumerate(targets.iterrows(), start=1):
        row_dict = row.to_dict()
        row_idx = row["sheet_row_index"]
        print(f"행 {row_idx} 검수 중... ({i}/{total_targets})")

        if progress_callback is not None:
            progress_callback(i, total_targets)

        # 🔹 영어 + 한국어 통합 검수
        combined_final, debug_bundle = analyze_row_with_both_langs(row_dict)

        results.append(
            {
                "sheet_row_index": row_idx,
                SUSPICION_SCORE_COL: combined_final.get("suspicion_score"),
                CONTENT_TYPO_REPORT_COL: combined_final.get("content_typo_report"),
                TRANSLATED_COL: combined_final.get("translated_typo_report"),
                MARKDOWN_REPORT_COL: combined_final.get("markdown_report"),
                STATUS_COL: "2. AI검수완료",
            }
        )

        if collect_raw:
            raw_results.append(
                {
                    "sheet_row_index": row_idx,
                    **debug_bundle,
                    "combined_final": combined_final,
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
        "raw_results": raw_results,
    }
