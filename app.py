# app.py
# -*- coding: utf-8 -*-
import json
import time
import re
from typing import Dict, Any, List

import streamlit as st
import google.generativeai as genai

from sheet_review import run_sheet_review

# --------------------------
# 0. Gemini 설정 (키는 secrets에서만 읽기)
# --------------------------
API_KEY = st.secrets.get("GEMINI_API_KEY")
if not API_KEY:
    st.error("GEMINI_API_KEY가 secrets에 설정되어 있지 않습니다.")
    st.stop()

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-001")


# -------------------------------------------------
# 공통 유틸
# -------------------------------------------------
def analyze_text_with_gemini(prompt: str, max_retries: int = 5) -> dict:
    """
    단일 텍스트 검사용 Gemini 호출.
    항상 dict를 리턴하도록 방어 로직을 넣음.
    """
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

            raw = getattr(response, "text", None)
            if raw is None or not str(raw).strip():
                return {
                    "suspicion_score": 5,
                    "content_typo_report": "AI 응답이 비어 있습니다.",
                    "translated_typo_report": "",
                    "markdown_report": "",
                }

            obj = json.loads(raw)

            if not isinstance(obj, dict):
                return {
                    "suspicion_score": 5,
                    "content_typo_report": f"AI 응답이 dict가 아님 (type={type(obj).__name__})",
                    "translated_typo_report": "",
                    "markdown_report": "",
                }

            return obj

        except Exception as e:
            last_error = e
            wait_time = 5 * (attempt + 1)
            print(f"[Gemini(single)] 호출 오류 (시도 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"→ {wait_time}초 후 재시도")
                time.sleep(wait_time)

    print("[Gemini(single)] 최대 재시도 횟수 초과.")
    return {
        "suspicion_score": 5,
        "content_typo_report": f"API 호출 실패: {last_error}",
        "translated_typo_report": "",
        "markdown_report": "",
    }


def drop_lines_not_in_source(source_text: str, report: str) -> str:
    """
    '- '원문' → '수정안': ...' 형식에서
    '원문'이 실제 source_text에 포함되지 않은 라인을 제거.
    (한국어/영어 공통 사용)
    """
    if not report:
        return ""

    cleaned: List[str] = []
    pattern = re.compile(r"^- '(.+?)' → '(.+?)':", re.UNICODE)

    for line in report.splitlines():
        s = line.strip()
        if not s:
            continue

        m = pattern.match(s)
        if not m:
            cleaned.append(s)
            continue

        original = m.group(1)
        if original in source_text:
            cleaned.append(s)
        else:
            continue

    return "\n".join(cleaned)


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

        if ok:
            # 이미 종결부호가 있는 문장인데 '마침표 없음'이라고 한 줄 → 버림
            continue
        else:
            cleaned_lines.append(s)

    return "\n".join(cleaned_lines)


def ensure_final_punctuation_error(text: str, report: str) -> str:
    """
    문단 마지막 문장의 끝에 종결부호(. ? !)가 없으면
    report에 오류를 강제로 한 줄 추가한다.
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

    # 이미 비슷한 멘트가 있으면 중복 추가 안 함
    if report and ("마지막 문장에 마침표" in report or "문장 끝에 마침표가 없" in report):
        return report

    line = "- '수 있었다' → '수 있었다.': 마지막 문장에 마침표가 없습니다."
    # 위 예시는 실제 원문 기준으로 나가지는 않지만, 요약형 한 줄로 사용
    if report:
        return report.rstrip() + "\n" + line
    else:
        return line


def ensure_sentence_end_punctuation(text: str, report: str) -> str:
    """
    문단 내 모든 문장의 끝에 종결부호(. ? !)가 있는지 대략 검사.
    누락된 문장이 하나라도 있으면 요약 메시지를 추가.
    다만 이미 다른 줄에서 종결부호 누락을 구체적으로 언급했다면
    중복 메시지는 추가하지 않는다.
    """
    if not text or not text.strip():
        return report or ""

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    missing = []

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        ok = False
        if s[-1] in ".?!":
            ok = True
        elif len(s) >= 2 and s[-1] in ['"', "'", "”", "’", "」", "』", "》", "〉", ")", "]"] and s[-2] in ".?!":
            ok = True

        if not ok:
            missing.append(s)

    if not missing:
        return report or ""

    # 이미 종결부호 관련 멘트가 있으면 요약 줄 생략
    if report and any(
        key in report
        for key in ["마지막 문장에 마침표", "종결부호", "문장 끝에 마침표가 없", "마침표가 없습니다"]
    ):
        return report

    line = "- 문장 끝에 종결부호(., ?, !)가 누락된 문장이 있습니다."

    if report:
        return report.rstrip() + "\n" + line
    else:
        return line


def dedup_korean_bullet_lines(report: str) -> str:
    """
    한국어 bullet 리포트에서 의미가 겹치는 줄을 정리한다.
    - 완전히 동일한 줄은 하나만 남김
    - '불필요한 마침표'류에서 원문이 부분 문자열 관계이면 더 긴 쪽만 유지
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

    entries = []
    for idx, l in enumerate(unique_lines):
        m = pattern.match(l)
        if not m:
            entries.append({"idx": idx, "raw": l, "orig": None, "msg": ""})
            continue
        orig, fixed, msg = m.group(1), m.group(2), m.group(3)
        entries.append({"idx": idx, "raw": l, "orig": orig, "msg": msg})

    to_drop = set()
    for i, e1 in enumerate(entries):
        if not e1["orig"] or "불필요한 마침표" not in e1["msg"]:
            continue
        for j, e2 in enumerate(entries):
            if i == j or not e2["orig"] or "불필요한 마침표" not in e2["msg"]:
                continue
            o1, o2 = e1["orig"], e2["orig"]
            if o1 in o2 and len(o1) < len(o2):
                to_drop.add(e1["idx"])
            elif o2 in o1 and len(o2) < len(o1):
                to_drop.add(e2["idx"])

    final_lines = [
        l for idx, l in enumerate(unique_lines) if idx not in to_drop
    ]

    return "\n".join(final_lines)


def validate_and_clean_analysis(result: dict, original_english_text: str | None = None) -> dict:
    """
    AI 응답에서 문체 제안 등을 필터링하고 점수를 보정 + (영어 쪽 추가 후처리)
    """
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

    # 영어 리포트 후처리
    english_report = reports["content_typo_report"]
    english_report = clean_self_equal_corrections(english_report)
    if original_english_text:
        english_report = drop_false_period_errors(original_english_text, english_report)
    reports["content_typo_report"] = english_report

    final_content = reports["content_typo_report"]
    final_translated = reports["translated_typo_report"]
    final_markdown = reports["markdown_report"]

    # score 기본값 보정
    try:
        score = int(score)
    except Exception:
        score = 1

    if score < 1:
        score = 1
    if score > 5:
        score = 5

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


# -------------------------------------------------
# 1-A. 한국어 단일 텍스트 검수 프롬프트 + 래퍼
# -------------------------------------------------
def create_korean_review_prompt_for_text(korean_text: str) -> str:
    prompt = f"""
당신은 기계적으로 동작하는 **Korean text proofreader**입니다.
당신의 유일한 임무는 아래 한국어 텍스트에서 **객관적이고 검증 가능한 오류만** 찾아내는 것입니다.
스타일, 어투, 자연스러움, 표현 개선, 의도 추론과 같은 주관적 판단은 절대 해서는 안 됩니다.

출력은 반드시 아래 4개의 key만 포함하는 **단일 JSON 객체**여야 합니다.
- "suspicion_score": 1~5 정수
- "content_typo_report": "" (비워두기 — 영어용 필드)
- "translated_typo_report": 한국어 오류 설명 (없으면 "")
- "markdown_report": "" (항상 빈 문자열)

모든 설명은 반드시 **한국어로** 작성해야 합니다.
오류가 하나도 없으면 모든 report 필드는 "" 여야 합니다.

------------------------------------------------------------
# 🚨 절대 금지 규칙 (Hallucination 방지 — 매우 중요)
------------------------------------------------------------
❌ 입력 텍스트에 존재하지 않는 단어·구절을 생성  
❌ 의도·감정·내용을 추론하여 새로운 문장을 제안  
❌ 문장을 바꾸거나 다른 말로 바꿔 표현  
❌ 입력되지 않은 단어를 수정 대상으로 지목  
❌ 내용 왜곡 또는 의미적 비평

오직 “입력 문자열 안에 실제로 존재하는 토큰”만 인용하고 수정해야 합니다.

또한, "- '원문' → '수정안': ..." 형식에서 '원문' 부분은
반드시 plain_korean 안에 실제로 존재하는 부분 문자열이어야 합니다.

------------------------------------------------------------
# 1. 한국어에서 반드시 잡아야 하는 객관적 오류
------------------------------------------------------------

(중략 – sheet 프롬프트와 동일 규칙)

------------------------------------------------------------
# 3. 검사할 텍스트
------------------------------------------------------------

plain_korean: "{korean_text}"

이제 위 규칙을 지키며 위의 한국어 텍스트를 검수하세요.
"""
    return prompt


def review_korean_text(korean_text: str) -> Dict[str, Any]:
    """한국어 텍스트 검수 래퍼"""
    prompt = create_korean_review_prompt_for_text(korean_text)
    raw = analyze_text_with_gemini(prompt)
    cleaned = validate_and_clean_analysis(raw)

    filtered = drop_lines_not_in_source(
        korean_text,
        cleaned.get("translated_typo_report", "") or "",
    )
    filtered = drop_false_korean_period_errors(filtered)
    filtered = ensure_final_punctuation_error(korean_text, filtered)
    filtered = ensure_sentence_end_punctuation(korean_text, filtered)
    filtered = dedup_korean_bullet_lines(filtered)

    return {
        "score": cleaned.get("suspicion_score"),
        "content_typo_report": cleaned.get("content_typo_report", ""),
        "translated_typo_report": filtered,
        "markdown_report": cleaned.get("markdown_report", ""),
        "raw": raw,
    }


# -------------------------------------------------
# 1-B. 영어 단일 텍스트 검수 프롬프트 + 래퍼
# -------------------------------------------------
def create_english_review_prompt_for_text(english_text: str) -> str:
    prompt = f"""
You are a machine-like **English text proofreader**.
Your ONLY job is to detect **objective, verifiable errors** in the following English text.
You are strictly forbidden from judging tone, style, naturalness, or suggesting alternative phrasing.

Your response MUST be a valid JSON object with exactly these keys:
- "suspicion_score": integer (1~5)
- "content_typo_report": string
- "translated_typo_report": string
- "markdown_report": string

All explanations in the *_report fields MUST be written in **Korean**.
If nothing is wrong, each report field MUST be an empty string "".

(중략 – 시트 영어 프롬프트와 동일 규칙)

plain_english: "{english_text}"
"""
    return prompt


def review_english_text(english_text: str) -> Dict[str, Any]:
    """영어 텍스트 검수 래퍼"""
    prompt = create_english_review_prompt_for_text(english_text)
    raw = analyze_text_with_gemini(prompt)
    cleaned = validate_and_clean_analysis(raw, original_english_text=english_text)
    return {
        "score": cleaned.get("suspicion_score"),
        "content_typo_report": cleaned.get("content_typo_report", ""),
        "raw": raw,
    }


# -------------------------------------------------
# 공통: JSON diff / 제안 추출
# -------------------------------------------------
def summarize_json_diff(raw: dict | None, final: dict | None) -> str:
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
    if not isinstance(raw, dict):
        return []
    collected = []
    fields = [
        raw.get("translated_typo_report", ""),
        raw.get("content_typo_report", ""),
        raw.get("markdown_report", ""),
    ]
    for block in fields:
        if not block:
            continue
        for line in block.split("\n"):
            line = line.strip()
            if not line:
                continue
            if not line.startswith("- "):
                line = f"- {line}"
            collected.append(line)
    return collected


def extract_english_suggestions_from_raw(raw: dict) -> list[str]:
    if not isinstance(raw, dict):
        return []
    collected: list[str] = []
    fields = [
        raw.get("content_typo_report", ""),
        raw.get("translated_typo_report", ""),
        raw.get("markdown_report", ""),
    ]
    for block in fields:
        if not block:
            continue
        for line in block.split("\n"):
            line = line.strip()
            if not line:
                continue
            if not line.startswith("- "):
                line = f"- {line}"
            collected.append(line)
    return collected


# -------------------------------------------------
# 2. Streamlit UI
# -------------------------------------------------
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
            st.session_state["ko_result"] = result

    if "ko_result" in st.session_state:
        result = st.session_state["ko_result"]
        score = result.get("score", 1)
        raw_json = result.get("raw", {}) or {}

        final_json = {
            "suspicion_score": result.get("score", 1),
            "translated_typo_report": result.get("translated_typo_report", ""),
        }

        raw_view = {
            "suspicion_score": raw_json.get("suspicion_score"),
            "translated_typo_report": raw_json.get("translated_typo_report", ""),
        }

        st.success("한국어 검수가 완료되었습니다!")
        st.metric("의심 점수 (1~5)", f"{float(score):.2f}")

        st.markdown("### 🔍 결과 비교 (Raw vs Final)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ✅ Final JSON (후처리 적용)")
            st.json(final_json, expanded=False)
        with col2:
            st.markdown("#### 🧪 Raw JSON (동일 필드만 발췌)")
            st.json(raw_view, expanded=False)

        st.markdown("#### 🔍 Raw vs Final 차이 요약")
        diff_md = summarize_json_diff(raw_view, final_json)
        st.markdown(diff_md)

        st.markdown("### 🛠 최종 수정 제안 사항")
        suggestions = extract_korean_suggestions_from_raw({"translated_typo_report": final_json["translated_typo_report"]})
        if not suggestions:
            st.info("보고할 수정 사항이 없습니다.")
        else:
            for s in suggestions:
                st.markdown(s)


# --- 영어 검수 탭 ---
with tab_en:
    st.subheader("영어 텍스트 검수")
    default_en = 'This is a simple understaning of the Al model.'
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

        final_json = {
            "suspicion_score": result.get("score", 1),
            "content_typo_report": result.get("content_typo_report", ""),
        }

        raw_view = {
            "suspicion_score": raw_json.get("suspicion_score"),
            "content_typo_report": raw_json.get("content_typo_report", ""),
        }

        st.success("영어 검수가 완료되었습니다!")
        st.metric("Suspicion score (1~5)", f"{float(score):.2f}")

        st.markdown("### 🔍 결과 비교 (Raw vs Final)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ✅ Final JSON (후처리 적용)")
            st.json(final_json, expanded=False)
        with col2:
            st.markdown("#### 🧪 Raw JSON (동일 필드만 발췌)")
            st.json(raw_view, expanded=False)

        st.markdown("#### 🔍 Raw vs Final 차이 요약")
        diff_md = summarize_json_diff(raw_view, final_json)
        st.markdown(diff_md)

        st.markdown("### 🛠 최종 수정 제안 사항 (영어 원문 기준)")
        suggestions = extract_english_suggestions_from_raw(raw_json)
        if not suggestions:
            st.info("보고할 수정 사항이 없습니다.")
        else:
            for s in suggestions:
                st.markdown(s)


# --- 시트 검수 탭 ---
with tab_sheet:
    st.subheader("📄 Google Sheets 시트 검수")

    # 🔽 하드코딩된 드롭다운 목록
    sheet_options = [
        "[DATA] Paragraph DB (교과서)",
        "[DATA] Paragraph DB (참고서)",
        "[DATA] Paragraph DB (모의고사)",
    ]

    worksheet_options = [
        "최종데이터",
        "22개정",
    ]

    # 🔽 스프레드시트 선택 드롭다운
    spreadsheet_name = st.selectbox(
        "스프레드시트 선택",
        options=sheet_options,
    )

    # 🔽 워크시트 선택 드롭다운
    worksheet_name = st.selectbox(
        "워크시트 선택",
        options=worksheet_options,
    )

    col_btn, col_blank = st.columns([1, 4])
    with col_btn:
        run_clicked = st.button("이 시트 검수 실행", type="primary")

    if run_clicked:
        if not spreadsheet_name or not worksheet_name:
            st.warning("스프레드시트와 워크시트를 모두 선택해주세요.")
        else:
            progress_bar = st.progress(0.0)
            progress_text = st.empty()

            with st.spinner("시트 검수 중입니다... (행이 많으면 시간이 걸려요)"):
                try:
                    summary = run_sheet_review(
                        spreadsheet_name,
                        worksheet_name,
                        collect_raw=True,
                        progress_callback=lambda done, total: (
                            progress_bar.progress(done / total),
                            progress_text.text(f"진행도: {done}/{total} 완료")
                        ),
                    )
                except Exception as e:
                    st.error(f"실행 중 오류가 발생했습니다: {e}")
                else:
                    progress_bar.progress(1.0)
                    st.success("검수 완료!")
                    st.session_state["sheet_summary"] = summary
                    st.session_state["raw_results"] = summary.get("raw_results", [])
                    st.rerun()


    summary = st.session_state.get("sheet_summary")
    raw_results = st.session_state.get("raw_results", [])

    if summary:
        st.divider()
        total_rows = summary.get("total_rows", 0)
        target_rows = summary.get("target_rows", 0)
        processed_rows = summary.get("processed_rows", 0)
        remaining_rows = max(target_rows - processed_rows, 0)

        st.success("✅ 시트 검수 작업 완료 (결과가 저장되었습니다)")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("전체 행 수", total_rows)
        with col_m2:
            st.metric("검수 대상 행 수", target_rows)
        with col_m3:
            st.metric("실제 처리된 행 수", processed_rows)
        with col_m4:
            st.metric("남은 대상 행", remaining_rows)

        st.info("👉 Google Sheets에서 SCORE / *_REPORT / STATUS 컬럼을 확인해주세요.")

        st.markdown("### 🐞 디버그: 특정 행의 Raw / Final JSON & Diff")

        if not raw_results:
            st.info("수집된 Raw 데이터가 없습니다. (검수 대상 행이 없었거나 오류 발생)")
        else:
            row_numbers = [item["sheet_row_index"] for item in raw_results]

            selected_candidate = st.selectbox(
                "Raw/Final JSON을 보고 싶은 행 번호를 선택하세요:",
                options=row_numbers,
                format_func=lambda x: f"행 {x}번",
            )

            if st.button("선택한 행 분석 보기"):
                st.session_state["selected_row"] = selected_candidate

            selected_row = st.session_state.get("selected_row")

            if selected_row is not None:
                selected_item = next(
                    (item for item in raw_results if item["sheet_row_index"] == selected_row),
                    None,
                )

                if selected_item:
                    st.markdown(f"#### 🔎 행 {selected_row}번 분석 결과")

                    view_mode = st.radio(
                        "어느 쪽 결과를 볼까요?",
                        [
                            "통합 결과 (시트 기록값)",
                            "영어 원문 전용 (content)",
                            "한국어 번역 전용 (content_translated)",
                            "마크다운 관련 오류 (content_markdown + content_markdown_translated)",
                        ],
                        horizontal=True,
                    )

                    # 1) 통합 결과: 시트에 실제로 적힌 combined_final 그대로
                    if view_mode.startswith("통합"):
                        st.markdown("##### 🧾 시트에 기록된 통합 결과 (combined_final)")
                        st.json(selected_item.get("combined_final", {}))

                    # 2) 영어 원문 전용 디버그
                    elif view_mode.startswith("영어"):
                        bundle = selected_item.get("english", {}) or {}
                        raw_json = bundle.get("raw") or {}
                        final_json = bundle.get("final") or {}

                        st.markdown("##### 📄 영어 원문 텍스트 (plain)")
                        st.code(bundle.get("text_plain", "") or "", language="markdown")

                        st.markdown("##### 📝 영어 마크다운 텍스트 (content_markdown)")
                        st.code(bundle.get("text_markdown", "") or "", language="markdown")

                        st.markdown("##### ⚡ Raw vs Final 차이점 (필터링 확인)")
                        diff_md = summarize_json_diff(raw_json, final_json)
                        st.markdown(diff_md)

                        st.divider()
                        col_final, col_raw = st.columns(2)
                        with col_raw:
                            st.markdown("##### 🤖 Raw JSON (AI 원본)")
                            st.json(raw_json)
                        with col_final:
                            st.markdown("##### 🧹 Final JSON (후처리 적용)")
                            st.json(final_json)

                    # 3) 한국어 번역 전용 디버그
                    elif view_mode.startswith("한국어"):
                        bundle = selected_item.get("korean", {}) or {}
                        raw_json = bundle.get("raw") or {}
                        final_json = bundle.get("final") or {}

                        st.markdown("##### 📄 한국어 번역 텍스트 (plain)")
                        st.code(bundle.get("text_plain", "") or "", language="markdown")

                        st.markdown("##### 📝 한국어 마크다운 텍스트 (content_markdown_translated)")
                        st.code(bundle.get("text_markdown", "") or "", language="markdown")

                        st.markdown("##### ⚡ Raw vs Final 차이점 (필터링 확인)")
                        diff_md = summarize_json_diff(raw_json, final_json)
                        st.markdown(diff_md)

                        st.divider()
                        col_final, col_raw = st.columns(2)
                        with col_raw:
                            st.markdown("##### 🤖 Raw JSON (AI 원본)")
                            st.json(raw_json)
                        with col_final:
                            st.markdown("##### 🧹 Final JSON (후처리 적용)")
                            st.json(final_json)

                    # 4) 마크다운 관련 오류만 모아서 보기
                    else:
                        combined_final = selected_item.get("combined_final", {}) or {}
                        markdown_report = combined_final.get("markdown_report", "") or ""

                        en_md = (selected_item.get("english", {}) or {}).get("text_markdown", "") or ""
                        ko_md = (selected_item.get("korean", {}) or {}).get("text_markdown", "") or ""

                        st.markdown("##### 📄 영어 마크다운 원문 (content_markdown)")
                        if en_md.strip():
                            st.code(en_md, language="markdown")
                        else:
                            st.info("영어 마크다운 텍스트가 비어 있습니다.")

                        st.markdown("##### 📄 한국어 마크다운 원문 (content_markdown_translated)")
                        if ko_md.strip():
                            st.code(ko_md, language="markdown")
                        else:
                            st.info("한국어 마크다운 텍스트가 비어 있습니다.")

                        st.markdown("##### 🧷 MARKDOWN_REPORT (두 언어 마크다운 오류 통합)")
                        if markdown_report.strip():
                            st.markdown(markdown_report)
                        else:
                            st.info("마크다운 관련으로 보고된 오류가 없습니다.")
                else:
                    st.warning("선택한 행의 데이터를 찾을 수 없습니다.")



# --- 설명 탭 ---
with tab_about:
    st.markdown("""
## 이 앱은?

- 한국어/영어 **단일 텍스트 검수기** + **Google Sheets 기반 배치 검수기**입니다.
- 스타일/어투/자연스러움은 건드리지 않고, **오탈자 / 조사 / 띄어쓰기 / 기본 문장부호 / 단순 스펠링 오류**에만 집중합니다.
""")


# --- 디버그 탭 ---
with tab_debug:
    st.markdown("여기는 추후에 로그, 디버그용 정보를 추가로 표시할 수 있는 영역입니다.")
