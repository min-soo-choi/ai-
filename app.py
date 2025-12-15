# app.py
# -*- coding: utf-8 -*-
import json
import time
import re
import html
from collections import Counter
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

# 한 chunk당 최대 길이 (원하는 값으로 조정 가능)
MAX_KO_CHUNK_LEN = 1000  # 한글 800~1200자 정도면 안정적

def split_korean_text_into_chunks(text: str, max_len: int = MAX_KO_CHUNK_LEN) -> List[str]:
    """
    긴 한국어 텍스트를 여러 chunk로 나눈다.
    - 기본 기준: max_len 글자
    - 가능하면 줄바꿈(\n) 앞에서 끊어서 문단 단위에 가깝게 유지
    """
    if not text:
        return []

    text = text.replace("\r\n", "\n")
    if len(text) <= max_len:
        return [text]

    chunks: List[str] = []
    n = len(text)
    start = 0

    while start < n:
        end = min(start + max_len, n)

        # end 근처에서 줄바꿈 기준으로 끊을 수 있으면 거기서 끊기
        split_pos = text.rfind("\n", start + int(max_len * 0.4), end)
        if split_pos == -1 or split_pos <= start:
            split_pos = end

        chunk = text[start:split_pos].strip("\n")
        if chunk:
            chunks.append(chunk)

        start = split_pos

    return chunks

# -------------------------------------------------
# PDF 텍스트 정리용 프롬프트 + 래퍼
# -------------------------------------------------

PDF_RESTORE_SYSTEM_PROMPT = """
너는 PDF에서 복사해 붙여넣은 한국어 시험지/해설 텍스트를,
원문의 의미를 유지하면서 구조와 서식을 정리해 주는 도우미이다.
아래 규칙을 순서대로, 엄격하게 지켜라.

1. 텍스트 복원 및 정비
- 오타 및 깨진 글자 복원:
  입력된 텍스트에서 OCR 오류로 보이는 깨진 문자(예: , ᆢ)나 명백한 오타
  (예: 연공 지능 → 인공 지능)를 문맥에 맞게 올바른 한글, 한자, 문장부호로 복원한다.
- 원문 유지:
  텍스트의 내용을 임의로 창작하거나 왜곡하지 말고, 원문의 의미를 그대로 보존한다.

2. 헤더(제목) 텍스트 변경 규칙 (중요)
텍스트 내의 다음 키워드들을 찾아 지정된 표준 헤더로 변경한다.

[정답 해설]
- 정답
- 정답인 이유
- ( ) 정답인 이유
- 정답 해설
- 정답 설명
- 해설
- [ ] 해설
- 해설:
※ ‘해설’ 관련 표현은 모두 [정답 해설]로 통합

[오답 해설]
- 오답
- 오답 해설
- 오답 풀이
- ( ) 오답 해설
- ( ) 해설 (문맥상 오답 풀이일 경우)

[적절하지 않은 이유]
- ➜ 적절하지 않은 이유
※ 화살표(➜)가 있는 경우

[적절한 이유]
- ➜ 적절한 이유
※ 화살표(➜)가 없는 경우

[출제 의도]
- 출제 의도
- 출제의도
※ 괄호만 [] 형태로 변경

[중세의도]
- 중세의도
※ 괄호만 [] 형태로 변경

3. 헤더 순서 재배치 (구조 교정)
- 변환 작업을 마친 후, 만약 [오답 해설]이 [정답 해설]보다 먼저 나오는 경우
  텍스트 내용은 그대로 두고 헤더의 위치만 서로 맞바꾼다.
- 목표 순서:
  반드시 [정답 해설] → [오답 해설] 순서를 유지한다.
- 헤더 바로 아래에 오는 본문 내용들은 헤더와 함께 묶어서 이동시킨다.

4. 문장 및 서식 정리 (가독성 최적화)
- 줄바꿈 병합:
  문장의 중간이 어색하게 끊겨 있는 경우, 이를 공백으로 치환하여 자연스럽게 연결한다.
- 번호 목록 분리:
  문장 중간이나 끝에 원 문자(①, ②, ③… / ㉠, ㉡…)가 붙어 있는 경우
  반드시 줄을 바꾼 뒤 번호를 시작한다.
- 빈 줄 제거:
  불필요한 빈 줄(엔터 두 번 이상)은 제거하고,
  단일 줄바꿈(엔터 한 번)만 사용한다.

※ 가능한 한 기존 텍스트에 있던 원기호/선지 내용을 그대로 사용하되,
   줄 위치와 줄바꿈만 정리한다.

5. 최종 출력 형식
- 완성된 텍스트는 복사하기 쉽도록
  반드시 회색 코드 블록(Code Block) 안에 담아서 출력한다.
- 코드 블록 밖에는 어떤 설명도 출력하지 말고,
  오직 정리된 텍스트만 코드 블록 안에 넣어라.
- 코드 블록 언어 표시는 text로 사용해도 되고, 생략해도 된다.

6) 블록 간 공백 규칙
- [정답 해설] 블록과 그 다음 블록 사이에는 빈 줄을 정확히 1줄만 둔다.
- [오답 해설] 블록과 원기호(①, ②, ㉠…) 목록 사이에도 빈 줄을 정확히 1줄만 둔다.
- 블록 내부에서는 불필요한 연속 빈 줄을 제거하고 논리적으로 필요한 경우에만 단일 줄바꿈을 유지한다.

"""

def normalize_inline_answer_marker(text: str) -> str:
    """
    문항 번호 + 정답 기호가 문장 안에 섞여 있는 경우를 정규화한다.

    예:
    "1) ④ ( ) ( ) 출제 유형 ... [정답 해설] ..."
    →
    "1) 정답: ④\n[정답 해설] ..."
    """
    if not text:
        return text

    text = text.replace("\r\n", "\n")

    # ①②③④⑤⑥⑦⑧⑨⑩
    circled_nums = "①②③④⑤⑥⑦⑧⑨⑩"

    # 문항 번호 + 정답 기호 패턴
    pattern = re.compile(
        rf"""
        (\b\d+\))            # 1) 같은 문항 번호
        \s*
        ([{circled_nums}])   # ④ 같은 정답 기호
        .*?
        (?=\[정답\s*해설\])  # [정답 해설] 직전까지만 먹음
        """,
        re.VERBOSE | re.DOTALL,
    )

    def repl(m):
        qno = m.group(1)
        ans = m.group(2)
        return f"{qno} 정답: {ans}\n"

    return pattern.sub(repl, text)


def tighten_between_answer_blocks(text: str) -> str:
    """
    [정답 해설] 블록과 [오답 해설] 헤더 사이에 들어간
    '빈 줄 1줄(또는 여러 줄)'을 제거해서 바로 붙인다.

    예)
    [정답 해설]
    해설 내용

    [오답 해설]

    → [정답 해설]
      해설 내용
      [오답 해설]
    """
    if not text:
        return text

    # '\n(빈 줄들)\n[오답 해설]' 패턴을 '\n[오답 해설]'로 바꿈
    # \s* 때문에 공백/탭이 섞여 있어도 같이 제거됨
    text = re.sub(r"\n\s*\n(\[오답 해설\])", r"\n\1", text)
    return text

def restore_pdf_text(raw_text: str) -> str:
    """
    PDF에서 복사한 난장판 텍스트를, 위 규칙에 따라 정리해 달라고 Gemini에 요청.
    - 입력: 원본 텍스트
    - 출력: 모델이 반환한 문자열 (가능하면 코드 블록을 그대로 사용)
    """
    if not raw_text:
        return ""

    # 모델에 넘길 프롬프트 구성
    prompt = f"""{PDF_RESTORE_SYSTEM_PROMPT}

----------------------------------------
아래는 PDF에서 복사해온 원본 텍스트이다.
이 텍스트를 위 규칙에 따라 정리하라.
반드시 정리된 최종 텍스트만 코드 블록 안에 넣어서 출력할 것.

[원본 텍스트 시작]
{raw_text}
[원본 텍스트 끝]
"""

    # 이 기능은 JSON이 아니라 순수 텍스트를 기대하므로
    # response_mime_type은 지정하지 않는다.
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.0},
    )
    text = getattr(response, "text", "") or ""
    stripped = text.strip()

    # 코드블록 안/밖을 처리하기 전에, 내용 부분 먼저 정리
    # 1) 코드블록이면 안쪽만 꺼내서 가공
    m = re.match(r"^```[^\n]*\n(.*)\n```$", stripped, re.S)
    if m:
        inner = m.group(1)
        inner = normalize_inline_answer_marker(inner)
        inner = tighten_between_answer_blocks(inner)
        stripped = f"```text\n{inner}\n```"
    else:
        # 코드블록이 아니라면 우리가 감싸주면서 정리
        inner = tighten_between_answer_blocks(stripped)
        inner = normalize_inline_answer_marker(inner)
        stripped = f"```text\n{inner}\n```"

    return stripped

def remove_first_line_in_code_block(block: str) -> str:
    """
    ```text
    AAA
    BBB
    CCC
    ```
    이런 문자열에서 AAA 줄만 지우고

    ```text
    BBB
    CCC
    ```
    로 돌려준다.
    코드블록이 아니어도 그냥 첫 줄만 제거해서 반환.
    """
    if not block:
        return block

    stripped = block.strip()

    # 1) 코드블록 형태인지 먼저 확인
    m = re.match(r"^```[^\n]*\n(.*)\n```$", stripped, re.S)
    if m:
        inner = m.group(1)
    else:
        inner = stripped

    lines = inner.splitlines()
    if not lines:
        new_inner = ""
    else:
        # 첫 줄 제거
        new_inner = "\n".join(lines[1:])

    # 코드블록이었던 경우 다시 감싸서 반환
    if m:
        return f"```text\n{new_inner}\n```"
    else:
        return new_inner




def _parse_report_with_pattern(source_text: str, report: str, pattern: re.Pattern[str]) -> List[Dict[str, Any]]:
    """
    공용 파서: "- '원문' → '수정안': 설명" 포맷을 받아 위치 정보를 계산한다.
    pattern: 언어별 허용 따옴표/화살표를 반영한 정규식.
    """
    if not report:
        return []

    # 원문 텍스트를 한 줄씩 쪼개고, 각 줄의 시작 offset을 기록
    lines = source_text.splitlines(keepends=True)
    line_starts: List[int] = []
    offset = 0
    for ln in lines:
        line_starts.append(offset)
        offset += len(ln)

    def index_to_line_col(idx: int) -> tuple[int, int]:
        line_no = 1
        for i, start in enumerate(line_starts):
            if i + 1 < len(line_starts) and line_starts[i + 1] <= idx:
                line_no += 1
            else:
                break
        line_start_idx = line_starts[line_no - 1]
        col_no = idx - line_start_idx + 1
        return line_no, col_no

    results: List[Dict[str, Any]] = []

    for line in report.splitlines():
        s = line.strip()
        if not s:
            continue

        m = pattern.match(s)
        if not m:
            continue

        orig = m.group(1)
        fixed = m.group(2)
        msg = m.group(3)

        idx = source_text.find(orig)
        if idx == -1:
            results.append({
                "original": orig,
                "fixed": fixed,
                "message": msg,
                "line": None,
                "col": None,
            })
            continue

        line_no, col_no = index_to_line_col(idx)
        results.append({
            "original": orig,
            "fixed": fixed,
            "message": msg,
            "line": line_no,
            "col": col_no,
        })

    return results


def parse_korean_report_with_positions(source_text: str, report: str) -> List[Dict[str, Any]]:
    """
    한국어용 리포트 파서
    - 기본: '- "원문" → "수정안": 설명' 형식
    - 허용: 따옴표 유무 모두 허용, 스마트 따옴표 허용, 종결부호 누락/여분 따옴표도 관대하게 매칭
    - 화살표는 → 또는 -> 허용
    """
    patterns = [
        # 1) 정규 포맷: 양쪽에 따옴표 있음
        re.compile(
            r"""^-\s*['"“”‘’](.+?)['"“”‘’]\s*(?:→|->)\s*['"“”‘’](.+?)['"“”‘’]\s*:\s*(.+?)\s*['"“”‘’]?$""",
            re.UNICODE,
        ),
        # 2) 따옴표가 아예 없는 경우도 허용
        re.compile(
            r"""^-\s*(.+?)\s*(?:→|->)\s*(.+?)\s*:\s*(.+?)\s*['"“”‘’]?$""",
            re.UNICODE,
        ),
    ]

    for pat in patterns:
        results = _parse_report_with_pattern(source_text, report, pat)
        if results:
            return results

    return []


def parse_english_report_with_positions(source_text: str, report: str) -> List[Dict[str, Any]]:
    """
    영어용 리포트 파서
    - 포맷은 동일하지만 영어 전용 규칙을 분리할 수 있도록 별도 함수로 유지
    """
    pattern = re.compile(
        r"""^-\s*['"“”‘’](.+?)['"“”‘’]\s*(?:→|->)\s*['"“”‘’](.+?)['"“”‘’]\s*:\s*(.+)$""",
        re.UNICODE,
    )
    return _parse_report_with_pattern(source_text, report, pattern)


# ✅ 하위 호환: 기본 파서는 한국어 규칙으로 동작
def parse_report_with_positions(source_text: str, report: str) -> List[Dict[str, Any]]:
    return parse_korean_report_with_positions(source_text, report)

def build_english_raw_report_for_highlight(raw_json: dict) -> str:
    """
    영어 raw_json에서 하이라이트용 리포트 문자열을 만든다.
    - two_pass_single_en 모드: 1차 Detector 기준 리포트 사용 (더 과검출)
    - 그 외: content_typo_report를 그대로 사용
    """
    if not isinstance(raw_json, dict):
        return ""

    mode = raw_json.get("mode")

    if mode == "two_pass_single_en":
        draft = raw_json.get("initial_report_from_detector", "") or ""
        return draft.strip()

    # fallback: 혹시 모드를 안 쓴 경우
    return (raw_json.get("content_typo_report") or "").strip()




def build_korean_raw_report_for_highlight(raw_json: dict) -> str:
    """
    한국어 raw_json에서 하이라이트용 리포트 문자열을 만든다.
    - single block: raw_json["translated_typo_report"] 그대로 사용
    - chunked: 각 chunk.raw.translated_typo_report를 블록 헤더와 함께 이어붙임
    """
    if not isinstance(raw_json, dict):
        return ""

    # chunking 모드
    if raw_json.get("mode") == "chunked":
        st.info("※ 텍스트가 길어 여러 블록으로 나뉘어 검사되었으며, \ 1차/2차 JSON은 chunk별 raw 정보로만 존재합니다.")
    else:
        with st.expander("1차 Detector JSON (필요 시)", expanded=False):
            st.json(raw_json.get("detector_clean", {}))
        with st.expander("2차 Judge JSON (필요 시)", expanded=False):
            st.json(raw_json.get("judge_clean", {}))
        lines: List[str] = []
        for chunk in raw_json.get("chunks", []):
            idx = chunk.get("index")
            raw = chunk.get("raw") or {}
            report = (raw.get("translated_typo_report") or "").strip()
            if not report:
                continue
            if idx is not None:
                lines.append(f"# [블록 {idx}]")
            lines.append(report)
        return "\n".join(lines)

    # 단일 블록 모드
    return (raw_json.get("translated_typo_report") or "").strip()

PUNCT_COLOR_MAP = {
    ".": "#fff3cd",  # 연노랑 (종결부호)
    "?": "#f8d7da",  # 연분홍 (물음표)
    "!": "#f5c6cb",  # 연한 빨강 (느낌표)
    ",": "#d1ecf1",  # 연하늘 (쉼표)
    ";": "#d6d8d9",  # 회색 톤 (세미콜론)
    ":": "#d6d8d9",  # 회색 톤 (콜론)
    '"': "#e0f7e9",  # 연연두 (쌍따옴표)
    "“": "#e0f7e9",
    "”": "#e0f7e9",
    "'": "#fce9d9",  # 연살구 (작은따옴표)
    "‘": "#fce9d9",
    "’": "#fce9d9",
}

PUNCT_GROUPS: dict[str, set[str]] = {
    "종결부호(.)": {"."},
    "물음표(?)": {"?"},
    "느낌표(!)": {"!"},
    "쉼표(,)": {","},
    "쌍따옴표": {'"', "“", "”"},
    "작은따옴표": {"'", "‘", "’"},
}

# 한국어/영어에서 자주 쓰는 문장부호 세트
PUNCT_CHARS = set(PUNCT_COLOR_MAP.keys()) | set([
    # 큰따옴표/작은따옴표
    '"', "'", "“", "”", "‘", "’",
    # 괄호류
    "(", ")", "[", "]", "{", "}",
    "「", "」", "『", "』", "〈", "〉", "《", "》",
    # 기타
    "…", "·",
])


def highlight_text_with_spans(
    source_text: str,
    spans: List[Dict[str, Any]],
    selected_punct_chars: set[str] | None = None,
) -> str:
    """
    spans: parse_report_with_positions() 결과.
    - spans에 해당하는 'original' 구간은 <mark>...</mark> 로 감싸서 오류 하이라이트.
    - 그 밖의 영역에 있는 문장부호는 기호별로 색을 다르게 주어 <span style="...">로 감싼다.

    ⚠️ 설계:
      - 오류 구간(<mark>) 안의 문장부호는 추가 색칠 없이 mark만 적용 (이미 강한 하이라이트).
      - 오류가 아닌 영역의 문장부호만 색상 하이라이트.
    """
    if not source_text:
        return ""

    # 1) 오류 구간 interval 계산
    intervals: List[tuple[int, int]] = []

    if spans:
        for span in spans:
            orig = span.get("original")
            if not orig:
                continue
            start = source_text.find(orig)
            if start == -1:
                continue
            end = start + len(orig)
            intervals.append((start, end))

    # intervals가 없으면, 오류는 없고 문장부호만 색칠
    if not intervals:
        result_parts: List[str] = []
        for ch in source_text:
            if ch in PUNCT_CHARS and (selected_punct_chars is None or ch in selected_punct_chars):
                color = PUNCT_COLOR_MAP.get(ch, "#e2e3e5")
                result_parts.append(
                    f"<span style='background-color: {color}; padding: 0 2px; font-weight: 700; font-size: 1.05em; border-radius: 2px;'>{html.escape(ch)}</span>"
                )
            else:
                result_parts.append(html.escape(ch))
        return "".join(result_parts)

    # 2) 오류 interval 정리 (겹치는 구간 병합)
    intervals.sort(key=lambda x: x[0])
    merged_intervals: List[tuple[int, int]] = []
    cur_start, cur_end = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_end:  # 겹치면 병합
            cur_end = max(cur_end, e)
        else:
            merged_intervals.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged_intervals.append((cur_start, cur_end))

    # 3) 한 글자씩 순회하며 HTML 생성
    result_parts: List[str] = []
    idx = 0
    interval_idx = 0
    in_error = False
    cur_err_end = None

    while idx < len(source_text):
        # 현재 위치가 새로운 오류 interval의 시작인지 확인
        if interval_idx < len(merged_intervals):
            start, end = merged_intervals[interval_idx]
        else:
            start, end = None, None

        if (not in_error) and (start is not None) and (idx == start):
            # 오류 구간 시작
            in_error = True
            cur_err_end = end
            result_parts.append("<mark style='background: #fff3a3; padding: 0 2px; font-weight: 700; font-size: 1.05em; border-radius: 2px;'>")

        ch = source_text[idx]

        if in_error:
            # 오류 구간 안에서는 문장부호 색칠 X, mark만 사용
            result_parts.append(html.escape(ch))
            idx += 1

            # 오류 구간 끝났는지 체크
            if cur_err_end is not None and idx >= cur_err_end:
                result_parts.append("</mark>")
                in_error = False
                interval_idx += 1
                cur_err_end = None
        else:
            # 오류 구간 밖: 문장부호면 색상 하이라이트
            if ch in PUNCT_CHARS and (selected_punct_chars is None or ch in selected_punct_chars):
                color = PUNCT_COLOR_MAP.get(ch, "#e2e3e5")
                result_parts.append(
                    f"<span style='background-color: {color}; padding: 0 2px; font-weight: 700; font-size: 1.05em; border-radius: 2px;'>{html.escape(ch)}</span>"
                )
            else:
                result_parts.append(html.escape(ch))
            idx += 1

    # 혹시 오류 구간이 열린 채로 끝난 경우 닫아주기 (이론상 거의 없음)
    if in_error:
        result_parts.append("</mark>")

    return "".join(result_parts)


def highlight_selected_punctuation(source_text: str, selected_keys: list[str]) -> str:
    """
    선택된 문장부호 그룹만 색상 하이라이트하고 나머지는 일반 텍스트로 보여준다.
    """
    if not source_text:
        return ""

    selected_chars: set[str] = set()
    for key in selected_keys:
        selected_chars.update(PUNCT_GROUPS.get(key, set()))

    result_parts: List[str] = []
    for ch in source_text:
        if ch in selected_chars and ch in PUNCT_COLOR_MAP:
            color = PUNCT_COLOR_MAP.get(ch, "#e2e3e5")
            result_parts.append(
                f"<span style='background-color: {color}; padding: 0 3px; font-weight: 700; font-size: 1.1em; border-radius: 3px;'>{html.escape(ch)}</span>"
            )
        else:
            result_parts.append(html.escape(ch))
    return "".join(result_parts)




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
    
    pattern = re.compile(
        r"""^-\s*(['"])(.+?)\1\s*(?:→|->)\s*(['"])(.+?)\3\s*:\s*(.+)$""",
        re.UNICODE,
    )

    for line in report.splitlines():
        s = line.strip()
        if not s:
            continue

        m = pattern.match(s)
        if not m:
            cleaned.append(s)
            continue

        original = m.group(2)
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
    
    pattern = re.compile(
    r"""^-\s*(['"])(.+?)\1\s*(?:→|->)\s*(['"])(.+?)\3\s*:""",
    re.UNICODE,
)

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

        orig = m.group(2).strip()
        fixed = m.group(4).strip()

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


def drop_false_whitespace_claims(text: str, report: str) -> str:
    """
    '불필요한 공백'류를 지적했지만 원문 조각에 공백/제로폭 공백이 전혀 없으면 제거한다.
    """
    if not report:
        return ""

    cleaned: list[str] = []
    pattern = re.compile(r"^- '(.+?)' → '(.+?)':.*(불필요한 공백|띄어쓰기|공백)", re.UNICODE)

    for line in report.splitlines():
        s = line.strip()
        if not s:
            continue

        m = pattern.match(s)
        if not m:
            cleaned.append(s)
            continue

        original = m.group(1)
        # 실제 공백/제로폭 공백이 하나도 없으면 오탐으로 간주
        if not re.search(r"[ \t\u3000\u200b\u200c\u200d]", original):
            continue

        cleaned.append(s)

    return "\n".join(cleaned)


def ensure_final_punctuation_error(text: str, report: str) -> str:
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

    # 🔴 여기에서 '수 있었다' 같은 예시를 쓰지 말고,
    #     그냥 설명만 추가한다.
    line = "- 문단 마지막 문장 끝에 마침표(또는 물음표, 느낌표)가 빠져 있으므로 적절한 문장부호를 추가해야 합니다."

    if report:
        return report.rstrip() + "\n" + line
    else:
        return line



def ensure_english_final_punctuation(text: str, report: str) -> str:
    """
    영어 텍스트의 '마지막 문장'이 ., ?, ! 로 끝나지 않으면
    아주 보수적인 요약 경고 한 줄을 추가한다.
    (쉼표/세미콜론/콜론 등으로 끝나는 경우 포함)
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
    # 따옴표/괄호 뒤에 .?! 가 있는 경우 허용
    elif last in ['"', "'", ")", "]", "”", "’"] and len(s) >= 2 and s[-2] in ".?!":
        end_ok = True

    if end_ok:
        return report or ""

    # 이미 비슷한 문구가 있으면 중복 추가 방지
    if report and ("종결부호" in report or "마침표" in report or "punctuation" in report):
        return report

    line = "- 마지막 문장이 종결부호(., ?, !)가 아닌 문장부호로 끝나 있어, 문장을 마침표 등으로 명확히 끝내는 것이 좋습니다."

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

def create_korean_detector_prompt_for_text(korean_text: str) -> str:
    """
    1차 패스: Detector
    - 가능한 많은 '잠재적 오류 후보'를 찾는 역할 (약간 과검출 허용)
    """
    safe_text = json.dumps(korean_text, ensure_ascii=False)

    prompt = f"""
당신은 1차 **Korean text proofreader (Detector)**입니다.
당신의 임무는 아래 한국어 텍스트에서 발생할 수 있는
**모든 잠재적 오류 후보를 최대한 많이 탐지하는 것**입니다.

이 단계에서는 약간의 과잉 탐지(False Positive)를 허용합니다.
(2차 Judge 단계에서 의미 변경·스타일 제안 등은 제거됩니다.)

출력은 반드시 아래 4개의 key만 포함하는 **단일 JSON 객체**여야 합니다.
- "suspicion_score": 1~5 정수
- "content_typo_report": "" (비워두기 — 영어용 필드)
- "translated_typo_report": "- '원문' → '수정안': 설명" 형식의 줄을 여러 개 포함한 문자열 (없으면 "")
- "markdown_report": "" (항상 빈 문자열)

모든 설명은 반드시 **한국어로** 작성해야 합니다.

------------------------------------------------------------
# 입력 텍스트 (JSON 문자열)
------------------------------------------------------------

아래는 전체 한국어 텍스트를 JSON 문자열로 인코딩한 값입니다.
이 값을 그대로 디코딩한 텍스트(plain_korean)를 기준으로만 검수해야 합니다.

plain_korean_json: {safe_text}

- plain_korean_json을 디코딩한 결과를 plain_korean이라고 부릅니다.
- "- '원문' → '수정안': 설명" 형식에서 '원문'은
  반드시 plain_korean 안에 실제로 존재하는 부분 문자열이어야 합니다.

------------------------------------------------------------
# 1. 이 단계에서 꼭 잡아야 하는 오류 (넓게 탐지)
------------------------------------------------------------

- 명백한 오탈자, 철자 오류
- 잘못된 띄어쓰기/붙여쓰기
- 조사·어미 오용
- 문장부호 오류 (마침표/쉼표/따옴표 짝/괄호 짝 등)
- 단어 내부가 이상하게 분리된 경우 (예: "된 다", "하 였다" 등)

이 단계에서는 다소 애매한 것까지 **후보로 잡아도** 괜찮습니다.
2차 Judge가 의미 변경/스타일 제안 등을 필터링합니다.

이제 plain_korean_json을 디코딩하여 plain_korean을 얻은 뒤,
위 기준에 따라 "- '원문' → '수정안': 설명" 형식으로 translated_typo_report를 생성하십시오.
"""
    return prompt


def create_korean_judge_prompt_for_text(korean_text: str, draft_report: str) -> str:
    """
    2차 패스: Judge
    - 1차 Detector가 만든 후보들(draft_report) 중에서
      '의미를 바꾸지 않는 객관적인 오류 수정'만 남기고 나머지를 제거하는 역할.
    """
    safe_text = json.dumps(korean_text, ensure_ascii=False)
    safe_report = json.dumps(draft_report, ensure_ascii=False)

    prompt = f"""
당신은 2차 **Korean text proofreader (Judge)**입니다.

역할:
- 1차 Detector가 만든 오류 후보 목록(draft_report) 중에서
  **의미를 바꾸지 않는 객관적인 오류만 남기고 나머지는 모두 제거**하는 것입니다.

------------------------------------------------------------
# 입력 1: 전체 한국어 원문 (JSON 문자열)
------------------------------------------------------------
plain_korean_json: {safe_text}

- plain_korean_json을 디코딩한 결과를 plain_korean이라고 부릅니다.

------------------------------------------------------------
# 입력 2: 1차 Detector의 후보 리포트 (JSON 문자열)
------------------------------------------------------------
draft_report_json: {safe_report}

- draft_report_json은 문자열이며,
  내부 형식은 "- '원문' → '수정안': 설명" 줄들이 줄바꿈으로 이어진 형태입니다.

각 줄에 대해 아래 기준으로 **채택/제거 여부**를 판단하십시오.

------------------------------------------------------------
# 채택 기준 (모든 조건을 만족해야 함)
------------------------------------------------------------

1. '원문'은 plain_korean 안에 실제로 존재하는 부분 문자열이어야 한다.
2. '수정안'은 다음과 같은 **형식적·객관적 수정**만 포함해야 한다.
   - 띄어쓰기/붙여쓰기 수정
   - 조사/어미 교정
   - 명백한 오탈자·철자 오류
   - 문장부호(마침표, 쉼표, 따옴표, 괄호 등) 교정
3. 의미를 바꾸는 어휘 변경이나 문장 구조 변경은 모두 제거한다.
4. 자연스러운 표현, 문체 개선, 톤 조정, 길이 줄이기/늘리기 등
   **스타일/표현 개선 목적의 수정**은 모두 제거한다.
5. plain_korean에 존재하지 않는 단어·구절을 '원문'으로 인용한 줄은 제거한다.

------------------------------------------------------------
# 출력
------------------------------------------------------------

반환 값은 반드시 아래 4개의 key를 가진 **단일 JSON 객체**여야 합니다.
- "suspicion_score": 1~5 정수 (남은 오류 후보의 심각도에 따라 판단)
- "content_typo_report": "" (비워두기)
- "translated_typo_report":
    draft_report_json에 포함된 줄들 중에서
    위 기준을 만족하는 줄만 남긴 "- '원문' → '수정안': 설명" 문자열
    (각 줄은 줄바꿈으로 구분)
- "markdown_report": "" (항상 빈 문자열)

draft_report_json에 있던 줄이라도, 위 기준을 만족하지 못하면
해당 줄은 완전히 제거하여 translated_typo_report에 포함하지 마십시오.
"""
    return prompt

# -------- Stage helpers (Detector / Judge / Final) --------

def get_korean_stage_reports(raw_bundle: dict, final_report: str) -> dict:
    """
    한국어 1차 / 2차 / 최종 리포트 문자열을 stage별로 돌려준다.
    return 예시:
    {
        "detector": "...",
        "judge": "...",
        "final": "..."
    }
    """
    if not isinstance(raw_bundle, dict):
        raw_bundle = {}

    detector_report = ""
    judge_report = ""

    # chunked 모드: 블록별 리포트를 헤더와 함께 이어붙인다.
    if raw_bundle.get("mode") == "chunked":
        det_lines: list[str] = []
        judge_lines: list[str] = []
        for chunk in raw_bundle.get("chunks", []):
            idx = chunk.get("index")
            raw = chunk.get("raw") or {}

            det_line = ""
            det_line = (raw.get("initial_report_from_detector") or "").strip()
            if not det_line:
                det_clean = raw.get("detector_clean") or {}
                if isinstance(det_clean, dict):
                    det_line = (det_clean.get("translated_typo_report") or "").strip()

            judge_line = (raw.get("final_report_before_rule_postprocess") or "").strip()
            if not judge_line:
                judge_clean = raw.get("judge_clean") or {}
                if isinstance(judge_clean, dict):
                    judge_line = (judge_clean.get("translated_typo_report") or "").strip()
            if not judge_line:
                judge_line = (raw.get("translated_typo_report") or "").strip()

            header = f"# [블록 {idx}]" if idx is not None else None
            if det_line:
                if header:
                    det_lines.append(header)
                det_lines.append(det_line)
            if judge_line:
                if header:
                    judge_lines.append(header)
                judge_lines.append(judge_line)

        detector_report = "\n".join(det_lines).strip()
        judge_report = "\n".join(judge_lines).strip()

    else:
        # 단일 블록 모드
        detector_clean = raw_bundle.get("detector_clean") or {}
        if isinstance(detector_clean, dict):
            detector_report = (detector_clean.get("translated_typo_report") or "").strip()

        judge_clean = raw_bundle.get("judge_clean") or {}
        if isinstance(judge_clean, dict):
            judge_report = (judge_clean.get("translated_typo_report") or "").strip()
        if not judge_report:
            judge_report = (raw_bundle.get("translated_typo_report") or "").strip()

    return {
        "detector": detector_report,
        "judge": judge_report,
        "final": (final_report or "").strip(),
    }


def get_english_stage_reports(raw_bundle: dict, final_report: str) -> dict:
    """
    영어 1차 / 2차 / 최종 리포트 반환
    """
    if not isinstance(raw_bundle, dict):
        raw_bundle = {}

    # 1차 Detector: initial_report_from_detector 우선
    detector_report = (raw_bundle.get("initial_report_from_detector") or "").strip()
    if not detector_report:
        detector_clean = raw_bundle.get("detector_clean") or {}
        if isinstance(detector_clean, dict):
            detector_report = (detector_clean.get("content_typo_report") or "").strip()

    # 2차 Judge: final_report_before_rule_postprocess 우선
    judge_report = (raw_bundle.get("final_report_before_rule_postprocess") or "").strip()
    if not judge_report:
        judge_clean = raw_bundle.get("judge_clean") or {}
        if isinstance(judge_clean, dict):
            judge_report = (judge_clean.get("content_typo_report") or "").strip()
    if not judge_report:
        judge_report = (raw_bundle.get("content_typo_report") or "").strip()

    return {
        "detector": detector_report,
        "judge": judge_report,
        "final": (final_report or "").strip(),
    }


def create_korean_review_prompt_for_text(korean_text: str) -> str:
    
     # 원문을 JSON 문자열로 한 번 감싸서, 인용부호/줄바꿈/특수문자를 안전하게 전달
    safe_text = json.dumps(korean_text, ensure_ascii=False)
    
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
- 단어 내부에서 불필요한 공백(예: '흘 린다', '된 다')만 오류로 인정한다.

============================================================
# 2. OUTPUT FORMAT (JSON Only)
============================================================
오류가 있을 경우 한 줄씩 bullet:

"- '원문' → '수정안': 오류 설명"

------------------------------------------------------------
# 3. 검사할 텍스트
------------------------------------------------------------

아래는 검수할 한국어 전체 텍스트를 JSON 문자열로 인코딩한 값입니다.
이 값을 그대로 문자열로 복원하여 검수에 사용하세요.

plain_korean_json: {safe_text}

- plain_korean_json 값은 JSON 인코딩된 문자열입니다.
- 이 값을 그대로 디코딩한 텍스트(plain_korean)를 기준으로만
  '- '원문' → '수정안': ...' 형식의 리포트를 생성해야 합니다.
- '원문' 부분은 반드시 plain_korean 안에 실제로 존재하는 부분 문자열이어야 합니다.

이제 위 규칙을 지키며 plain_korean_json에 담긴 한국어 텍스트를 검수하세요.
"""
    return prompt


def _review_korean_single_block(korean_text: str) -> Dict[str, Any]:
    """
    ✅ 2패스(Detector → Judge) 기반 한국어 단일 블록 검수
    1차: Detector 프롬프트로 가능한 많은 오류 후보를 수집
    2차: Judge 프롬프트로 의미 변경/스타일 제안/환각 등을 필터링
    + 기존 규칙 기반 후처리(drop_lines_not_in_source 등)를 한 번 더 적용
    """

    # 1️⃣ 1차 패스: Detector
    detector_prompt = create_korean_detector_prompt_for_text(korean_text)
    detector_raw = analyze_text_with_gemini(detector_prompt)
    detector_clean = validate_and_clean_analysis(detector_raw)

    draft_report = detector_clean.get("translated_typo_report", "") or ""

    # 2️⃣ 2차 패스: Judge
    judge_prompt = create_korean_judge_prompt_for_text(korean_text, draft_report)
    judge_raw = analyze_text_with_gemini(judge_prompt)
    judge_clean = validate_and_clean_analysis(judge_raw)

    # 2차 결과 기준으로 점수/리포트 사용
    score = judge_clean.get("suspicion_score", 1)
    try:
        score = int(score)
    except Exception:
        score = 3

    final_report = judge_clean.get("translated_typo_report", "") or ""

    # 3️⃣ 규칙 기반 후처리 (기존 로직 그대로 유지)
    filtered = drop_lines_not_in_source(
        korean_text,
        final_report,
    )
    filtered = drop_false_korean_period_errors(filtered)
    filtered = drop_false_whitespace_claims(korean_text, filtered)
    filtered = ensure_final_punctuation_error(korean_text, filtered)
    filtered = ensure_sentence_end_punctuation(korean_text, filtered)
    filtered = dedup_korean_bullet_lines(filtered)
    filtered = drop_lines_not_in_source(korean_text, filtered)  # 한 번 더 검증

    # 4️⃣ raw 번들 구성 (UI 호환 + 디버그용 정보 포함)
    raw_bundle = {
        "mode": "two_pass_single",
        # UI가 그대로 쓸 수 있도록 상위 요약값도 넣어둠
        "suspicion_score": score,
        "translated_typo_report": final_report,
        # 디버그용 상세 단계 정보
        "detector_raw": detector_raw,
        "detector_clean": detector_clean,
        "judge_raw": judge_raw,
        "judge_clean": judge_clean,
        "initial_report_from_detector": draft_report,
        "final_report_before_rule_postprocess": final_report,
    }

    return {
        "score": score,
        "content_typo_report": "",          # 한국어 탭에서는 사용 안 함
        "translated_typo_report": filtered, # 규칙 기반 후처리까지 적용된 최종 리포트
        "markdown_report": "",
        "raw": raw_bundle,
    }

def review_korean_text(korean_text: str) -> Dict[str, Any]:
    """
    한국어 텍스트 검수 (chunk 지원 버전)

    - 텍스트 길이가 짧으면: 기존 single block 로직 그대로 사용
    - 텍스트가 길면: 여러 chunk로 나눈 뒤, 각 chunk를 개별 검수해서
      리포트를 합쳐서 반환
    """
    # 1) chunking
    chunks = split_korean_text_into_chunks(korean_text, max_len=MAX_KO_CHUNK_LEN)

    # chunk가 1개면 기존 로직 그대로
    if len(chunks) == 1:
        return _review_korean_single_block(korean_text)

    # 2) 여러 chunk를 순차 검수
    merged_report_lines: List[str] = []
    raw_list: List[Dict[str, Any]] = []
    max_score = 1

    for idx, chunk in enumerate(chunks, start=1):
        res = _review_korean_single_block(chunk)

        score = res.get("score", 1) or 1
        max_score = max(max_score, score)

        report = (res.get("translated_typo_report") or "").strip()
        if report:
            # 필요하면 chunk 번호를 구분용 헤더로 달아줄 수 있음
            merged_report_lines.append(f"# [블록 {idx}]")
            merged_report_lines.append(report)

        raw_list.append({
            "index": idx,
            "text": chunk,
            "raw": res.get("raw", {}),
            "score": score,
        })

    merged_report = "\n".join(merged_report_lines).strip()

    # 리포트가 하나도 없으면 score를 1로 통일
    if not merged_report:
        max_score = 1
    elif max_score <= 1:
        max_score = 3  # 뭔가 보고는 있는데 score가 1인 경우 기본 3으로 올리는 것도 가능

    # raw에는 chunk별 정보 전체를 묶어서 넣어둔다
    raw_bundle = {
        "mode": "chunked",
        "chunk_count": len(chunks),
        "chunks": raw_list,
        "suspicion_score": max_score,  # ✅ 추가
    }


    return {
        "score": max_score,
        "content_typo_report": "",              # 한국어 탭에서는 사용 안 하므로 비워둠
        "translated_typo_report": merged_report,
        "markdown_report": "",
        "raw": raw_bundle,
    }


# -------------------------------------------------
# 1-B. 영어 단일 텍스트 검수 프롬프트 + 래퍼
# -------------------------------------------------
def create_english_detector_prompt_for_text(english_text: str) -> str:
    """
    1차 패스: Detector
    - 가능한 많은 '잠재적 오류 후보'를 찾아내는 역할 (과검출 약간 허용)
    """
    safe_text = json.dumps(english_text, ensure_ascii=False)

    prompt = f"""
You are the first-pass **English text proofreader (Detector)**.

Your job is to detect **as many potential objective errors as possible** in the given English text.
You may slightly over-detect (allow some false positives), because a second-pass Judge will filter them.

Your response MUST be a single JSON object with EXACTLY these keys:
- "suspicion_score": integer 1~5
- "content_typo_report": string
- "translated_typo_report": ""   (keep empty, not used here)
- "markdown_report": ""          (keep empty)

Requirements for "content_typo_report":
- It MUST be a newline-joined list of bullet lines.
- Each line MUST follow this exact format (in Korean):

  - '원문' → '수정안': 오류 설명

- All explanations MUST be written in Korean.
- '원문' MUST be an exact substring of the original English text (after decoding).

The types of errors you should detect widely in this Detector pass:

- English spelling mistakes
- Split-word errors: "under stand" → "understand", "s imp le" → "simple"
- AI context "Al" (A + small L) that should be "AI" (artificial intelligence)
- Capitalization errors (sentence start, "i" instead of "I", proper nouns)
- Clear duplicate words ("the the")
- Obvious punctuation problems (missing final punctuation, ",." / ".." etc.)

------------------------------------------------------------
# Input: English text (JSON string)
------------------------------------------------------------

plain_english_json: {safe_text}

- Decode plain_english_json to obtain plain_english.
- In each bullet line "- '원문' → '수정안': 설명",
  '원문' MUST be a substring of plain_english.

Now, carefully detect as many *potential* objective errors as possible,
and output them in "content_typo_report" following the format above.
"""
    return prompt


def create_english_judge_prompt_for_text(english_text: str, draft_report: str) -> str:
    """
    2차 패스: Judge
    - Detector가 만든 후보들 중에서 '의미를 바꾸지 않는 객관적 오류'만 남기고 필터링
    """
    safe_text = json.dumps(english_text, ensure_ascii=False)
    safe_report = json.dumps(draft_report, ensure_ascii=False)

    prompt = f"""
You are the second-pass **English text proofreader (Judge)**.

Your role:
- Given the original English text and a candidate error list (draft_report),
  you MUST **keep only the lines that are objective, safe corrections**,
  and discard everything else.

------------------------------------------------------------
# Input 1: original English text (JSON string)
------------------------------------------------------------
plain_english_json: {safe_text}

- Decode this JSON string to get plain_english.

------------------------------------------------------------
# Input 2: Detector's candidate report (JSON string)
------------------------------------------------------------
draft_report_json: {safe_report}

- draft_report_json is a JSON string of the candidate report.
- When decoded, it is a multi-line string.
- Each line has the format:

  - '원문' → '수정안': 설명

------------------------------------------------------------
# Filtering Criteria (ALL must be satisfied to keep a line)
------------------------------------------------------------

1. '원문' MUST be an exact substring of plain_english.
2. '수정안' MUST represent an **objective, verifiable correction**, such as:
   - spelling / split-word correction
   - clear capitalization fix
   - obvious punctuation fix (missing final ., ?, !, duplicated punctuation, etc.)
3. You MUST REMOVE any line that:
   - rewrites the sentence for style or naturalness,
   - changes wording in a way that could change meaning,
   - adds or removes content beyond a minimal error fix,
   - is just a stylistic suggestion (better wording, tone, clarity, etc.).
4. If '원문' does not appear in plain_english at all, that line MUST be removed.

------------------------------------------------------------
# Output
------------------------------------------------------------

Return EXACTLY ONE JSON object with keys:
- "suspicion_score": integer 1~5 (based on remaining errors)
- "content_typo_report":
    a multi-line string containing ONLY the kept bullet lines
    in the same format "- '원문' → '수정안': 설명"
- "translated_typo_report": ""   (leave empty)
- "markdown_report": ""          (leave empty)

If no candidate lines satisfy all criteria, "content_typo_report" MUST be "".
All explanations MUST still be written in Korean.
"""
    return prompt



def create_english_review_prompt_for_text(english_text: str) -> str:
    # 영어 원문도 JSON 문자열로 안전하게 감싸기
    safe_text = json.dumps(english_text, ensure_ascii=False)

    
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

------------------------------------------------------------
# 1. RULES FOR ENGLISH OBJECTIVE ERRORS
------------------------------------------------------------

## (A) Split-Word Errors (항상 오타로 취급 — 매우 중요)
If an English word appears with an incorrect internal space,
AND removing the space yields a valid English word,
you MUST treat it as a spelling error.

## (B) Normal English spelling mistakes (MUST detect)
Any token similar to a valid English word (1–2 letters swapped/missing) MUST be flagged.

## (C) AI 문맥에서 "Al" → "AI" (항상 잡기)
If the surrounding sentence mentions:
model / system / tool / chatbot / LLM / agent / dataset / training / inference
then “Al” (A+소문자 l) MUST be interpreted as a typo for “AI”.

## (D) Capitalization Errors
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
- '."' / '!"' / '?"'
- ".’" / "!’" / "?’"

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


Below is the entire English text encoded as a JSON string.
You MUST decode this JSON string to obtain the original text,
and ONLY use that decoded text as the source for all 'original' spans.

plain_english_json: {safe_text}

- plain_english_json is a JSON-encoded string of the original English text.
- You MUST decode it and use the decoded text (plain_english) as the ONLY source.
- In "- '원문' → '수정안': ..." format, '원문' MUST be an exact substring of plain_english.

Now, following all the above rules, carefully proofread the text in plain_english_json.
"""
    return prompt


def review_english_text(english_text: str) -> Dict[str, Any]:
    """
    영어 텍스트 검수 (2-pass: Detector -> Judge)
    - 1차 Detector: 잠재적 오류 후보를 넓게 수집
    - 2차 Judge: 의미 변경/스타일 제안/환각 제거
    - + 규칙 기반 후처리 (drop_lines_not_in_source, ensure_english_final_punctuation)
    """
    # 1️⃣ 1차 패스: Detector
    detector_prompt = create_english_detector_prompt_for_text(english_text)
    detector_raw = analyze_text_with_gemini(detector_prompt)
    detector_clean = validate_and_clean_analysis(
        detector_raw,
        original_english_text=english_text,
    )

    draft_report = detector_clean.get("content_typo_report", "") or ""

    # 2️⃣ 2차 패스: Judge
    judge_prompt = create_english_judge_prompt_for_text(english_text, draft_report)
    judge_raw = analyze_text_with_gemini(judge_prompt)
    judge_clean = validate_and_clean_analysis(
        judge_raw,
        original_english_text=english_text,
    )

    score = judge_clean.get("suspicion_score", 1)
    try:
        score = int(score)
    except Exception:
        score = 3
    score = max(1, min(5, score))

    final_report = judge_clean.get("content_typo_report", "") or ""

    # 3️⃣ 규칙 기반 후처리 (영어용)
    #   - LLM이 혹시 잘못 인용한 라인 제거
    #   - 마지막 문장 종결부호 관련 요약 메시지 추가 (보수적으로)
    filtered = drop_lines_not_in_source(english_text, final_report)
    filtered = ensure_english_final_punctuation(english_text, filtered)
    filtered = drop_lines_not_in_source(english_text, filtered)  # 한 번 더 검증

    # 4️⃣ raw 번들 구성 (UI/디버그용)
    raw_bundle = {
        "mode": "two_pass_single_en",
        "suspicion_score": score,
        "content_typo_report": final_report,  # Judge 결과(룰 전)
        "detector_raw": detector_raw,
        "detector_clean": detector_clean,
        "judge_raw": judge_raw,
        "judge_clean": judge_clean,
        "initial_report_from_detector": draft_report,
        "final_report_before_rule_postprocess": final_report,
    }

    return {
        "score": score,
        "content_typo_report": filtered,  # 룰 후처리까지 끝난 최종 리포트
        "raw": raw_bundle,
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

tab_ko, tab_en, tab_pdf, tab_sheet, tab_about, tab_debug = st.tabs(
    ["✏️ 한국어 검수", "✏️ 영어 검수","📄 PDF 텍스트 정리", "📄 시트 검수", "ℹ️ 설명", "🐞 디버그"]
)

# --- 한국어 검수 탭 ---
# --- 한국어 검수 탭 ---
with tab_ko:
    st.subheader("한국어 텍스트 검수")
    default_ko = "이것은 테스트 문장 입니다, 그는.는 학교에 갔다,"
    text_ko = st.text_area("한국어 텍스트 입력", value=default_ko, height=220)

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

        # 최종 리포트
        final_report_ko = (result.get("translated_typo_report") or "").strip()

        # 1차 / 2차 / 최종 stage별 문자열 추출
        stage_reports_ko = get_korean_stage_reports(raw_json, final_report_ko)

        # 화면용 JSON (최종 기준)
        final_json_display = {
            "의심 점수": score,
            "한국어 검수_report": stage_reports_ko["final"],
        }
        raw_json_display = {
            "의심 점수": raw_json.get("suspicion_score"),
            "한국어 검수_report": stage_reports_ko["judge"],  # 2차 Judge 결과
        }

        st.success("한국어 검수가 완료되었습니다!")
        st.metric("의심 점수 (1~5) 1점 -> GOOD 5점 -> BAD", f"{float(score):.2f}")

        # ---------------- 하이라이트 카드 ----------------
        with st.container():
            st.markdown("### 🖍 오류 위치 · 하이라이트")

            stage_choice_ko = st.radio(
                "하이라이트 기준 선택",
                ["최종(Final)", "2차 Judge", "1차 Detector"],
                horizontal=True,
                key="ko_highlight_mode",
            )

            if stage_choice_ko == "최종(Final)":
                report_for_highlight = stage_reports_ko["final"]
                mode_label = "최종(Final) 기준"
            elif stage_choice_ko == "2차 Judge":
                report_for_highlight = stage_reports_ko["judge"]
                mode_label = "2차 Judge 기준"
            else:
                report_for_highlight = stage_reports_ko["detector"]
                mode_label = "1차 Detector 기준"

            spans_ko = parse_korean_report_with_positions(text_ko, report_for_highlight)

            default_punct_keys = list(PUNCT_GROUPS.keys())
            selected_punct_keys_ko = st.multiselect(
                "문장부호 선택",
                options=default_punct_keys,
                default=default_punct_keys,
                key="ko_punct_filter",
                help="선택한 부호만 색상 표시",
            )

            st.markdown(f"#### 🔦 {mode_label} 하이라이트")
            if spans_ko:
                for span in spans_ko:
                    if span["line"] is None:
                        st.markdown(
                            f"- `{span['original']}` → `{span['fixed']}`: {span['message']}"
                        )
                    else:
                        st.markdown(
                            f"- L{span['line']}, C{span['col']} — "
                            f"`{span['original']}` → `{span['fixed']}`: {span['message']}"
                        )
            else:
                st.info(f"{mode_label}으로 하이라이트할 항목이 없습니다. 원문을 그대로 표시합니다.")

            view_mode_ko = st.radio(
                "보기 모드",
                ["오류 하이라이트", "문장부호만"],
                horizontal=True,
                key="ko_view_mode_toggle",
            )

            selected_chars_ko = (
                set().union(*(PUNCT_GROUPS[k] for k in selected_punct_keys_ko))
                if selected_punct_keys_ko else set()
            )
            if view_mode_ko == "오류 하이라이트":
                highlighted_ko = highlight_text_with_spans(
                    text_ko,
                    spans_ko if spans_ko else [],
                    selected_punct_chars=selected_chars_ko,
                )
            else:
                highlighted_ko = highlight_selected_punctuation(text_ko, selected_punct_keys_ko)
            st.markdown(
                f"<div style='background:#f7f7f7; border:1px solid #e5e5e5; border-radius:8px; padding:12px;'>"
                f"<pre style='white-space: pre-wrap; background:transparent; margin:0; font-weight:600;'>{highlighted_ko}</pre>"
                f"</div>",
                unsafe_allow_html=True,
            )

            punct_counts_ko = Counter(ch for ch in text_ko if ch in PUNCT_COLOR_MAP)
            badge_order_ko = [
                (".", "종결부호"),
                ("?", "물음표"),
                ("!", "느낌표"),
                (",", "쉼표"),
                ('"', "쌍따옴표"),
                ("'", "작은따옴표"),
            ]
            badges_ko = []
            for ch, label in badge_order_ko:
                count = punct_counts_ko.get(ch, 0)
                color = PUNCT_COLOR_MAP.get(ch, "#e2e3e5")
                badges_ko.append(
                    f"<span style='background-color: {color}; padding: 2px 6px; border-radius: 4px; margin-right: 6px; display: inline-block;'>{label}: {count}</span>"
                )

            st.markdown(
                f"<div style='border: 1px solid #e9ecef; border-radius: 8px; padding: 10px; background: #f8f9fa; margin-bottom: 6px;'>{''.join(badges_ko)}</div>",
                unsafe_allow_html=True,
            )

            st.caption("※ 동일한 구절이 여러 번 등장하는 경우, 첫 번째 위치가 하이라이트될 수 있습니다.")
            st.markdown("""
                <small>
                <b>문장부호 색상 안내:</b><br>
                <span style='background-color: #fff3cd; padding: 0 3px;'>.</span> 종결부호 (., etc) &nbsp;
                <span style='background-color: #f8d7da; padding: 0 3px;'>?</span> 물음표 &nbsp;
                <span style='background-color: #f5c6cb; padding: 0 3px;'>!</span> 느낌표 &nbsp;
                <span style='background-color: #d1ecf1; padding: 0 3px;'>,</span> 쉼표 &nbsp;
                <span style='background-color: #e0f7e9; padding: 0 3px;'>&ldquo;</span> 쌍따옴표 &nbsp;
                <span style='background-color: #fce9d9; padding: 0 3px;'>&lsquo;</span> 작은따옴표 &nbsp;
                <span style='background-color: #d6d8d9; padding: 0 3px;'>; :</span> 기타 문장부호
                </small>
                """, unsafe_allow_html=True)

        # ---------------- 결과 비교 / 제안 사항 카드 ----------------
        with st.container():
            st.markdown("### 📊 결과 비교 · 제안")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### ✅ Final JSON (후처리 적용)")
                st.json(final_json_display, expanded=False)
            with col2:
                st.markdown("#### 🧪 Raw JSON (2차 Judge 기준)")
                st.json(raw_json_display, expanded=False)

            with st.expander("1차 Detector JSON (필요 시)", expanded=False):
                st.json(raw_json.get("detector_clean", {}))
            with st.expander("2차 Judge JSON (필요 시)", expanded=False):
                st.json(raw_json.get("judge_clean", {}))

            st.markdown("### 🛠 최종 수정 제안 사항 (최종 기준)")
            suggestions = extract_korean_suggestions_from_raw(
                {"translated_typo_report": stage_reports_ko["final"]}
            )
            if not suggestions:
                st.info("보고할 수정 사항이 없습니다.")
            else:
                for s in suggestions:
                    st.markdown(s)



# --- 영어 검수 탭 ---
with tab_en:
    st.subheader("영어 텍스트 검수")
    default_en = 'This is a simple understaning of the Al model.'
    text_en = st.text_area("English text input", value=default_en, height=220)

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

        # 최종 리포트
        final_report_en = (result.get("content_typo_report") or "").strip()
        stage_reports_en = get_english_stage_reports(raw_json, final_report_en)

        final_json = {
            "의심 점수": score,
            "영문 검수_report": stage_reports_en["final"],
        }
        raw_view = {
            "의심 점수": raw_json.get("suspicion_score"),
            "영문 검수_report": stage_reports_en["judge"],  # 2차 Judge
        }

        st.success("영어 검수가 완료되었습니다!")
        st.metric("의심 점수 (1~5) 1점 -> GOOD 5점 -> BAD", f"{float(score):.2f}")

        # ---------------- 하이라이트 카드 ----------------
        with st.container():
            st.markdown("### 🖍 오류 위치 · 하이라이트")

            view_mode_en = st.radio(
                "하이라이트 기준 선택",
                ["최종(Final)", "2차 Judge", "1차 Detector"],
                horizontal=True,
                key="en_highlight_mode",
            )

            if view_mode_en == "최종(Final)":
                report_for_highlight = stage_reports_en["final"]
                mode_label_en = "최종(Final) 기준"
            elif view_mode_en == "2차 Judge":
                report_for_highlight = stage_reports_en["judge"]
                mode_label_en = "2차 Judge 기준"
            else:
                report_for_highlight = stage_reports_en["detector"]
                mode_label_en = "1차 Detector 기준"

            spans_en = parse_english_report_with_positions(text_en, report_for_highlight)

            default_punct_keys = list(PUNCT_GROUPS.keys())
            selected_punct_keys_en = st.multiselect(
                "문장부호 선택",
                options=default_punct_keys,
                default=default_punct_keys,
                key="en_punct_filter",
                help="선택한 부호만 색상 표시",
            )

            st.markdown(f"#### 🔦 {mode_label_en} 하이라이트")
            if spans_en:
                for span in spans_en:
                    if span["line"] is None:
                        st.markdown(
                            f"- `{span['original']}` → `{span['fixed']}`: {span['message']}"
                        )
                    else:
                        st.markdown(
                            f"- L{span['line']}, C{span['col']} — "
                            f"`{span['original']}` → `{span['fixed']}`: {span['message']}"
                        )
            else:
                st.info(f"{mode_label_en}으로 하이라이트할 항목이 없습니다. 원문을 그대로 표시합니다.")

            selected_chars_en = (
                set().union(*(PUNCT_GROUPS[k] for k in selected_punct_keys_en))
                if selected_punct_keys_en else set()
            )
            view_mode_en_toggle = st.radio(
                "보기 모드",
                ["오류 하이라이트", "문장부호만"],
                horizontal=True,
                key="en_view_mode_toggle",
            )
            if view_mode_en_toggle == "오류 하이라이트":
                highlighted_en = highlight_text_with_spans(
                    text_en,
                    spans_en if spans_en else [],
                    selected_punct_chars=selected_chars_en,
                )
            else:
                highlighted_en = highlight_selected_punctuation(text_en, selected_punct_keys_en)
            st.markdown(
                f"<div style='background:#f7f7f7; border:1px solid #e5e5e5; border-radius:8px; padding:12px;'>"
                f"<pre style='white-space: pre-wrap; background:transparent; margin:0; font-weight:600;'>{highlighted_en}</pre>"
                f"</div>",
                unsafe_allow_html=True,
            )

            punct_counts_en = Counter(ch for ch in text_en if ch in PUNCT_COLOR_MAP)
            badge_order_en = [
                (".", "종결부호"),
                ("?", "물음표"),
                ("!", "느낌표"),
                (",", "쉼표"),
                ('"', "쌍따옴표"),
                ("'", "작은따옴표"),
            ]
            badges_en = []
            for ch, label in badge_order_en:
                count = punct_counts_en.get(ch, 0)
                color = PUNCT_COLOR_MAP.get(ch, "#e2e3e5")
                badges_en.append(
                    f"<span style='background-color: {color}; padding: 2px 6px; border-radius: 4px; margin-right: 6px; display: inline-block;'>{label}: {count}</span>"
                )

            st.markdown(
                f"<div style='border: 1px solid #e9ecef; border-radius: 8px; padding: 10px; background: #f8f9fa; margin-bottom: 6px;'>{''.join(badges_en)}</div>",
                unsafe_allow_html=True,
            )

            st.caption("※ 동일한 구절이 여러 번 등장하는 경우, 첫 번째 위치가 하이라이트될 수 있습니다.")
            st.markdown("""
                <small>
                <b>문장부호 색상 안내:</b><br>
                <span style='background-color: #fff3cd; padding: 0 3px;'>.</span> 종결부호 (., etc) &nbsp;
                <span style='background-color: #f8d7da; padding: 0 3px;'>?</span> 물음표 &nbsp;
                <span style='background-color: #f5c6cb; padding: 0 3px;'>!</span> 느낌표 &nbsp;
                <span style='background-color: #d1ecf1; padding: 0 3px;'>,</span> 쉼표 &nbsp;
                <span style='background-color: #e0f7e9; padding: 0 3px;'>&ldquo;</span> 쌍따옴표 &nbsp;
                <span style='background-color: #fce9d9; padding: 0 3px;'>&lsquo;</span> 작은따옴표 &nbsp;
                <span style='background-color: #d6d8d9; padding: 0 3px;'>; :</span> 기타 문장부호
                </small>
                """, unsafe_allow_html=True)

        # 결과 비교 / 제안 사항 카드
        with st.container():
            st.markdown("### 📊 결과 비교 · 제안")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### ✅ Final JSON (후처리 적용)")
                st.json(final_json, expanded=False)
            with col2:
                st.markdown("#### 🧪 Raw JSON (2차 Judge 기준)")
                st.json(raw_view, expanded=False)

            st.markdown("#### 🔍 Raw vs Final 차이 요약")
            diff_md_en = summarize_json_diff(raw_view, final_json)
            st.markdown(diff_md_en)

            st.markdown("### 🛠 최종 수정 제안 사항 (최종 기준)")
            suggestions_en = extract_english_suggestions_from_raw(
                {"content_typo_report": stage_reports_en["final"]}
            )
            if not suggestions_en:
                st.info("보고할 수정 사항이 없습니다.")
            else:
                for s in suggestions_en:
                    st.markdown(s)

            with st.expander("1차 Detector JSON (필요 시)", expanded=False):
                st.json(raw_json.get("detector_clean", {}))
            with st.expander("2차 Judge JSON (필요 시)", expanded=False):
                st.json(raw_json.get("judge_clean", {}))


# --- PDF 텍스트 정리 탭 ---
# --- PDF 텍스트 정리 탭 ---
with tab_pdf:
    st.subheader("📄 PDF에서 복사한 텍스트 정리")
    st.caption("PDF에서 복사한 텍스트를 붙여넣고 정리 + 첫 줄 삭제까지 할 수 있습니다.")

    pdf_raw_text = st.text_area(
        "PDF에서 복사한 원본 텍스트",
        height=300,
        key="pdf_input_text",
    )

    colA, colB = st.columns([1, 1])
    with colA:
        auto_trim_pdf = st.checkbox("앞뒤 공백 자동 제거", value=True, key="pdf_trim")

    with colB:
        run_pdf = st.button("텍스트 정리 실행", type="primary", key="pdf_run")

    if run_pdf:
        if not pdf_raw_text.strip():
            st.warning("먼저 텍스트를 입력해주세요.")
        else:
            text_to_send = pdf_raw_text.strip() if auto_trim_pdf else pdf_raw_text
            with st.spinner("Gemini가 텍스트를 정리하는 중입니다..."):
                cleaned_block = restore_pdf_text(text_to_send)
            # ✅ 정리된 결과를 세션에 저장
            st.session_state["pdf_cleaned"] = cleaned_block

    cleaned_block = st.session_state.get("pdf_cleaned")

    if cleaned_block:
        st.markdown("#### ✅ 정리된 텍스트")

        # 🔘 여기서 '맨 위 줄 지우기' 버튼
        if st.button("맨 위 줄만 지우기", key="pdf_delete_first_line"):
            st.session_state["pdf_cleaned"] = remove_first_line_in_code_block(cleaned_block)
            st.rerun()

        # 최신 상태 보여주기
        st.markdown(st.session_state["pdf_cleaned"])





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

    st.title("📘 텍스트 자동 검수기 설명서")
    st.caption("이 탭은 전체 앱의 구조와 동작 방식을 설명합니다.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "✨ 앱 소개",
        "✏️ 한국어 검수",
        "✏️ 영어 검수",
        "📄 시트 배치 검수",
        "🎯 철학 & 규칙"
    ])

    # -------------------------
    # 1) 앱 소개 탭
    # -------------------------
    with tab1:
        st.markdown("""
## ✨ 이 앱은 무엇을 하나요?

이 앱은 **한국어/영어 단일 텍스트 검수기**와  
**Google Sheets 기반 배치 검수기**를 포함한 **통합 자동 검수 플랫폼**입니다.

- 자연스러움, 문체, 표현 개선 등 **주관적 수정은 전혀 하지 않습니다.**  
- 오직 **객관적으로 검증 가능한 오류만** 검출합니다.  
- 모든 검수는 **JSON-only 응답 + 후처리 안정화 로직** 기반으로 작동하여  
  오탐(False Positive)과 누락을 최소화합니다.

---
""")

    # -------------------------
    # 2) 한국어 검수 탭
    # -------------------------
    with tab2:
        st.markdown("""
# ✏️ 한국어 검수 (Korean Proofreading)

## 🔍 기능 개요
한국어 텍스트에서 다음과 같은 **형식적·명백한 오류**만 검출합니다:

**검출하는 오류**
- 오탈자 / 반복 문자  
- 조사·어미 오류  
- 명백한 띄어쓰기 오류  
- 문장부호 오류  
  - 종결부호 누락  
  - 따옴표 짝 불일치  
  - 이상한 쉼표·마침표  
- (옵션) 단어 내부 분리 오류 (`된 다` → `된다`)

**검출하지 않는 항목**
- 자연스러운 표현 변경  
- 의미가 달라질 가능성이 있는 수정  
- 문장 재작성 수준의 교정  
- escape/markdown 기반 가짜 오류  

---

## 🧠 작동 방식

1. **한국어 전용 프롬프트 생성**  
   - "원문 의미 보존" 원칙을 강하게 명시  
   - 예시 토큰 출력 금지  
2. **Gemini(JSON mode, temperature=0)** 호출  
3. **후처리 단계**  
   - 스타일 제안 제거  
   - 존재하지 않는 '원문' 기반 수정 제거  
   - escape 기반 오류 제거  
   - 종결부호·따옴표 관련 오탐 제거  
   - plain / markdown 오류 분리  
4. **최종 출력**  
   - suspicion_score (1~5)  
   - translated_typo_report  
   - raw vs final JSON 비교 가능

---

## 🧪 2-패스 구조 (Detector → Judge)
- **1차 Detector**: 가능한 많은 오류 후보를 넓게 탐지 (약간 과검출 허용)
- **2차 Judge**: 의미 변경/스타일 제안/환각을 필터링해 **객관적 오류만 남김**
- UI에서 Detector/Judge/Final을 각각 선택해 하이라이트와 리포트를 비교할 수 있습니다.

---
""")

    # -------------------------
    # 3) 영어 검수 탭
    # -------------------------
    with tab3:
        st.markdown("""
# ✏️ 영어 검수 (English Proofreading)

## 🔍 기능 개요
영어 텍스트의 **객관적 오류만** 탐지합니다.

**검출하는 오류**
- 스펠링 오류  
- split-word 오류 (`wi th`, `o f` 등)  
- AI 문맥에서 `Al` → `AI` 오표기  
- 대문자 규칙 위반  
- 중복 단어  
- 종결부호 누락  

**검출하지 않는 항목**
- 스타일·표현 개선  
- 자연스러운 문장으로의 재작성  
- 마크다운/escape 기반 오류  

---

## 🧠 작동 방식

1. **영어 전용 프롬프트 생성**
2. **Gemini(JSON mode)** 호출  
3. **후처리**  
   - self-equal 라인 제거  
   - 원문 미존재 토큰 제거  
   - 가짜 종결부호 오류 제거  
   - 스타일 제안 차단  
4. plain / markdown 오류 분리

**출력 요소**
- suspicion_score  
- content_typo_report  
- raw JSON / final JSON / diff

---
""")

    # -------------------------
    # 4) 시트 배치 검수 탭
    # -------------------------
    with tab4:
        st.markdown("""
# 📄 Google Sheets 기반 배치 검수

## 📘 입력 스키마

| 컬럼명 | 설명 |
|-------|------|
| `content` | 영어 plain |
| `content_markdown` | 영어 markdown |
| `content_translated` | 한국어 plain |
| `content_markdown_translated` | 한국어 markdown |
| `STATUS` | `"1. AI검수요청"` 시 검수 대상 |

---

## 🧠 행 단위 처리 과정

### **1) 영어 검수**
- plain + markdown 결합  
- 모델 호출 → 후처리  
- plain / markdown 오류 분리  

### **2) 한국어 검수**
- plain + markdown 결합  
- escape 제거, 종결부호 확인 등 후처리  
- (옵션) 내부 분리 휴리스틱  
- plain / markdown 오류 분리  

### **3) 최종 결과 생성**
- suspicion_score = **max(영어 score, 한국어 score)**  
- content_typo_report  
- translated_typo_report  
- markdown_report  
- STATUS = `"2. AI검수완료"`

### **4) 시트 업데이트**
검수 결과가 자동으로 시트 각 행에 기록됩니다.

---

## 🛠 디버그 기능

검수 후 특정 행을 선택하면:

- 영어/한국어 원문  
- raw JSON  
- final JSON  
- plain vs markdown 분석  
- Raw vs Final Diff  

를 통해 **오류 검출 품질을 세밀하게 확인**할 수 있습니다.

---
""")

    # -------------------------
    # 5) 전체 철학 및 규칙 탭
    # -------------------------
    with tab5:
        st.markdown("""
# 🎯 전체 시스템 철학 및 규칙

## ✔ 의미 보존 원칙
모든 검수 로직은  
**“원문의 의미와 의도를 절대 바꾸지 않는다”**  
를 최우선 원칙으로 합니다.

---

## ✔ Hallucination 방지
- `'원문'`은 반드시 실제 텍스트에 존재해야 함  
- JSON-only 응답  
- 예시 토큰(AAA 등) 출력 금지  
- 스타일·문체 제안 전부 제거  

---

## ✔ 목표
- **객관적 오류만 정확하게 검출**  
- 후처리로 오탐 최소화  
- plain/markdown을 분리하여 출처를 명확하게 표현  

---
""")



# --- 디버그 탭 ---
with tab_debug:
    st.markdown("여기는 추후에 로그, 디버그용 정보를 추가로 표시할 수 있는 영역입니다.")
