# app.py
# -*- coding: utf-8 -*-
import json
import time
import re
import html
import io
import csv
import hashlib
import difflib
from collections import Counter
from typing import Dict, Any, List
from datetime import datetime, timezone
import uuid
import traceback

import gspread
from google.oauth2.service_account import Credentials


import streamlit as st
import google.generativeai as genai

# 로그 설정 (없으면 비활성)
LOG_SHEET_ID = st.secrets.get("LOG_SHEET_ID")
LOG_WORKSHEET_NAME = st.secrets.get("LOG_WORKSHEET", "usage_log_v2")
LOGGING_ENABLED = bool(LOG_SHEET_ID)
LOGGING_REASON = None if LOGGING_ENABLED else "LOG_SHEET_ID가 설정되어 있지 않아 로깅이 비활성화되었습니다."

from sheet_review import run_sheet_review
LOG_HEADERS = [
    "timestamp_utc",
    "session_id",
    "feature",
    "model",
    "status",
    "latency_ms",
    "prompt_tokens",
    "output_tokens",
    "total_tokens",
    "cost_usd",
    "error",
]

# --------------------------
# 0. Gemini 설정 (키는 secrets에서만 읽기)
# --------------------------
API_KEY = st.secrets.get("GEMINI_API_KEY")
if not API_KEY:
    st.error("GEMINI_API_KEY가 secrets에 설정되어 있지 않습니다.")
    st.stop()

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-001")

MODEL_NAME = "gemini-2.0-flash-001"

def log_event(row: dict):
    """
    Gemini 호출 1회에 대한 로그를 Google Sheets에 기록
    """
    ws = _get_log_worksheet()
    if ws is None:
        return  # 로깅 비활성 or 시트 접근 실패 시 조용히 무시

    now_utc = datetime.now(timezone.utc).isoformat()

    values = [
        now_utc,                              # timestamp_utc
        _get_session_id(),                    # session_id (없으면 생성)
        row.get("feature", ""),
        row.get("model", ""),
        row.get("status", ""),
        row.get("latency_ms", 0),
        row.get("prompt_tokens", 0),
        row.get("output_tokens", 0),
        row.get("total_tokens", 0),
        row.get("cost_usd", 0.0),
        row.get("error", ""),
    ]

    timestamp_col = _get_header_col_index(ws, "timestamp_utc") or 1
    target_row = _find_first_empty_row_in_col(ws, col=timestamp_col, start_row=2)
    start_cell = f"{_col_to_a1(timestamp_col)}{target_row}"
    end_cell = f"{_col_to_a1(timestamp_col + len(values) - 1)}{target_row}"
    ws.update(f"{start_cell}:{end_cell}", [values], value_input_option="RAW")


def _col_to_a1(col_index: int) -> str:
    """1-based column index to A1 letter (e.g., 1 -> A)."""
    result = ""
    while col_index:
        col_index, rem = divmod(col_index - 1, 26)
        result = chr(65 + rem) + result
    return result


def _is_blank_cell(value: str) -> bool:
    if value is None:
        return True
    text = str(value)
    text = text.replace("\u00a0", " ")
    return text.strip() == ""


def _find_first_empty_row_in_col(ws, col: int = 1, start_row: int = 2) -> int:
    """Find the first empty row in the given column, starting after headers."""
    col_letter = _col_to_a1(col)
    max_rows = ws.row_count
    values = ws.get(f"{col_letter}{start_row}:{col_letter}{max_rows}")

    for offset, row in enumerate(values):
        cell_value = row[0] if row else ""
        if _is_blank_cell(cell_value):
            return start_row + offset

    return start_row + len(values)


def _get_header_col_index(ws, header_name: str) -> int | None:
    headers = ws.row_values(1)
    target = header_name.strip().lower()
    for idx, header in enumerate(headers, start=1):
        if str(header).strip().lower() == target:
            return idx
    return None
    
def _get_worksheet_by_name(sheet_id: str, worksheet_name: str):
    """로그용 스프레드시트에서 특정 워크시트를 가져오거나 생성"""
    if not LOGGING_ENABLED:
        return None

    # 서비스계정은 dict 또는 JSON 문자열 두 형태 모두 지원
    if "GCP_SERVICE_ACCOUNT_JSON" in st.secrets:
        raw = st.secrets["GCP_SERVICE_ACCOUNT_JSON"]
        sa_info = raw if isinstance(raw, dict) else json.loads(raw)
    elif "gcp_service_account" in st.secrets:
        raw = st.secrets["gcp_service_account"]
        sa_info = raw if isinstance(raw, dict) else json.loads(raw)
    else:
        raise RuntimeError("GCP 서비스 계정 정보가 secrets에 없습니다.")

    creds = Credentials.from_service_account_info(
        sa_info,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)

    try:
        ws = sh.worksheet(worksheet_name)
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_name, rows=5000, cols=30)

    return ws


def _load_service_account_info() -> dict:
    """secrets에서 GCP 서비스 계정 정보를 읽는다."""
    if "GCP_SERVICE_ACCOUNT_JSON" in st.secrets:
        raw = st.secrets["GCP_SERVICE_ACCOUNT_JSON"]
        return raw if isinstance(raw, dict) else json.loads(raw)
    if "gcp_service_account" in st.secrets:
        raw = st.secrets["gcp_service_account"]
        return raw if isinstance(raw, dict) else json.loads(raw)
    raise RuntimeError("GCP 서비스 계정 정보가 secrets에 없습니다.")


@st.cache_resource
def _get_gspread_client():
    sa_info = _load_service_account_info()
    creds = Credentials.from_service_account_info(
        sa_info,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    return gspread.authorize(creds)


def _build_unique_headers(raw_headers: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    headers: list[str] = []
    for idx, header in enumerate(raw_headers, start=1):
        base = (header or "").strip() or f"column_{idx}"
        count = seen.get(base, 0) + 1
        seen[base] = count
        headers.append(base if count == 1 else f"{base}_{count}")
    return headers


@st.cache_data(ttl=120)
def _load_worksheet_records_by_name(spreadsheet_name: str, worksheet_name: str):
    gc = _get_gspread_client()
    worksheet = gc.open(spreadsheet_name).worksheet(worksheet_name)
    values = worksheet.get_all_values()

    if not values:
        return [], []

    headers = _build_unique_headers(values[0])
    records: list[dict] = []

    for row_idx, row in enumerate(values[1:], start=2):
        if not any(str(cell).strip() for cell in row):
            continue

        padded = row + [""] * max(0, len(headers) - len(row))
        records.append(
            {
                "sheet_row_index": row_idx,
                **{header: padded[i] if i < len(padded) else "" for i, header in enumerate(headers)},
            }
        )

    return headers, records


def _normalize_row_to_v2(header: list[str], row: list[str]) -> dict:
    """
    v1 시트에서 읽은 row(리스트)를 header에 맞춰 dict로 만든 뒤,
    v2(LOG_HEADERS) 스키마로 정규화한다.
    """
    # 1) v1 dict 만들기 (row 길이가 짧아도 안전하게)
    d = {h: (row[i] if i < len(row) else "") for i, h in enumerate(header)}

    # 2) v2 스키마로 변환
    feature = d.get("feature", "") or ""
    model = d.get("model", "") or d.get("MODEL_NAME", "") or ""

    status = d.get("status", "") or d.get("STATUS", "") or ""
    # v1에서 "OK"/"ERR" 같은 값이면 v2에 맞춰 소문자로 정규화
    if str(status).upper() == "OK":
        status = "ok"
    elif str(status).upper() in ["ERR", "ERROR", "FAIL"]:
        status = "error"

    latency_ms = d.get("latency_ms", "") or 0

    prompt_tokens = d.get("prompt_tokens", "") or 0
    output_tokens = d.get("output_tokens", "") or 0
    total_tokens = d.get("total_tokens", "") or 0

    # v1의 log_gemini_call 루트는 cost_usd 컬럼이 없고 마지막 칸이 error_msg였을 수 있음
    cost_usd = d.get("cost_usd", "")
    error = d.get("error", "")

    # cost_usd가 비어있으면 토큰으로 재계산 시도
    def to_int(x):
        try:
            return int(float(x))
        except Exception:
            return 0

    def to_float(x):
        try:
            return float(x)
        except Exception:
            return 0.0

    p = to_int(prompt_tokens)
    o = to_int(output_tokens)

    if (cost_usd is None) or (str(cost_usd).strip() == ""):
        # cost_usd가 없으면 재계산
        cost_usd = calc_gemini_flash_cost_usd(p, o)
    else:
        cost_usd = to_float(cost_usd)

    # total_tokens가 비어있으면 p+o
    t = to_int(total_tokens)
    if t <= 0:
        t = p + o

    # timestamp_utc가 없으면 timestamp 같은 이름을 찾아봄
    ts = d.get("timestamp_utc", "") or d.get("timestamp", "") or ""

    # session_id가 없으면 빈 값
    session_id = d.get("session_id", "") or d.get("sid", "") or ""

    return {
        "timestamp_utc": ts,
        "session_id": session_id,
        "feature": feature,
        "model": model,
        "status": status,
        "latency_ms": to_int(latency_ms),
        "prompt_tokens": p,
        "output_tokens": o,
        "total_tokens": t,
        "cost_usd": float(cost_usd),
        "error": error or "",
    }


def migrate_usage_log_to_v2(
    source_ws_name: str = "usage_log",
    target_ws_name: str = "usage_log_v2",
    batch_size: int = 300,
):
    """
    usage_log(v1) -> usage_log_v2(v2) 마이그레이션
    - v2 워크시트를 만들고 LOG_HEADERS로 헤더 고정
    - v1 데이터를 읽어서 v2 스키마로 정규화 후 append
    """
    if not LOG_SHEET_ID:
        raise RuntimeError("LOG_SHEET_ID가 secrets에 없습니다.")

    # source / target ws 열기
    src = _get_worksheet_by_name(LOG_SHEET_ID, source_ws_name)
    tgt = _get_worksheet_by_name(LOG_SHEET_ID, target_ws_name)

    # source 전체 값
    src_values = src.get_all_values()
    if not src_values or len(src_values) < 2:
        return {"migrated": 0, "skipped": 0, "reason": "source 시트에 데이터가 없습니다."}

    src_header = src_values[0]
    src_rows = src_values[1:]

    # target 헤더 세팅(강제)
    tgt_values = tgt.get_all_values()
    if not tgt_values:
        tgt.append_row(LOG_HEADERS, value_input_option="RAW")
    else:
        # 헤더가 다르면 1행을 v2 헤더로 덮어쓰기
        if tgt_values[0] != LOG_HEADERS:
            tgt.delete_rows(1)
            tgt.insert_row(LOG_HEADERS, index=1, value_input_option="RAW")

    # 중복 방지: target에 이미 있는 timestamp_utc + feature + latency_ms 조합을 set으로 만듦(간단키)
    # 데이터가 너무 많으면 비용이 커질 수 있으니, 필요하면 꺼도 됨.
    existing = set()
    tgt_all = tgt.get_all_values()
    if len(tgt_all) > 1:
        hdr = tgt_all[0]
        idx_ts = hdr.index("timestamp_utc")
        idx_ft = hdr.index("feature")
        idx_lt = hdr.index("latency_ms")
        for r in tgt_all[1:]:
            ts = r[idx_ts] if idx_ts < len(r) else ""
            ft = r[idx_ft] if idx_ft < len(r) else ""
            lt = r[idx_lt] if idx_lt < len(r) else ""
            existing.add((ts, ft, lt))

    migrated = 0
    skipped = 0

    # batch append 준비
    buffer = []

    for row in src_rows:
        norm = _normalize_row_to_v2(src_header, row)
        dedup_key = (norm["timestamp_utc"], norm["feature"], str(norm["latency_ms"]))

        if dedup_key in existing:
            skipped += 1
            continue

        buffer.append([
            norm["timestamp_utc"],
            norm["session_id"],
            norm["feature"],
            norm["model"],
            norm["status"],
            norm["latency_ms"],
            norm["prompt_tokens"],
            norm["output_tokens"],
            norm["total_tokens"],
            norm["cost_usd"],
            norm["error"],
        ])

        if len(buffer) >= batch_size:
            tgt.append_rows(buffer, value_input_option="RAW")
            migrated += len(buffer)
            buffer = []

    if buffer:
        tgt.append_rows(buffer, value_input_option="RAW")
        migrated += len(buffer)

    return {"migrated": migrated, "skipped": skipped, "target": target_ws_name}


def gemini_call(feature: str, prompt: str, generation_config: dict):
    t0 = time.time()
    try:
        response = model.generate_content(prompt, generation_config=generation_config)
        latency_ms = int((time.time() - t0) * 1000)

        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        total_tokens = int(getattr(usage, "total_token_count", prompt_tokens + output_tokens) or (prompt_tokens + output_tokens))

        cost_usd = calc_gemini_flash_cost_usd(prompt_tokens, output_tokens)

        log_event({
            "feature": feature,
            "model": "gemini-2.0-flash-001",
            "status": "ok",
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
            "error": "",
        })

        return response

    except Exception as e:
        latency_ms = int((time.time() - t0) * 1000)
        log_event({
            "feature": feature,
            "model": "gemini-2.0-flash-001",
            "status": "error",
            "latency_ms": latency_ms,
            "prompt_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "error": str(e),
        })
        raise


def _get_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]

def calc_gemini_flash_cost_usd(prompt_tokens: int, output_tokens: int) -> float:
    # Gemini 2.0 Flash (Standard) - text pricing
    in_cost_per_1m = 0.10
    out_cost_per_1m = 0.40
    return (prompt_tokens / 1_000_000) * in_cost_per_1m + (output_tokens / 1_000_000) * out_cost_per_1m


@st.cache_resource
def _get_log_worksheet():
    if not LOGGING_ENABLED:
        return None

    try:
        # 서비스계정은 dict 또는 JSON 문자열 두 형태 모두 지원
        if "GCP_SERVICE_ACCOUNT_JSON" in st.secrets:
            raw = st.secrets["GCP_SERVICE_ACCOUNT_JSON"]
            sa_info = raw if isinstance(raw, dict) else json.loads(raw)
        elif "gcp_service_account" in st.secrets:
            raw = st.secrets["gcp_service_account"]
            sa_info = raw if isinstance(raw, dict) else json.loads(raw)
        else:
            st.warning("LOGGING_ENABLED 이지만 GCP 서비스 계정 정보가 없습니다. 로깅이 비활성화됩니다.")
            return None

        creds = Credentials.from_service_account_info(
            sa_info,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ],
        )
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(LOG_SHEET_ID)

        # 1) 워크시트 가져오기 (없으면 생성)
        try:
            ws = sh.worksheet(LOG_WORKSHEET_NAME)
        except gspread.exceptions.WorksheetNotFound:
            ws = sh.add_worksheet(title=LOG_WORKSHEET_NAME, rows=2000, cols=20)

        # 2) 시트가 완전 비어있으면 헤더 생성
        #    (이미 뭔가 있으면 건드리지 않음)
        if len(ws.get_all_values()) == 0:
            ws.append_row(LOG_HEADERS)

        return ws

    except Exception as e:
        # 최소한 원인 추적 가능하게 로그 남기기
        st.error(f"[LOG] worksheet init failed: {e}")
        return None
def _extract_token_usage(response) -> dict:
    """
    SDK/버전별로 usage 메타가 없을 수도 있으니 최대한 방어적으로 추출.
    (일반적으로 usage_metadata에 prompt/candidates/total token count가 들어옵니다.) :contentReference[oaicite:1]{index=1}
    """
    usage = {"prompt_tokens": None, "output_tokens": None, "total_tokens": None}

    um = getattr(response, "usage_metadata", None)
    if um is None:
        return usage

    # 객체/딕트 모두 대응
    getter = (lambda k: um.get(k)) if isinstance(um, dict) else (lambda k: getattr(um, k, None))

    usage["prompt_tokens"] = getter("prompt_token_count")
    usage["output_tokens"] = getter("candidates_token_count")
    usage["total_tokens"]  = getter("total_token_count")
    return usage

def log_gemini_call(feature: str, response=None, latency_ms: int | None = None, ok: bool = True, error_msg: str = ""):
    if not LOGGING_ENABLED:
        if LOGGING_REASON:
            st.info(f"[로그 비활성화] {LOGGING_REASON}")
        return
    try:
        ws = _get_log_worksheet()
        if ws is None:
            return
        ts = datetime.now(timezone.utc).isoformat()
        sid = _get_session_id()

        usage = _extract_token_usage(response) if response is not None else {}
        row = [
            ts,                 # timestamp (UTC)
            sid,                # session_id (익명)
            feature,            # 예: ko_detector / en_judge / pdf_restore ...
            MODEL_NAME,
            "OK" if ok else "ERR",
            latency_ms,
            usage.get("prompt_tokens"),
            usage.get("output_tokens"),
            usage.get("total_tokens"),
            (error_msg or "")[:500],
        ]
        ws.append_row(row, value_input_option="USER_ENTERED")
    except Exception as e:
        # 로그 실패가 앱 동작을 막으면 안 되니 조용히 넘김 (원하면 st.warning으로 바꿔도 됨)
        print(f"[LOGGING FAILED] {e}")

def generate_content_logged(feature: str, prompt: str, generation_config: dict):
    t0 = time.time()
    try:
        resp = model.generate_content(prompt, generation_config=generation_config)
        latency_ms = int((time.time() - t0) * 1000)
        log_gemini_call(feature, response=resp, latency_ms=latency_ms, ok=True)
        return resp
    except Exception as e:
        latency_ms = int((time.time() - t0) * 1000)
        log_gemini_call(feature, response=None, latency_ms=latency_ms, ok=False, error_msg=str(e))
        raise


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


※ 정답/오답 분리 규칙 (매우 중요)

- 다음과 같은 패턴이 하나의 문단에 함께 나타나는 경우,
  반드시 정답과 오답을 분리하여 출력해야 한다.

  예:
  "정답 ①: ... ②③④⑤는 ..."
  "정답: ① ... ②, ③, ④, ⑤는 ..."
  "①은 ..., 나머지는 ..."

- 처리 규칙:
  1) "정답 ①" 또는 "정답: ①"이 발견되면
     → "정답 ①"을 단독 한 줄로 분리한다.

  2) 정답 번호 바로 뒤에 오는 설명 문장은
     반드시 [정답 해설] 아래에 배치한다.

  3) 같은 문단에서 다음 표현이 발견되면:
     - "②③④⑤는"
     - "②, ③, ④, ⑤는"
     - "나머지는"
     - "기타 보기는"
     이는 모두 오답 설명으로 간주한다.

  4) 오답 설명은 반드시 [오답 해설] 헤더 아래로 이동시킨다.

  5) 오답 번호는 "②, ③, ④, ⑤"처럼 쉼표로 구분된 형태로 통일한다.

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
    response = gemini_call(feature="ui.pdf_restore.single", prompt=prompt, generation_config={"temperature": 0.0})
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




def analyze_text_with_gemini(prompt: str, feature: str, max_retries: int = 5) -> dict:

    """
    단일 텍스트 검사용 Gemini 호출.
    항상 dict를 리턴하도록 방어 로직을 넣음.
    """
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            response = gemini_call(
            feature=feature,
            prompt=prompt,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.0,
            },
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


def _review_korean_single_block(korean_text: str, block_id: int | None = None) -> Dict[str, Any]:
    det_feature = f"ui.ko_proof.detector.block_{block_id}" if block_id else "ui.ko_proof.detector.single"
    jud_feature = f"ui.ko_proof.judge.block_{block_id}"    if block_id else "ui.ko_proof.judge.single"

    # 1️⃣ 1차 패스: Detector
    detector_prompt = create_korean_detector_prompt_for_text(korean_text)
    detector_raw = analyze_text_with_gemini(
        detector_prompt,
        feature=det_feature,
    )
    detector_clean = validate_and_clean_analysis(detector_raw)

    draft_report = detector_clean.get("translated_typo_report", "") or ""

    # 2️⃣ 2차 패스: Judge
    judge_prompt = create_korean_judge_prompt_for_text(korean_text, draft_report)
    judge_raw = analyze_text_with_gemini(
        judge_prompt,
        feature=jud_feature,
    )
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
        res = _review_korean_single_block(chunk, block_id=idx)


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
    detector_raw = analyze_text_with_gemini(detector_prompt, feature="ui.en_proof.detector.single")
    detector_clean = validate_and_clean_analysis(
        detector_raw,
        original_english_text=english_text,
    )

    draft_report = detector_clean.get("content_typo_report", "") or ""

    # 2️⃣ 2차 패스: Judge
    judge_prompt = create_english_judge_prompt_for_text(english_text, draft_report)
    judge_raw    = analyze_text_with_gemini(judge_prompt,    feature="ui.en_proof.judge.single")
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


def _json_diff_keys(raw: dict | None, final: dict | None) -> list[str]:
    if not isinstance(raw, dict):
        raw = {}
    if not isinstance(final, dict):
        final = {}
    keys = []
    for key in sorted(set(raw.keys()) | set(final.keys())):
        if raw.get(key, "<없음>") != final.get(key, "<없음>"):
            keys.append(key)
    return keys


def format_json_diff_html(raw: dict | None, final: dict | None) -> str:
    if not isinstance(raw, dict):
        raw = {}
    if not isinstance(final, dict):
        final = {}

    rows = []
    for key in sorted(set(raw.keys()) | set(final.keys())):
        rv = raw.get(key, "<없음>")
        fv = final.get(key, "<없음>")
        if rv == fv:
            continue

        rv_str = json.dumps(rv, ensure_ascii=False) if isinstance(rv, (dict, list)) else str(rv)
        fv_str = json.dumps(fv, ensure_ascii=False) if isinstance(fv, (dict, list)) else str(fv)

        rows.append(
            "<div class='diff-item'>"
            f"<div class='diff-key'>{html.escape(str(key))}</div>"
            f"<div class='diff-raw'>raw: {html.escape(rv_str)}</div>"
            f"<div class='diff-final'>final: {html.escape(fv_str)}</div>"
            "</div>"
        )

    if not rows:
        return "<div class='diff-empty'>차이가 없습니다. (raw와 final이 동일합니다.)</div>"

    return "".join(rows)


def _extract_report_targets(report: str) -> list[str]:
    if not report:
        return []

    targets = []
    patterns = [
        re.compile(r"^- '(.+?)' → '(.+?)':", re.UNICODE),
        re.compile(r'^- "(.+?)" → "(.+?)":', re.UNICODE),
    ]

    for line in report.splitlines():
        s = line.strip()
        if not s:
            continue
        for pattern in patterns:
            m = pattern.match(s)
            if m:
                original = m.group(1)
                if original:
                    targets.append(original)
                break

    # 중복 제거 (긴 표현 우선)
    unique = sorted(set(targets), key=len, reverse=True)
    return unique


def _find_text_spans(text: str, targets: list[str]) -> list[tuple[int, int]]:
    spans = []
    for target in targets:
        start = 0
        while True:
            idx = text.find(target, start)
            if idx == -1:
                break
            spans.append((idx, idx + len(target)))
            start = idx + len(target)
    return spans


def highlight_text_with_reports(text: str, raw_report: str, final_report: str) -> str:
    if not text:
        return ""

    raw_targets = _extract_report_targets(raw_report)
    final_targets = _extract_report_targets(final_report)
    if not raw_targets and not final_targets:
        return html.escape(text)

    raw_spans = _find_text_spans(text, raw_targets)
    final_spans = _find_text_spans(text, final_targets)
    if not raw_spans and not final_spans:
        return html.escape(text)

    events = {}
    for s, e in raw_spans:
        events.setdefault(s, [0, 0])[0] += 1
        events.setdefault(e, [0, 0])[0] -= 1
    for s, e in final_spans:
        events.setdefault(s, [0, 0])[1] += 1
        events.setdefault(e, [0, 0])[1] -= 1

    positions = sorted(events.keys())
    if not positions:
        return html.escape(text)

    parts = []
    last = 0
    raw_active = 0
    final_active = 0

    for pos in positions:
        if pos > last:
            segment = text[last:pos]
            escaped = html.escape(segment)
            if raw_active or final_active:
                if raw_active and final_active:
                    cls = "text-highlight text-highlight-both"
                elif raw_active:
                    cls = "text-highlight text-highlight-raw"
                else:
                    cls = "text-highlight text-highlight-final"
                parts.append(f"<mark class='{cls}'>{escaped}</mark>")
            else:
                parts.append(escaped)

        raw_active += events[pos][0]
        final_active += events[pos][1]
        last = pos

    if last < len(text):
        segment = text[last:]
        escaped = html.escape(segment)
        if raw_active or final_active:
            if raw_active and final_active:
                cls = "text-highlight text-highlight-both"
            elif raw_active:
                cls = "text-highlight text-highlight-raw"
            else:
                cls = "text-highlight text-highlight-final"
            parts.append(f"<mark class='{cls}'>{escaped}</mark>")
        else:
            parts.append(escaped)

    return "".join(parts)


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


def _pick_default_english_column(headers: list[str]) -> str:
    if not headers:
        return ""
    preferred = [
        "content",
        "english",
        "english_text",
        "passage",
        "english_passage",
        "지문",
        "영어지문",
    ]
    headers_lower = {h.lower(): h for h in headers}
    for key in preferred:
        if key in headers_lower:
            return headers_lower[key]
    return headers[0]


def _rows_to_csv_bytes(rows: list[dict], columns: list[str]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({col: row.get(col, "") for col in columns})
    return buffer.getvalue().encode("utf-8-sig")


def _normalize_for_dedupe(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _hash_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _make_passage_dedupe_keys(row: dict) -> set[str]:
    keys: set[str] = set()

    passage_id = _normalize_for_dedupe(row.get("passage_id", ""))
    source_id = _normalize_for_dedupe(row.get("source_id", ""))
    passage_title = _normalize_for_dedupe(row.get("passage_title", ""))
    content = _normalize_for_dedupe(row.get("content", ""))

    if passage_id:
        keys.add(f"pid:{passage_id}")
    if source_id and passage_title:
        keys.add(f"sid_title:{source_id}|{passage_title}")
    if content:
        keys.add(f"content:{_hash_text(content)}")
    if passage_title and content:
        keys.add(f"title_content:{passage_title}|{_hash_text(content)}")

    return keys


def _build_full_text_diff_html(left_text: str, right_text: str) -> tuple[str, str]:
    left = str(left_text or "")
    right = str(right_text or "")
    matcher = difflib.SequenceMatcher(None, left, right)

    left_parts: list[str] = []
    right_parts: list[str] = []

    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        left_seg = html.escape(left[i1:i2])
        right_seg = html.escape(right[j1:j2])

        if op == "equal":
            left_parts.append(left_seg)
            right_parts.append(right_seg)
        else:
            if left_seg:
                left_parts.append(
                    f"<span style='background:#fecaca; color:#7f1d1d; font-weight:700;'>{left_seg}</span>"
                )
            if right_seg:
                right_parts.append(
                    f"<span style='background:#fecaca; color:#7f1d1d; font-weight:700;'>{right_seg}</span>"
                )

    return "".join(left_parts), "".join(right_parts)


MOCK_CSV_COLUMNS = [
    "content",
    "content_markdown",
    "content_translated",
    "content_markdown_translated",
    "footnote",
    "passage_id",
    "source_id",
    "book_title",
    "unit_title",
]

TEMP_SAVE_COLUMNS = [
    "slot_label",
    "null_locked",
    "passage_title",
    *MOCK_CSV_COLUMNS,
]


def _is_null_locked(row: dict) -> bool:
    v = row.get("null_locked", "")
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _slot_has_payload(row: dict) -> bool:
    check_cols = ["content", "passage_id", "source_id", "book_title", "unit_title", "passage_title"]
    return any(str(row.get(col, "")).strip() for col in check_cols)


@st.cache_data(ttl=120)
def _list_worksheet_titles(spreadsheet_name: str) -> list[str]:
    gc = _get_gspread_client()
    sh = gc.open(spreadsheet_name)
    return [ws.title for ws in sh.worksheets()]


def _save_rows_to_worksheet(spreadsheet_name: str, worksheet_name: str, rows: list[dict], columns: list[str]) -> None:
    gc = _get_gspread_client()
    sh = gc.open(spreadsheet_name)

    try:
        ws = sh.worksheet(worksheet_name)
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(
            title=worksheet_name,
            rows=max(1000, len(rows) + 20),
            cols=max(20, len(columns) + 2),
        )

    matrix = [columns]
    for row in rows:
        matrix.append([row.get(col, "") for col in columns])

    needed_rows = max(1000, len(matrix) + 20)
    needed_cols = max(20, len(columns) + 2)
    if ws.row_count < needed_rows or ws.col_count < needed_cols:
        ws.resize(rows=max(ws.row_count, needed_rows), cols=max(ws.col_count, needed_cols))

    ws.clear()
    end_cell = f"{_col_to_a1(len(columns))}{len(matrix)}"
    ws.update(f"A1:{end_cell}", matrix, value_input_option="RAW")
    # 저장 직후 목록/불러오기 캐시가 오래된 값을 보여주지 않도록 무효화
    _list_worksheet_titles.clear()
    _load_rows_from_worksheet.clear()
    _load_worksheet_records_by_name.clear()


@st.cache_data(ttl=120)
def _load_rows_from_worksheet(spreadsheet_name: str, worksheet_name: str) -> list[dict]:
    gc = _get_gspread_client()
    sh = gc.open(spreadsheet_name)
    ws = sh.worksheet(worksheet_name)
    values = ws.get_all_values()
    if not values:
        return []

    headers = _build_unique_headers(values[0])
    loaded: list[dict] = []
    for row in values[1:]:
        padded = row + [""] * max(0, len(headers) - len(row))
        item = {h: padded[i] if i < len(padded) else "" for i, h in enumerate(headers)}
        # 완전 빈 행은 제외
        if any(str(item.get(col, "")).strip() for col in headers):
            loaded.append(item)
    return loaded


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

tab_ko, tab_en, tab_pdf, tab_sheet, tab_passage, tab_mock_csv, tab_about, tab_debug = st.tabs(
    ["✏️ 한국어 검수", "✏️ 영어 검수","📄 PDF 텍스트 정리", "📄 시트 검수", "📚 영어 지문 조회", "📥 모의고사 CSV", "ℹ️ 설명", "🐞 디버그"]
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
            st.markdown(
                """
                <style>
                .diff-item { border: 1px solid #e5e7eb; border-radius: 10px; padding: 10px 12px; margin: 10px 0; background: #f8fafc; }
                .diff-key { font-weight: 700; margin-bottom: 6px; color: #0f172a; }
                .diff-raw { background: #fff1f2; padding: 6px 8px; border-radius: 6px; margin-bottom: 6px; color: #7f1d1d; }
                .diff-final { background: #ecfdf3; padding: 6px 8px; border-radius: 6px; color: #14532d; }
                .diff-empty { color: #475569; }
                .text-block { background: #f8fafc; color: #0f172a; border-radius: 10px; padding: 12px 14px; overflow-y: auto; white-space: pre-wrap; line-height: 1.6; border: 1px solid #e5e7eb; }
                .text-block-markdown { background: #ffffff; color: #0f172a; border-radius: 10px; padding: 12px 14px; overflow-y: auto; white-space: pre-wrap; line-height: 1.6; border: 1px solid #e2e8f0; }
                .text-highlight { padding: 1px 2px; border-radius: 3px; }
                .text-highlight-raw { background: #fecaca; color: #7f1d1d; }
                .text-highlight-final { background: #bbf7d0; color: #14532d; }
                .text-highlight-both { background: #fde68a; color: #92400e; }
                </style>
                """,
                unsafe_allow_html=True,
            )

            diff_meta = {}
            for item in raw_results:
                row_index = item.get("sheet_row_index")
                en = item.get("english", {}) or {}
                ko = item.get("korean", {}) or {}
                en_diff = _json_diff_keys(en.get("raw"), en.get("final"))
                ko_diff = _json_diff_keys(ko.get("raw"), ko.get("final"))
                diff_meta[row_index] = {
                    "total": len(en_diff) + len(ko_diff),
                    "en": len(en_diff),
                    "ko": len(ko_diff),
                }

            only_diff_rows = st.checkbox("Raw/Final 차이 있는 행만 보기", value=True)
            row_numbers = [item["sheet_row_index"] for item in raw_results]
            if only_diff_rows:
                row_numbers = [r for r in row_numbers if diff_meta.get(r, {}).get("total", 0) > 0]
            if not row_numbers:
                st.info("차이가 있는 행이 없습니다. 전체 행으로 표시합니다.")
                row_numbers = [item["sheet_row_index"] for item in raw_results]

            diff_rows_count = sum(1 for r in diff_meta if diff_meta[r]["total"] > 0)
            st.caption(f"raw/final 차이 있는 행: {diff_rows_count} / {len(raw_results)}")

            text_block_height = st.slider(
                "원문 텍스트 영역 높이(px)",
                min_value=200,
                max_value=800,
                value=360,
                step=20,
                key="sheet_text_height",
            )
            markdown_block_height = st.slider(
                "마크다운 텍스트 영역 높이(px)",
                min_value=200,
                max_value=800,
                value=300,
                step=20,
                key="sheet_markdown_height",
            )

            selected_candidate = st.selectbox(
                "Raw/Final JSON을 보고 싶은 행 번호를 선택하세요:",
                options=row_numbers,
                format_func=lambda x: (
                    f"행 {x}번"
                    f" (diff {diff_meta.get(x, {}).get('total', 0)}"
                    f" | EN {diff_meta.get(x, {}).get('en', 0)}"
                    f" | KO {diff_meta.get(x, {}).get('ko', 0)})"
                ),
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
                        highlighted = highlight_text_with_reports(
                            bundle.get("text_plain", "") or "",
                            bundle.get("raw_report_plain", "") or "",
                            bundle.get("report_plain", "") or "",
                        )
                        st.markdown(
                            f"<div class='text-block' style='height: {text_block_height}px'>{highlighted}</div>",
                            unsafe_allow_html=True,
                        )

                        st.markdown("##### 📝 영어 마크다운 텍스트 (content_markdown)")
                        st.markdown(
                            (
                                "<div class='text-block-markdown' "
                                f"style='height: {markdown_block_height}px'>"
                                f"{highlight_text_with_reports(bundle.get('text_markdown', '') or '', bundle.get('raw_report_markdown', '') or '', bundle.get('report_markdown', '') or '')}"
                                "</div>"
                            ),
                            unsafe_allow_html=True,
                        )

                        st.markdown("##### ⚡ Raw vs Final 차이점 (필터링 확인)")
                        diff_html = format_json_diff_html(raw_json, final_json)
                        st.markdown(diff_html, unsafe_allow_html=True)

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
                        highlighted = highlight_text_with_reports(
                            bundle.get("text_plain", "") or "",
                            bundle.get("raw_report_plain", "") or "",
                            bundle.get("report_plain", "") or "",
                        )
                        st.markdown(
                            f"<div class='text-block' style='height: {text_block_height}px'>{highlighted}</div>",
                            unsafe_allow_html=True,
                        )

                        st.markdown("##### 📝 한국어 마크다운 텍스트 (content_markdown_translated)")
                        st.markdown(
                            (
                                "<div class='text-block-markdown' "
                                f"style='height: {markdown_block_height}px'>"
                                f"{highlight_text_with_reports(bundle.get('text_markdown', '') or '', bundle.get('raw_report_markdown', '') or '', bundle.get('report_markdown', '') or '')}"
                                "</div>"
                            ),
                            unsafe_allow_html=True,
                        )

                        st.markdown("##### ⚡ Raw vs Final 차이점 (필터링 확인)")
                        diff_html = format_json_diff_html(raw_json, final_json)
                        st.markdown(diff_html, unsafe_allow_html=True)

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
                            st.markdown(
                                (
                                    "<div class='text-block-markdown' "
                                    f"style='height: {markdown_block_height}px'>"
                                    f"{html.escape(en_md)}"
                                    "</div>"
                                ),
                                unsafe_allow_html=True,
                            )
                        else:
                            st.info("영어 마크다운 텍스트가 비어 있습니다.")

                        st.markdown("##### 📄 한국어 마크다운 원문 (content_markdown_translated)")
                        if ko_md.strip():
                            st.markdown(
                                (
                                    "<div class='text-block-markdown' "
                                    f"style='height: {markdown_block_height}px'>"
                                    f"{html.escape(ko_md)}"
                                    "</div>"
                                ),
                                unsafe_allow_html=True,
                            )
                        else:
                            st.info("한국어 마크다운 텍스트가 비어 있습니다.")

                        st.markdown("##### 🧷 MARKDOWN_REPORT (두 언어 마크다운 오류 통합)")
                        if markdown_report.strip():
                            st.markdown(markdown_report)
                        else:
                            st.info("마크다운 관련으로 보고된 오류가 없습니다.")
                else:
                    st.warning("선택한 행의 데이터를 찾을 수 없습니다.")



# --- 영어 지문 조회 탭 ---
with tab_passage:
    st.subheader("📚 영어 지문 시트 조회")
    st.caption("Google Sheets의 영어 지문 시트를 연결하고 키워드/행 번호로 조회합니다.")

    passage_sheet_name = st.secrets.get("PASSAGE_SHEET_NAME", "[DATA] Paragraph DB (영어 통합)")
    worksheet_options = ["교과서", "참고서", "모의고사"]
    default_passage_ws = st.secrets.get("PASSAGE_WORKSHEET_NAME", "교과서")
    default_ws_index = worksheet_options.index(default_passage_ws) if default_passage_ws in worksheet_options else 0
    requested_cols = [
        "passage_id",
        "passage_title",
        "source_id",
        "studio_title",
        "unit_order",
        "unit_title",
        "content",
        "content_markdown",
        "content_translated",
        "content_markdown_translated",
    ]

    col_p1, col_p2, col_p3 = st.columns([2, 2, 1])
    with col_p1:
        st.text_input("스프레드시트 이름", value=passage_sheet_name, key="passage_sheet_name_fixed", disabled=True)
    with col_p2:
        passage_ws_name = st.selectbox(
            "워크시트 이름",
            options=worksheet_options,
            index=default_ws_index,
            key="passage_ws_name",
        )
    with col_p3:
        st.write("")
        st.write("")
        if st.button("새로고침", key="passage_reload"):
            _load_worksheet_records_by_name.clear()
    try:
        headers, records = _load_worksheet_records_by_name(
            passage_sheet_name,
            passage_ws_name,
        )
    except Exception as e:
        headers, records = [], []
        st.error(f"시트를 불러오지 못했습니다: {e}")

    if headers:
        st.success(f"연결 완료: {passage_sheet_name} / {passage_ws_name}")
        available_cols = [c for c in requested_cols if c in headers]
        missing_cols = [c for c in requested_cols if c not in headers]
        if missing_cols:
            st.warning(f"시트에 없는 컬럼: {', '.join(missing_cols)}")
        if not available_cols:
            st.error("요청하신 컬럼이 시트에 없습니다. 헤더를 확인해주세요.")
            available_cols = headers

        if "passage_search_params" not in st.session_state:
            st.session_state["passage_search_params"] = {
                "q_content": "",
                "q_studio": "",
                "q_unit": "",
                "q_passage_title": "",
                "row_query": "",
                "only_non_empty": True,
            }
        params = st.session_state.get("passage_search_params", {})
        with st.form("passage_search_form", clear_on_submit=False):
            col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns([2, 2, 2, 2, 1])
            with col_f1:
                q_content_input = st.text_input(
                    "content 검색",
                    value=params.get("q_content", ""),
                    key="passage_q_content",
                    placeholder="예: climate",
                )
            with col_f2:
                q_studio_input = st.text_input(
                    "studio_title 검색",
                    value=params.get("q_studio", ""),
                    key="passage_q_studio",
                    placeholder="예: ebs",
                )
            with col_f3:
                q_unit_input = st.text_input(
                    "unit_title 검색",
                    value=params.get("q_unit", ""),
                    key="passage_q_unit",
                    placeholder="예: environment",
                )
            with col_f4:
                q_passage_title_input = st.text_input(
                    "passage_title 검색",
                    value=params.get("q_passage_title", ""),
                    key="passage_q_passage_title",
                    placeholder="예: The Future of AI",
                )
            with col_f5:
                row_query_input = st.text_input(
                    "행 번호 조회(쉼표 구분)",
                    value=params.get("row_query", ""),
                    key="passage_row_query",
                    placeholder="예: 12, 35, 104",
                )
                only_non_empty_input = st.checkbox(
                    "빈 지문 제외",
                    value=bool(params.get("only_non_empty", True)),
                    key="passage_non_empty",
                )
            submitted = st.form_submit_button("검색 실행", type="primary")

        col_reset_l, _ = st.columns([1, 4])
        with col_reset_l:
            if st.button("검색 초기화", key="passage_search_reset"):
                st.session_state["passage_search_params"] = {
                    "q_content": "",
                    "q_studio": "",
                    "q_unit": "",
                    "q_passage_title": "",
                    "row_query": "",
                    "only_non_empty": True,
                }
                st.session_state["passage_single_selected_row"] = None

        if submitted:
            st.session_state["passage_search_params"] = {
                "q_content": q_content_input.strip().lower(),
                "q_studio": q_studio_input.strip().lower(),
                "q_unit": q_unit_input.strip().lower(),
                "q_passage_title": q_passage_title_input.strip().lower(),
                "row_query": row_query_input.strip(),
                "only_non_empty": bool(only_non_empty_input),
            }
            st.session_state["passage_single_selected_row"] = None

        params = st.session_state.get("passage_search_params", {})
        q_content = params.get("q_content", "")
        q_studio = params.get("q_studio", "")
        q_unit = params.get("q_unit", "")
        q_passage_title = params.get("q_passage_title", "")
        row_query = params.get("row_query", "")
        only_non_empty = bool(params.get("only_non_empty", True))

        row_numbers: set[int] = set()
        if row_query:
            for token in row_query.split(","):
                token = token.strip()
                if token.isdigit():
                    row_numbers.add(int(token))

        def _contains_partial(row: dict, col_name: str, query: str) -> bool:
            if not query:
                return True
            return query in str(row.get(col_name, "")).lower()

        active_queries = [
            ("content", q_content),
            ("studio_title", q_studio),
            ("unit_title", q_unit),
            ("passage_title", q_passage_title),
        ]
        active_queries = [(col, q) for col, q in active_queries if q]

        has_search_input = bool(active_queries or row_numbers)
        if not has_search_input:
            st.info("검색어(또는 행 번호)를 입력하면 해당 결과만 표시됩니다.")
            filtered = []
        else:
            filtered = records
            if only_non_empty:
                filtered = [r for r in filtered if str(r.get("content", "")).strip()]

            if active_queries:
                filtered = [
                    r for r in filtered
                    if any(_contains_partial(r, col, q) for col, q in active_queries)
                ]
            if row_numbers:
                filtered = [r for r in filtered if int(r.get("sheet_row_index", 0)) in row_numbers]

        st.caption(f"조회 결과: {len(filtered)}개")

        if has_search_input and filtered:
            preview_source = filtered[:200]
            preview_row_ids = [item.get("sheet_row_index") for item in preview_source]
            selected_row_id = st.session_state.get("passage_single_selected_row")
            if selected_row_id not in preview_row_ids:
                selected_row_id = None
                st.session_state["passage_single_selected_row"] = None

            preview_rows = []
            result_columns = ["studio_title", "unit_title", "passage_title", "content"]
            for item in preview_source:
                row_view = {"선택": item.get("sheet_row_index") == selected_row_id}
                for col_name in result_columns:
                    value = str(item.get(col_name, ""))
                    limit = 120 if col_name == "content" else 80
                    row_view[col_name] = value if len(value) <= limit else f"{value[:limit]}..."
                preview_rows.append(row_view)

            edited_rows = st.data_editor(
                preview_rows,
                use_container_width=True,
                hide_index=True,
                key="passage_result_editor",
                column_config={
                    "선택": st.column_config.CheckboxColumn("선택", help="상세/DIFF에 사용할 항목 선택"),
                },
                disabled=["studio_title", "unit_title", "passage_title", "content"],
            )

            if hasattr(edited_rows, "to_dict"):
                edited_records = edited_rows.to_dict("records")
            else:
                edited_records = edited_rows

            checked_ids = []
            for idx, row in enumerate(edited_records):
                if idx < len(preview_source) and bool(row.get("선택", False)):
                    checked_ids.append(preview_source[idx].get("sheet_row_index"))

            next_selected_row_id = selected_row_id
            if not checked_ids:
                next_selected_row_id = None
            elif len(checked_ids) == 1:
                next_selected_row_id = checked_ids[0]
            else:
                if selected_row_id in checked_ids:
                    others = [rid for rid in checked_ids if rid != selected_row_id]
                    next_selected_row_id = others[-1] if others else selected_row_id
                else:
                    next_selected_row_id = checked_ids[-1]

            if (next_selected_row_id != selected_row_id) or (len(checked_ids) > 1):
                st.session_state["passage_single_selected_row"] = next_selected_row_id

            selected_items = []
            if next_selected_row_id is not None:
                selected_items = [
                    item for item in preview_source
                    if item.get("sheet_row_index") == next_selected_row_id
                ]

            st.caption(f"선택된 항목: {len(selected_items)}개")

            selected = None
            if selected_items:
                selected = selected_items[0]

            if selected:
                selected_row = selected.get("sheet_row_index", "")
                st.markdown(f"#### 📌 선택 항목 상세 (행 {selected_row})")

                short_cols = [c for c in ["passage_id", "passage_title", "source_id", "studio_title", "unit_order", "unit_title"] if c in available_cols]
                if short_cols:
                    meta = {c: selected.get(c, "") for c in short_cols}
                    st.json(meta, expanded=False)

                st.markdown("##### content")
                st.text_area(
                    "content",
                    value=str(selected.get("content", "")),
                    height=260,
                    key=f"passage_content_only_{selected_row}",
                    disabled=True,
                )

                st.markdown("##### 🔍 content Diff 비교")
                content_text = str(selected.get("content", "") or "")
                col_diff_left, col_diff_right = st.columns(2)
                with col_diff_left:
                    external_text = st.text_area(
                        "좌측: 외부에서 복사한 텍스트",
                        value="",
                        height=260,
                        key=f"passage_diff_external_{selected_row}",
                        placeholder="여기에 비교할 텍스트를 붙여넣으세요.",
                    )
                with col_diff_right:
                    st.text_area(
                        "우측: 조회한 시트 content",
                        value=content_text,
                        height=260,
                        key=f"passage_diff_content_{selected_row}",
                        disabled=True,
                    )

                compare_btn = st.button("비교 실행", key=f"passage_diff_run_{selected_row}")
                diff_text_key = f"passage_diff_text_{selected_row}"
                if compare_btn:
                    if external_text.strip():
                        st.session_state[diff_text_key] = external_text
                    else:
                        st.warning("비교할 외부 텍스트를 먼저 입력해주세요.")
                        st.session_state.pop(diff_text_key, None)

                committed_external_text = st.session_state.get(diff_text_key, "")
                if committed_external_text:
                    similarity = difflib.SequenceMatcher(
                        None,
                        content_text,
                        committed_external_text,
                    ).ratio() * 100
                    st.metric("유사도", f"{similarity:.2f}%")
                    left_html, right_html = _build_full_text_diff_html(committed_external_text, content_text)
                    st.markdown("##### 🧾 전체 지문 비교 (차이 지점 빨간색)")
                    view_left, view_right = st.columns(2)
                    with view_left:
                        st.markdown(
                            (
                                "<div style='border:1px solid #e5e7eb; border-radius:10px; padding:12px; "
                                "background:#ffffff; min-height:240px; white-space:pre-wrap; line-height:1.6;'>"
                                f"{left_html}"
                                "</div>"
                            ),
                            unsafe_allow_html=True,
                        )
                    with view_right:
                        st.markdown(
                            (
                                "<div style='border:1px solid #e5e7eb; border-radius:10px; padding:12px; "
                                "background:#ffffff; min-height:240px; white-space:pre-wrap; line-height:1.6;'>"
                                f"{right_html}"
                                "</div>"
                            ),
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("외부 텍스트를 입력하고 `비교 실행`을 누르면 Diff를 보여줍니다.")
            else:
                st.info("조회 결과에서 좌측 체크박스로 항목을 선택하면 상세/DIFF를 볼 수 있습니다.")
        elif has_search_input:
            st.info("조건에 맞는 지문이 없습니다. 검색어나 행 번호를 확인해주세요.")
    else:
        st.info("헤더를 찾지 못했습니다. 시트의 1행 헤더를 확인해주세요.")


# --- 모의고사 CSV 탭 ---
with tab_mock_csv:
    st.subheader("📥 모의고사 CSV")
    st.caption("모의고사 시트를 검색해 항목을 담고, CSV로 일괄 다운로드합니다.")

    if "mock_exam_export_rows" not in st.session_state:
        st.session_state["mock_exam_export_rows"] = []

    passage_sheet_name = st.secrets.get("PASSAGE_SHEET_NAME", "[DATA] Paragraph DB (영어 통합)")
    mock_ws_name = "모의고사"

    st.markdown("### 🧩 채우기 템플릿")
    col_t1, col_t2, col_t3 = st.columns([1, 1, 2])
    with col_t1:
        template_lessons = st.number_input("과 수", min_value=1, max_value=50, value=2, step=1, key="mock_tpl_lessons")
    with col_t2:
        template_items = st.number_input("과당 번호 수", min_value=1, max_value=100, value=10, step=1, key="mock_tpl_items")
    with col_t3:
        custom_lesson_counts = st.text_input(
            "과별 문제 수(선택)",
            value="",
            key="mock_tpl_custom_counts",
            placeholder="예: 1:5,3:8",
            help="과 번호(1부터) 기준 덮어쓰기입니다. 예: 1:5 -> 1과를 5문항으로 설정",
        ).strip()
        st.write("")
        if st.button("템플릿 생성 (예: 1과 1~10, 2과 1~10)", key="mock_tpl_generate_btn"):
            templated_rows = []
            lesson_count_map = {lesson: int(template_items) for lesson in range(1, int(template_lessons) + 1)}
            parse_error = False

            if custom_lesson_counts:
                for token in custom_lesson_counts.split(","):
                    token = token.strip()
                    if not token:
                        continue
                    if ":" not in token:
                        parse_error = True
                        break
                    lesson_no_str, count_str = token.split(":", 1)
                    lesson_no_str = lesson_no_str.strip()
                    count_str = count_str.strip()
                    if (not lesson_no_str.isdigit()) or (not count_str.isdigit()):
                        parse_error = True
                        break
                    lesson_no = int(lesson_no_str)
                    item_count = int(count_str)
                    if lesson_no <= 0 or item_count <= 0 or lesson_no > int(template_lessons):
                        parse_error = True
                        break
                    lesson_count_map[lesson_no] = item_count

            if parse_error:
                st.warning(
                    f"과별 문제 수 형식이 올바르지 않습니다. 예: 1:5,3:8 (범위: 1~{int(template_lessons)})"
                )
            else:
                for lesson in range(1, int(template_lessons) + 1):
                    count = int(lesson_count_map.get(lesson, int(template_items)))
                    for item_no in range(1, count + 1):
                        row = {col: "" for col in MOCK_CSV_COLUMNS}
                        row["slot_label"] = f"{lesson}과 {item_no}번"
                        row["null_locked"] = ""
                        row["passage_title"] = ""
                        templated_rows.append(row)

            if templated_rows:
                st.session_state["mock_exam_export_rows"] = templated_rows
                st.session_state["mock_action_selected_idx"] = 0
                st.success(f"템플릿 생성 완료: {len(templated_rows)}칸")

    st.markdown("### 💾 임시 저장")
    save_tab_name = st.text_input(
        "저장 탭 이름",
        value=st.session_state.get("mock_temp_save_tab_name", "임시저장_01"),
        key="mock_temp_save_tab_name",
        placeholder="예: 1차_모의고사_작업본",
    ).strip()

    base_tabs = {"교과서", "참고서", "모의고사", "지문 통합"}
    try:
        all_titles = _list_worksheet_titles(passage_sheet_name)
    except Exception as e:
        all_titles = []
        st.warning(f"저장 탭 목록 조회 실패: {e}")

    temp_titles = [t for t in all_titles if t not in base_tabs]
    selected_temp_tab = st.selectbox(
        "불러올 임시 저장 탭",
        options=temp_titles if temp_titles else [""],
        key="mock_temp_load_tab",
        format_func=lambda x: x if x else "(임시 저장 탭 없음)",
    )

    col_sv1, col_sv2 = st.columns([1, 1])
    with col_sv1:
        if st.button("현재 누적 목록 저장", key="mock_save_rows_btn"):
            rows_to_save = st.session_state.get("mock_exam_export_rows", [])
            if not save_tab_name:
                st.warning("저장할 탭 이름을 입력해주세요.")
            elif not rows_to_save:
                st.warning("저장할 누적 목록이 없습니다.")
            else:
                try:
                    _save_rows_to_worksheet(
                        passage_sheet_name,
                        save_tab_name,
                        rows_to_save,
                        TEMP_SAVE_COLUMNS,
                    )
                except Exception as e:
                    st.error(f"임시 저장 실패: {e}")
                else:
                    st.success(f"'{save_tab_name}' 탭에 임시 저장했습니다.")
                    st.rerun()
    with col_sv2:
        if st.button("선택 탭 불러오기", key="mock_load_rows_btn"):
            if not selected_temp_tab:
                st.warning("불러올 임시 저장 탭을 선택해주세요.")
            else:
                try:
                    loaded_rows = _load_rows_from_worksheet(passage_sheet_name, selected_temp_tab)
                except Exception as e:
                    st.error(f"불러오기 실패: {e}")
                else:
                    st.session_state["mock_exam_export_rows"] = loaded_rows
                    st.session_state["mock_action_selected_idx"] = 0 if loaded_rows else None
                    st.success(f"'{selected_temp_tab}' 탭에서 {len(loaded_rows)}개 행을 불러왔습니다.")
                    st.rerun()

    st.markdown("### 🔎 모의고사 지문 검색")
    requested_cols = [
        "passage_id",
        "passage_title",
        "source_id",
        "studio_title",
        "unit_order",
        "unit_title",
        "content",
        "content_markdown",
        "content_translated",
        "content_markdown_translated",
        "book_title",
        "footnote",
    ]

    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        st.text_input(
            "스프레드시트 이름",
            value=passage_sheet_name,
            key="mock_csv_sheet_name_fixed",
            disabled=True,
        )
    with col_s2:
        st.text_input("워크시트", value=mock_ws_name, key="mock_csv_ws_fixed", disabled=True)
        if st.button("검색 데이터 새로고침", key="mock_csv_reload"):
            _load_worksheet_records_by_name.clear()

    try:
        headers, records = _load_worksheet_records_by_name(
            passage_sheet_name,
            mock_ws_name,
        )
    except Exception as e:
        headers, records = [], []
        st.error(f"모의고사 시트를 불러오지 못했습니다: {e}")

    if headers:
        available_cols = [c for c in requested_cols if c in headers]
        flash = st.session_state.pop("mock_csv_flash", None)
        if isinstance(flash, dict):
            level = str(flash.get("level", "")).strip().lower()
            msg = str(flash.get("msg", "")).strip()
            if msg:
                if level == "success":
                    st.success(msg)
                elif level == "warning":
                    st.warning(msg)
                else:
                    st.info(msg)
        if "mock_csv_search_params" not in st.session_state:
            st.session_state["mock_csv_search_params"] = {
                "q_content": "",
                "q_studio": "",
                "q_unit": "",
                "q_passage_title": "",
                "row_query": "",
                "only_non_empty": True,
            }
        if "mock_csv_search_version" not in st.session_state:
            st.session_state["mock_csv_search_version"] = 0

        params = st.session_state.get("mock_csv_search_params", {})
        with st.form("mock_csv_search_form", clear_on_submit=False):
            col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns([2, 2, 2, 2, 1])
            with col_f1:
                q_content_input = st.text_input(
                    "content 검색",
                    value=params.get("q_content", ""),
                    key="mock_csv_q_content",
                    placeholder="예: climate",
                )
            with col_f2:
                q_studio_input = st.text_input(
                    "studio_title 검색",
                    value=params.get("q_studio", ""),
                    key="mock_csv_q_studio",
                    placeholder="예: ebs",
                )
            with col_f3:
                q_unit_input = st.text_input(
                    "unit_title 검색",
                    value=params.get("q_unit", ""),
                    key="mock_csv_q_unit",
                    placeholder="예: environment",
                )
            with col_f4:
                q_passage_title_input = st.text_input(
                    "passage_title 검색",
                    value=params.get("q_passage_title", ""),
                    key="mock_csv_q_passage_title",
                    placeholder="예: The Future of AI",
                )
            with col_f5:
                row_query_input = st.text_input(
                    "행 번호 조회(쉼표 구분)",
                    value=params.get("row_query", ""),
                    key="mock_csv_row_query",
                    placeholder="예: 12, 35, 104",
                )
                only_non_empty_input = st.checkbox(
                    "빈 지문 제외",
                    value=bool(params.get("only_non_empty", True)),
                    key="mock_csv_non_empty",
                )
            submitted = st.form_submit_button("검색 실행", type="primary")

        col_reset_l, col_reset_r = st.columns([1, 4])
        with col_reset_l:
            if st.button("검색 초기화", key="mock_csv_search_reset"):
                st.session_state["mock_csv_search_params"] = {
                    "q_content": "",
                    "q_studio": "",
                    "q_unit": "",
                    "q_passage_title": "",
                    "row_query": "",
                    "only_non_empty": True,
                }
                st.session_state["mock_csv_selected_row_ids"] = []

        if submitted:
            st.session_state["mock_csv_search_params"] = {
                "q_content": q_content_input.strip().lower(),
                "q_studio": q_studio_input.strip().lower(),
                "q_unit": q_unit_input.strip().lower(),
                "q_passage_title": q_passage_title_input.strip().lower(),
                "row_query": row_query_input.strip(),
                "only_non_empty": bool(only_non_empty_input),
            }
            st.session_state["mock_csv_selected_row_ids"] = []
            st.session_state["mock_csv_search_version"] = int(st.session_state.get("mock_csv_search_version", 0)) + 1

        params = st.session_state.get("mock_csv_search_params", {})
        q_content = params.get("q_content", "")
        q_studio = params.get("q_studio", "")
        q_unit = params.get("q_unit", "")
        q_passage_title = params.get("q_passage_title", "")
        row_query = params.get("row_query", "")
        only_non_empty = bool(params.get("only_non_empty", True))

        row_numbers: set[int] = set()
        if row_query:
            for token in row_query.split(","):
                token = token.strip()
                if token.isdigit():
                    row_numbers.add(int(token))

        active_queries = [
            ("content", q_content),
            ("studio_title", q_studio),
            ("unit_title", q_unit),
            ("passage_title", q_passage_title),
        ]
        active_queries = [(col, q) for col, q in active_queries if q]
        has_search_input = bool(active_queries or row_numbers)

        def _contains_partial(row: dict, col_name: str, query: str) -> bool:
            if not query:
                return True
            return query in str(row.get(col_name, "")).lower()

        if has_search_input:
            filtered = records
            if only_non_empty:
                filtered = [r for r in filtered if str(r.get("content", "")).strip()]
            if active_queries:
                filtered = [
                    r for r in filtered
                    if any(_contains_partial(r, col, q) for col, q in active_queries)
                ]
            if row_numbers:
                filtered = [r for r in filtered if int(r.get("sheet_row_index", 0)) in row_numbers]

            st.caption(f"검색 결과: {len(filtered)}개")
            if filtered:
                result_columns = ["studio_title", "unit_title", "passage_title", "content"]
                preview_source = filtered[:200]
                selected_row_ids = st.session_state.get("mock_csv_selected_row_ids", [])
                preview_row_ids = [item.get("sheet_row_index") for item in preview_source]
                selected_row_ids = [rid for rid in selected_row_ids if rid in preview_row_ids]
                st.session_state["mock_csv_selected_row_ids"] = selected_row_ids
                prev_selected_row_ids = list(selected_row_ids)
                preview_rows = []
                for item in preview_source:
                    rid = item.get("sheet_row_index")
                    is_selected = rid in selected_row_ids
                    row_view = {
                        "선택": is_selected,
                        "상태": "선택됨" if is_selected else "",
                    }
                    for col_name in result_columns:
                        value = str(item.get(col_name, ""))
                        limit = 120 if col_name == "content" else 80
                        row_view[col_name] = value if len(value) <= limit else f"{value[:limit]}..."
                    preview_rows.append(row_view)

                edited_rows = st.data_editor(
                    preview_rows,
                    use_container_width=True,
                    hide_index=True,
                    key=f"mock_csv_result_editor_{st.session_state.get('mock_csv_search_version', 0)}",
                    column_config={
                        "선택": st.column_config.CheckboxColumn("선택", help="CSV 목록에 담을 항목 체크"),
                        "상태": st.column_config.TextColumn("상태"),
                    },
                    disabled=["상태", "studio_title", "unit_title", "passage_title", "content"],
                )

                if hasattr(edited_rows, "to_dict"):
                    edited_records = edited_rows.to_dict("records")
                else:
                    edited_records = edited_rows

                selected_row_ids = []
                for idx, row in enumerate(edited_records):
                    if idx < len(preview_source) and bool(row.get("선택", False)):
                        selected_row_ids.append(preview_source[idx].get("sheet_row_index"))
                st.session_state["mock_csv_selected_row_ids"] = selected_row_ids
                if selected_row_ids != prev_selected_row_ids:
                    st.rerun()

                selected_items = []
                for item in preview_source:
                    if item.get("sheet_row_index") in selected_row_ids:
                        selected_items.append(item)

                st.caption(f"선택된 항목: {len(selected_items)}개")

                current_rows = st.session_state.get("mock_exam_export_rows", [])
                if current_rows and all("slot_label" in r for r in current_rows):
                    filled_count = sum(1 for r in current_rows if _slot_has_payload(r))
                    null_count = sum(1 for r in current_rows if _is_null_locked(r))
                    st.caption(f"템플릿 채움 상태: {filled_count}/{len(current_rows)} (NULL 고정 {null_count}칸)")

                if st.button("체크한 항목을 CSV 목록에 추가", key="mock_csv_add_row_btn"):
                    if not selected_items:
                        st.warning("먼저 항목을 선택해주세요.")
                    else:
                        existing_keys: set[str] = set()
                        for row in current_rows:
                            existing_keys.update(_make_passage_dedupe_keys(row))

                        added_count = 0
                        skipped_count = 0

                        for selected in selected_items:
                            export_row = {"sheet_row_index": selected.get("sheet_row_index", "")}
                            for col in MOCK_CSV_COLUMNS:
                                export_row[col] = selected.get(col, "")
                            # 누적 목록/삭제 선택 라벨에서 제목 표시용(다운로드 컬럼에는 영향 없음)
                            export_row["passage_title"] = selected.get("passage_title", "")

                            new_keys = _make_passage_dedupe_keys(export_row)
                            if new_keys & existing_keys:
                                skipped_count += 1
                                continue

                            if current_rows and all("slot_label" in r for r in current_rows):
                                empty_idx = next(
                                    (
                                        i for i, r in enumerate(current_rows)
                                        if (not _is_null_locked(r)) and (not _slot_has_payload(r))
                                    ),
                                    None,
                                )
                                if empty_idx is None:
                                    skipped_count += 1
                                    continue
                                slot_label = current_rows[empty_idx].get("slot_label", "")
                                merged = dict(current_rows[empty_idx])
                                merged.update(export_row)
                                merged["slot_label"] = slot_label
                                merged["null_locked"] = ""
                                current_rows[empty_idx] = merged
                            else:
                                # 템플릿이 아닌 일반 목록은 신규 추가를 맨 아래로만 추가
                                current_rows.append(export_row)
                            existing_keys.update(new_keys)
                            added_count += 1

                        st.session_state["mock_exam_export_rows"] = current_rows
                        st.session_state["mock_csv_selected_row_ids"] = []
                        st.session_state["mock_csv_search_version"] = int(st.session_state.get("mock_csv_search_version", 0)) + 1
                        if added_count and skipped_count:
                            st.session_state["mock_csv_flash"] = {
                                "level": "info",
                                "msg": f"{added_count}개 추가, {skipped_count}개는 중복/빈 슬롯 부족으로 제외되었습니다.",
                            }
                        elif added_count:
                            st.session_state["mock_csv_flash"] = {
                                "level": "success",
                                "msg": f"{added_count}개 항목을 CSV 목록에 추가했습니다.",
                            }
                        else:
                            st.session_state["mock_csv_flash"] = {
                                "level": "warning",
                                "msg": f"추가된 항목이 없습니다. {skipped_count}개는 중복/빈 슬롯 부족으로 제외되었습니다.",
                            }
                        st.rerun()
            else:
                st.info("조건에 맞는 지문이 없습니다. 검색어/행 번호를 확인해주세요.")
        else:
            st.info("검색어(또는 행 번호)를 입력하면 결과를 표시합니다.")
    else:
        st.info("모의고사 시트의 헤더를 찾지 못했습니다. 1행 헤더를 확인해주세요.")

    st.divider()
    st.markdown("### 🗂 누적 목록 관리")

    export_rows = st.session_state.get("mock_exam_export_rows", [])
    st.caption(f"현재 담긴 행 수: {len(export_rows)}")

    def _is_fixed_slot_mode(rows: list[dict]) -> bool:
        return bool(rows) and all("slot_label" in r for r in rows)

    def _payload_only(row: dict) -> dict:
        # slot_label은 고정 슬롯 식별자이므로 이동 대상에서 제외하고,
        # null_locked를 포함한 실제 행 상태는 함께 이동시킨다.
        return {k: v for k, v in row.items() if k != "slot_label"}

    def _apply_payload_to_slot(slot_row: dict, payload: dict) -> dict:
        new_row = {"slot_label": slot_row.get("slot_label", "")}
        for k, v in payload.items():
            if k != "slot_label":
                new_row[k] = v
        if "null_locked" not in new_row:
            new_row["null_locked"] = slot_row.get("null_locked", "")
        return new_row

    col_e1, col_e2 = st.columns([1, 1])
    with col_e1:
        if st.button("담은 목록 비우기", key="clear_mock_export_rows"):
            st.session_state["mock_exam_export_rows"] = []
            st.session_state["mock_action_selected_idx"] = None
            st.rerun()
    with col_e2:
        publish_tab_name = st.text_input(
            "저장할 시트 탭 이름",
            value=st.session_state.get("mock_publish_tab_name", "작업결과_01"),
            key="mock_publish_tab_name",
            placeholder="예: 작업결과_2026_03_17",
        ).strip()
        if st.button("시트 탭으로 저장", key="publish_to_sheet_btn"):
            if not publish_tab_name:
                st.warning("저장할 시트 탭 이름을 입력해주세요.")
            elif not export_rows:
                st.warning("저장할 누적 목록이 없습니다.")
            else:
                if _is_fixed_slot_mode(export_rows):
                    # 슬롯 모드에서는 빈 슬롯/NULL 고정 슬롯도 행 자체를 유지해서 저장한다.
                    rows_to_save = []
                    payload_cols = ["passage_title", *MOCK_CSV_COLUMNS]
                    for row in export_rows:
                        out = dict(row)
                        if _is_null_locked(row):
                            # NULL 고정 슬롯은 payload를 빈 값으로 고정 저장
                            for col in payload_cols:
                                out[col] = ""
                        rows_to_save.append(out)
                else:
                    rows_to_save = [
                        row for row in export_rows
                        if any(str(row.get(col, "")).strip() for col in ["content", "passage_id", "source_id", "book_title", "unit_title", "passage_title"])
                    ]
                if not rows_to_save:
                    st.warning("저장할 유효 데이터가 없습니다.")
                else:
                    try:
                        _save_rows_to_worksheet(
                            passage_sheet_name,
                            publish_tab_name,
                            rows_to_save,
                            TEMP_SAVE_COLUMNS,
                        )
                    except Exception as e:
                        st.error(f"시트 저장 실패: {e}")
                    else:
                        st.success(f"'{publish_tab_name}' 탭에 {len(rows_to_save)}개 행을 저장했습니다.")

    if export_rows:
        selected_idx = st.session_state.get("mock_action_selected_idx", None)
        if not isinstance(selected_idx, int) or not (0 <= selected_idx < len(export_rows)):
            selected_idx = None
            st.session_state["mock_action_selected_idx"] = None

        select_preview = []
        for i, row in enumerate(export_rows[:100]):
            content_preview = str(row.get("content", ""))
            if len(content_preview) > 120:
                content_preview = f"{content_preview[:120]}..."
            select_preview.append(
                {
                    "_idx": i,
                    "선택": i == selected_idx,
                    "slot": row.get("slot_label", ""),
                    "null_state": "NULL" if _is_null_locked(row) else "",
                    "passage_id": row.get("passage_id", ""),
                    "passage_title": row.get("passage_title", ""),
                    "content_preview": content_preview,
                }
            )
        edited_select_preview = st.data_editor(
            select_preview,
            use_container_width=True,
            hide_index=True,
            key="mock_action_select_table_editor",
            column_config={
                "선택": st.column_config.CheckboxColumn("선택", help="작업할 행을 하나만 선택하세요."),
                "null_state": st.column_config.TextColumn("NULL"),
                "_idx": None,
            },
            disabled=["slot", "null_state", "passage_id", "passage_title", "content_preview"],
        )
        if hasattr(edited_select_preview, "to_dict"):
            select_records = edited_select_preview.to_dict("records")
        else:
            select_records = edited_select_preview

        checked_idxs = []
        for rec in select_records:
            idx_raw = rec.get("_idx")
            try:
                idx = int(idx_raw)
            except Exception:
                continue
            if idx >= len(export_rows):
                continue
            if bool(rec.get("선택", False)):
                checked_idxs.append(idx)

        next_selected_idx = selected_idx
        if not checked_idxs:
            next_selected_idx = None
        elif len(checked_idxs) == 1:
            next_selected_idx = checked_idxs[0]
        else:
            if selected_idx in checked_idxs:
                others = [idx for idx in checked_idxs if idx != selected_idx]
                next_selected_idx = others[-1] if others else selected_idx
            else:
                next_selected_idx = checked_idxs[-1]

        if (next_selected_idx != selected_idx) or (len(checked_idxs) > 1):
            st.session_state["mock_action_selected_idx"] = next_selected_idx
            st.rerun()

        remove_idx = st.session_state.get("mock_action_selected_idx", None)
        if remove_idx is None:
            st.info("누적 목록 표에서 작업할 행을 먼저 선택해주세요.")
        else:
            st.caption(f"선택된 행: {remove_idx + 1}번째")

        if _is_fixed_slot_mode(export_rows):
            null_col1, null_col2 = st.columns([1, 1])
            with null_col1:
                if st.button("선택 슬롯 NULL 고정", key="mock_set_null_lock_btn", disabled=(remove_idx is None)):
                    rows = st.session_state.get("mock_exam_export_rows", [])
                    if isinstance(remove_idx, int) and 0 <= remove_idx < len(rows):
                        rows[remove_idx] = _apply_payload_to_slot(rows[remove_idx], {})
                        rows[remove_idx]["passage_title"] = ""
                        rows[remove_idx]["null_locked"] = "TRUE"
                        st.session_state["mock_exam_export_rows"] = rows
                        st.session_state["mock_action_selected_idx"] = remove_idx
                        st.rerun()
            with null_col2:
                if st.button("선택 슬롯 NULL 해제", key="mock_unset_null_lock_btn", disabled=(remove_idx is None)):
                    rows = st.session_state.get("mock_exam_export_rows", [])
                    if isinstance(remove_idx, int) and 0 <= remove_idx < len(rows):
                        rows[remove_idx]["null_locked"] = ""
                        st.session_state["mock_exam_export_rows"] = rows
                        st.session_state["mock_action_selected_idx"] = remove_idx
                        st.rerun()

        move_col1, move_col2, move_col3 = st.columns([1, 1, 2])
        with move_col1:
            if st.button("선택 항목 위로", key="mock_move_up_btn", disabled=(remove_idx is None)):
                rows = st.session_state.get("mock_exam_export_rows", [])
                if isinstance(remove_idx, int) and 0 < remove_idx < len(rows):
                    if _is_fixed_slot_mode(rows):
                        p1 = _payload_only(rows[remove_idx - 1])
                        p2 = _payload_only(rows[remove_idx])
                        rows[remove_idx - 1] = _apply_payload_to_slot(rows[remove_idx - 1], p2)
                        rows[remove_idx] = _apply_payload_to_slot(rows[remove_idx], p1)
                    else:
                        rows[remove_idx - 1], rows[remove_idx] = rows[remove_idx], rows[remove_idx - 1]
                    st.session_state["mock_exam_export_rows"] = rows
                    st.session_state["mock_action_selected_idx"] = remove_idx - 1
                    st.rerun()
        with move_col2:
            if st.button("선택 항목 아래로", key="mock_move_down_btn", disabled=(remove_idx is None)):
                rows = st.session_state.get("mock_exam_export_rows", [])
                if isinstance(remove_idx, int) and 0 <= remove_idx < len(rows) - 1:
                    if _is_fixed_slot_mode(rows):
                        p1 = _payload_only(rows[remove_idx])
                        p2 = _payload_only(rows[remove_idx + 1])
                        rows[remove_idx] = _apply_payload_to_slot(rows[remove_idx], p2)
                        rows[remove_idx + 1] = _apply_payload_to_slot(rows[remove_idx + 1], p1)
                    else:
                        rows[remove_idx], rows[remove_idx + 1] = rows[remove_idx + 1], rows[remove_idx]
                    st.session_state["mock_exam_export_rows"] = rows
                    st.session_state["mock_action_selected_idx"] = remove_idx + 1
                    st.rerun()
        with move_col3:
            target_position = st.selectbox(
                "선택 항목 이동 위치",
                options=list(range(1, len(export_rows) + 1)),
                format_func=lambda x: f"{x}번째",
                key="mock_move_target_pos",
            )
            if st.button("위치로 이동", key="mock_move_to_pos_btn", disabled=(remove_idx is None)):
                rows = st.session_state.get("mock_exam_export_rows", [])
                if isinstance(remove_idx, int) and 0 <= remove_idx < len(rows):
                    new_idx = max(0, min(len(rows) - 1, int(target_position) - 1))
                    if _is_fixed_slot_mode(rows):
                        payloads = [_payload_only(r) for r in rows]
                        moving = payloads.pop(remove_idx)
                        payloads.insert(new_idx, moving)
                        rows = [_apply_payload_to_slot(rows[i], payloads[i]) for i in range(len(rows))]
                    else:
                        item = rows.pop(remove_idx)
                        rows.insert(new_idx, item)
                    st.session_state["mock_exam_export_rows"] = rows
                    st.session_state["mock_action_selected_idx"] = new_idx
                    st.rerun()

        if st.button("선택 행 삭제", key="mock_remove_row_btn", disabled=(remove_idx is None)):
            rows = st.session_state.get("mock_exam_export_rows", [])
            if isinstance(remove_idx, int) and 0 <= remove_idx < len(rows):
                if _is_fixed_slot_mode(rows):
                    rows[remove_idx] = _apply_payload_to_slot(rows[remove_idx], {})
                    next_idx = remove_idx
                else:
                    rows.pop(remove_idx)
                    next_idx = min(remove_idx, max(len(rows) - 1, 0)) if rows else None
                st.session_state["mock_exam_export_rows"] = rows
                st.session_state["mock_action_selected_idx"] = next_idx
                st.rerun()

        preview = []
        for i, row in enumerate(export_rows[:100]):
            preview.append(
                {
                    "_idx": i,
                    "slot": row.get("slot_label", ""),
                    "null_locked": _is_null_locked(row),
                    "passage_id": row.get("passage_id", ""),
                    "book_title": row.get("book_title", ""),
                    "unit_title": row.get("unit_title", ""),
                    "passage_title": row.get("passage_title", ""),
                }
            )
        edited_preview = st.data_editor(
            preview,
            use_container_width=True,
            hide_index=True,
            key="mock_slot_table_editor",
            column_config={
                "null_locked": st.column_config.CheckboxColumn("NULL 유지", help="체크하면 해당 슬롯은 자동 채움 대상에서 제외됩니다."),
                "_idx": None,
            },
            disabled=["slot", "passage_id", "book_title", "unit_title", "passage_title"],
        )

        if hasattr(edited_preview, "to_dict"):
            edited_records = edited_preview.to_dict("records")
        else:
            edited_records = edited_preview

        rows = st.session_state.get("mock_exam_export_rows", [])
        changed = False
        blocked_null_lock = 0
        for rec in edited_records:
            idx_raw = rec.get("_idx")
            try:
                idx = int(idx_raw)
            except Exception:
                continue
            if not (0 <= idx < len(rows)):
                continue
            current_locked = _is_null_locked(rows[idx])
            wanted_locked = bool(rec.get("null_locked", False))
            # passage_id가 있는 행은 "새로 NULL 고정"만 막고,
            # 기존 상태를 렌더링 과정에서 자동 해제하지는 않는다.
            if wanted_locked and (not current_locked) and str(rows[idx].get("passage_id", "")).strip():
                blocked_null_lock += 1
                wanted_locked = current_locked
            if wanted_locked != current_locked:
                rows[idx]["null_locked"] = "TRUE" if wanted_locked else ""
                if wanted_locked:
                    rows[idx] = _apply_payload_to_slot(rows[idx], {})
                    rows[idx]["passage_title"] = ""
                    rows[idx]["null_locked"] = "TRUE"
                changed = True
        if blocked_null_lock:
            st.warning(f"passage_id가 있는 {blocked_null_lock}개 행은 NULL 유지로 설정할 수 없습니다.")
        if changed:
            st.session_state["mock_exam_export_rows"] = rows
            st.rerun()
    else:
        st.info("아직 담긴 항목이 없습니다. 영어 지문 조회 탭에서 행을 추가해주세요.")


# --- 설명 탭 ---
# --- 설명 탭 ---
with tab_about:
    st.title("📘 텍스트 자동 검수기 설명서")
    st.caption("이 탭은 전체 앱의 구조와 동작 방식을 설명합니다.")

    about_sections = {
        "✨ 앱 소개": """
## ✨ 이 앱은 무엇을 하나요?

이 앱은 **한국어/영어 단일 텍스트 검수기**와  
**Google Sheets 기반 배치 검수기**를 포함한 **통합 자동 검수 플랫폼**입니다.

- 자연스러움, 문체, 표현 개선 등 **주관적 수정은 전혀 하지 않습니다.**  
- 오직 **객관적으로 검증 가능한 오류만** 검출합니다.  
- 모든 검수는 **JSON-only 응답 + 후처리 안정화 로직** 기반으로 작동하여  
  오탐(False Positive)과 누락을 최소화합니다.

---
""",
        "✏️ 한국어 검수": """
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
""",
        "✏️ 영어 검수": """
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
""",
        "📄 시트 배치 검수": """
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
""",
        "📄 해설 텍스트 정리": """
# ✏️ 해설 텍스트 변환

## 🔍 기능 개요
해설 텍스트를 **[정답 해설] / [오답 해설]** 양식에 맞게 변환합니다.

- **[출제 유형] ~** 삭제됩니다.
- 정답인 이유/답이 아닌 이유 형식은 **[정답 해설] / [오답 해설]** 양식으로 변환됩니다.

---

## 🧠 작동 방식

1. PDF에서 OCR한 텍스트를 넣어줍니다.
2. 텍스트 정리 실행 버튼을 클릭합니다.
3. 변환된 텍스트를 PDF와 비교 후 일치할 경우 복사해서 해설 영역에 넣어주세요.

---
""",
        "🎯 철학 & 규칙": """
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
""",
    }

    selected_section = st.radio(
        "섹션 선택",
        options=list(about_sections.keys()),
        horizontal=True,
        key="about_section_selector",
    )

    st.markdown(about_sections.get(selected_section, ""))

# --- 디버그 탭 ---
with tab_debug:
    st.subheader("🐞 디버그 / 정산")
    st.caption("Gemini 호출 로그를 기반으로 기능별 비용 및 토큰 사용량을 집계합니다.")
    

    ws = _get_log_worksheet()
    if ws is None:
        st.warning("로그 시트를 불러올 수 없습니다.")
        st.stop()

    try:
        values = ws.get_all_values()
    except Exception as e:
        st.warning(f"로그 시트 조회 중 오류가 발생했습니다: {e}")
        st.info("권한/시트 존재 여부/Google API 일시 오류를 확인한 뒤 다시 시도해주세요.")
        st.stop()

    if not values or len(values) < 2:
        st.info("아직 로그 데이터가 없습니다.")
        st.stop()

    import pandas as pd

    header = values[0]
    rows = values[1:]
    normalized = [_normalize_row_to_v2(header, row) for row in rows]
    df = pd.DataFrame(normalized)

    KRW_PER_USD = st.number_input(
        "환율 (KRW/USD)", min_value=500, max_value=3000, value=1450, step=10
    )

    # -------------------------------
    # ✅ 안전 처리: 컬럼 없으면 먼저 생성
    # -------------------------------
    for col in ["cost_usd", "prompt_tokens", "output_tokens", "total_tokens"]:
        if col not in df.columns:
            df[col] = 0

    # 숫자 변환 + 결측 처리
    df["cost_usd"] = pd.to_numeric(df["cost_usd"], errors="coerce").fillna(0.0)
    df["prompt_tokens"] = pd.to_numeric(df["prompt_tokens"], errors="coerce").fillna(0).astype(int)
    df["output_tokens"] = pd.to_numeric(df["output_tokens"], errors="coerce").fillna(0).astype(int)
    df["total_tokens"] = pd.to_numeric(df["total_tokens"], errors="coerce").fillna(0).astype(int)

    # timestamp → 날짜 (컬럼 없을 수도 있으니 방어)
    if "timestamp_utc" not in df.columns:
        df["timestamp_utc"] = None
    df["date"] = pd.to_datetime(df["timestamp_utc"], errors="coerce").dt.date

    # -------------------------------
    # 필터 UI
    # -------------------------------
    st.markdown("### 🔍 필터")

    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        feature_options = sorted(df["feature"].dropna().unique().tolist()) if "feature" in df.columns else []
        feature_filter = st.multiselect("Feature 선택", options=feature_options, default=None)

    with col_f2:
        model_options = sorted(df["model"].dropna().unique().tolist()) if "model" in df.columns else []
        model_filter = st.multiselect("Model 선택", options=model_options, default=None)

    with col_f3:
        date_options = sorted(df["date"].dropna().unique().tolist())
        date_filter = st.multiselect("날짜 선택", options=date_options, default=None)

    if feature_filter and "feature" in df.columns:
        df = df[df["feature"].isin(feature_filter)]
    if model_filter and "model" in df.columns:
        df = df[df["model"].isin(model_filter)]
    if date_filter:
        df = df[df["date"].isin(date_filter)]

    # -------------------------------
    # 전체 요약
    # -------------------------------
    st.markdown("### 💰 전체 요약")

    total_cost = float(df["cost_usd"].sum())
    total_cost_krw = total_cost * KRW_PER_USD
    total_calls = int(len(df))
    total_tokens = int(df["total_tokens"].sum())

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("총 호출 수", f"{total_calls:,}")
    col_m2.metric("총 토큰 수", f"{total_tokens:,}")
    col_m3.metric("총 비용 (USD)", f"${total_cost:.4f}")
    col_m4.metric("총 비용 (KRW)", f"₩{total_cost_krw:,.0f}")

    # -------------------------------
    # Feature별 정산
    # -------------------------------
    st.markdown("### 🧾 Feature별 정산")

    if "feature" not in df.columns:
        st.warning("로그에 feature 컬럼이 없어 Feature별 집계를 할 수 없습니다.")
    else:
        feature_summary = (
            df.groupby("feature", dropna=False)
            .agg(
                calls=("feature", "count"),
                total_cost_usd=("cost_usd", "sum"),
                total_tokens=("total_tokens", "sum"),
                prompt_tokens=("prompt_tokens", "sum"),
                output_tokens=("output_tokens", "sum"),
            )
            .sort_values("total_cost_usd", ascending=False)
            .reset_index()
        )

        # ✅ 원화 컬럼 추가
        feature_summary["total_cost_krw"] = feature_summary["total_cost_usd"] * KRW_PER_USD

        # 보기 좋게 컬럼 순서 정리 (선택)
        feature_summary = feature_summary[
            ["feature", "calls", "total_cost_usd", "total_cost_krw", "total_tokens", "prompt_tokens", "output_tokens"]
        ]

        st.dataframe(feature_summary, use_container_width=True, hide_index=True)

    # -------------------------------
    # 날짜별 비용 추이
    # -------------------------------
    st.markdown("### 📈 날짜별 비용 추이")

    daily_cost = (
        df.groupby("date", dropna=False)
        .agg(total_cost_usd=("cost_usd", "sum"))
        .reset_index()
        .sort_values("date")
    )
    daily_cost["total_cost_krw"] = daily_cost["total_cost_usd"] * KRW_PER_USD

    # USD 그래프(기존)
    st.line_chart(daily_cost.set_index("date")["total_cost_usd"])

    # KRW도 같이 보고 싶으면 아래도 추가로 켜면 됨
    st.line_chart(daily_cost.set_index("date")["total_cost_krw"])

    # -------------------------------
    # 원본 로그 (확인용)
    # -------------------------------
    with st.expander("📄 원본 로그 데이터 보기 (최근 200건)", expanded=False):
        if "timestamp_utc" in df.columns:
            view_df = df.sort_values("timestamp_utc", ascending=False).head(200)
        else:
            view_df = df.head(200)
        st.dataframe(view_df, use_container_width=True)
