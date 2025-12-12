# AI 텍스트 검수기 (Korean/English, Gemini 기반)

한국어/영어 단일 텍스트 검수와 Google Sheets 배치 검수를 제공하는 Streamlit 앱입니다.  
스타일·의역 없이 **객관적 오류**(오탈자, 띄어쓰기, 문장부호 등)만 탐지하도록 설계되었으며, Gemini JSON 모드 + 2‑패스(Detector → Judge) + 규칙 후처리를 결합했습니다.

## 1. 설계 개요
- **입력 채널**
  - 단일 텍스트: 한국어/영어 탭
  - 배치: Google Sheets (STATUS='1. AI검수요청' 행만 처리)
- **LLM 호출**
  - Gemini 2.0 flash JSON 모드, temperature=0, 최대 5회 재시도
  - 2‑패스: 1차 Detector(과검출 허용) → 2차 Judge(스타일/의역 제거)
  - 한국어는 길면 chunk 처리, 블록별 결과를 헤더와 함께 병합
- **후처리**
  - 존재하지 않는 원문 인용 제거, self‑equal 제거, 과도한 길이 수정 제거
  - 종결부호 오탐/공백 스타일 오탐/불필요한 공백 오탐 제거
  - plain vs markdown 분리 저장, 문장부호 누락 강제 추가(요약 메시지)
- **UI**
  - Detector/Judge/Final 하이라이트 토글, 오류가 없어도 원문 항상 표시
  - 문장부호 선택 필터 + 카운트 배지, 색상 안내
  - Raw vs Final 비교, Diff, 수정 제안, Detector/Judge JSON expander

## 2. 실행 방법
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 필수 secrets (`.streamlit/secrets.toml` 예시)
```toml
GEMINI_API_KEY = "YOUR_KEY"
GCP_SERVICE_ACCOUNT_JSON = """{ ... 서비스계정 JSON ... }"""
```

## 3. 주요 기능 흐름 (app.py)
1) **프롬프트 생성**
   - 한국어: Detector/ Judge/ Review 프롬프트 분리, chunk 지원
   - 영어: Detector/ Judge/ Review 프롬프트 분리
2) **LLM 호출**: `analyze_text_with_gemini` (JSON mode, retry 5회)
3) **검수 파이프라인**
   - 한국어: `_review_korean_single_block` → chunk 시 `review_korean_text`
   - 영어: `review_english_text`
4) **후처리 핵심**
   - `drop_lines_not_in_source`(원문 불일치 제거)
   - `drop_false_korean_period_errors`, `drop_false_period_claims`, `ensure_final_punctuation_error`, `ensure_sentence_end_punctuation`
   - `drop_false_whitespace_claims`(불필요한 공백 오탐 제거)
   - `dedup_korean_bullet_lines`, `clean_self_equal_corrections`
5) **하이라이트**
   - 언어별 파서(`parse_korean_report_with_positions`, `parse_english_report_with_positions`)
   - `highlight_text_with_spans`: 오류 mark + 선택 문장부호만 색상 적용
   - 모드: Detector / Judge / Final, 보기 모드: 오류 하이라이트 / 문장부호만
6) **UI 카드**
   - 상단: 오류 위치 · 하이라이트 (기준 선택, 문장부호 선택, 보기 모드, 프리뷰, 부호 카운트)
   - 하단: 결과 비교 · 제안 (Final/Raw JSON, Diff, 수정 제안, Detector/Judge JSON)
7) **설명 탭**
   - 2‑패스 구조, 한국어/영어 검수 규칙, 시트 배치 검수 흐름을 문서화

## 4. 배치 검수 흐름 (sheet_review.py)
1) **입력 스키마**
   - 컬럼: `content`, `content_markdown`, `content_translated`, `content_markdown_translated`, `STATUS`, `SCORE`, `CONTENT_TYPO_REPORT`, `TRANSLATED_TYPO_REPORT`, `MARKDOWN_REPORT`
   - 대상: `STATUS == "1. AI검수요청"`
2) **행 처리**
   - 영어: `create_english_review_prompt` → `analyze_text_with_gemini` → `validate_and_clean_analysis` → `sanitize_report` → `ensure_sentence_end_punctuation`
   - 한국어: `create_korean_review_prompt` → 동일 흐름 + `ensure_final_punctuation_error`/`dedup_korean_bullet_lines`
   - plain/markdown 분리: `split_report_by_source`, markdown 오류는 `MARKDOWN_REPORT`로 집계
   - 통합 스코어: 영어/한국어 score 중 max
3) **후처리 필터** (sheet_review.py 공통)
   - `remove_self_equal`, `drop_escape_false`, `drop_language_switch`, `drop_large_edits`
   - `drop_false_period_claims`, `drop_punctuation_space_style`, `drop_false_whitespace_claims`
   - `drop_lines_not_in_source`(공백 제거 후 재검증)
4) **출력**
   - 각 행: SCORE, *_REPORT, STATUS="2. AI검수완료" 업데이트
   - `collect_raw=True` 시 raw/final 번들 저장 (디버그 탭에서 조회)

## 5. 문장부호 색상 팔레트
- 종결부호 `.`: `#ffe08a`
- 물음표 `?`: `#f2a6b3`
- 느낌표 `!`: `#f28b90`
- 쉼표 `,`: `#9fd3e6`
- 쌍따옴표 `"”“`: `#b9e6c8`
- 작은따옴표 `''‘’`: `#f7b58d`
- 세미콜론/콜론: `#bfc2c4`

## 6. 설계 포인트 요약
- **2‑패스 안정화**: 1차 과검출, 2차 필터링으로 스타일/의역 제거
- **룰 기반 후처리**: 종결부호/공백/escape/언어스위치 등 휴리스틱 필터
- **오류 없는 경우에도 원문 표시**: 문장부호 필터/카운트와 함께 활용 가능
- **chunk 지원(한국어)**: 긴 텍스트를 블록별 검수, 헤더로 구분
- **UI 투명성**: Raw/Final/차이/수정 제안/Detector·Judge JSON을 모두 노출

## 7. 확장 가이드
- 테마: 문장부호 색상/배경 색상은 `PUNCT_COLOR_MAP`과 UI 스타일에서 조정
- 규칙 추가: 후처리 필터(`drop_*` 계열)나 프롬프트 텍스트 수정
- 시트 스키마 변경 시: 컬럼 상수(STATUS_COL 등)와 split 로직을 함께 수정
- 성능: 2‑패스/재시도/슬립(0.5s) → 필요 시 동시성/레이트리밋 튜닝

## 8. 실행/오류 대응
- `GEMINI_API_KEY`, `GCP_SERVICE_ACCOUNT_JSON` 미설정 시 앱이 즉시 종료
- `worksheet.get_all_records()`는 빈 행을 건너뛰므로, 정확한 행 인덱스는 `sheet_row_index`로 관리(헤더 보정 +2)
- 긴 텍스트/여러 chunk 시 호출 수 증가 → 비용·시간 유념

---

필요한 항목(예: 테스트 방법, 배포 방법, 더 자세한 프롬프트 전문)이 있다면 README에 추가해 주세요. 그대로 노션으로 옮겨도 구조가 유지됩니다.
