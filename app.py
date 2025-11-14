import re
import time
import asyncio
import unicodedata
import string
from typing import Any, Dict, Optional, List

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

MESSAGES_URL = "https://november7-730026606190.europe-west1.run.app/messages"

app = FastAPI(
    title="Member Messages QA Service",
    description="Ask natural-language questions about member messages.",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskBody(BaseModel):
    question: str = Field(..., description="Natural-language question to answer")
    model_config = {"json_schema_extra": {"example": {"question": "When is Layla planning her trip to London?"}}}

class AskResponse(BaseModel):
    answer: str
    matched_message: Optional[Dict[str, Any]] = None
    evidence: Optional[List[Dict[str, Any]]] = None

CACHE: Dict[str, Any] = {"data": None, "ts": 0.0}
CACHE_TTL_SEC = 180

MONTH_RX = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
ORD_RX = r"(?:st|nd|rd|th)?"
REL_RX = r"(?:tomorrow|today|tonight|this weekend|next week|next month|this week)"
DATE_RX = re.compile(rf"\b{MONTH_RX}\s+\d{{1,2}}{ORD_RX}(?:,\s*\d{{4}})?\b", re.IGNORECASE)
ISO_RX = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
REL_RXC = re.compile(rf"\b{REL_RX}\b", re.IGNORECASE)

NUMBER_WORDS = {
    "one": "1","two": "2","three": "3","four": "4","five": "5",
    "six": "6","seven": "7","eight": "8","nine": "9","ten": "10"
}
NUM_DIGIT_RX = re.compile(r"\b\d+\b")
NUM_WORD_RX = re.compile(r"\b(" + "|".join(NUMBER_WORDS.keys()) + r")\b", re.IGNORECASE)

MEETING_TERMS = ("meeting","meet ")
TICKET_TERMS = ("ticket","tickets","vip","broadway","met")
TRAVEL_TERMS = ("trip","travel","flight","fly","flying","visit","booking","book","reservation","first class")

CITY_TERMS = {
    "london","paris","berlin","tokyo","dubai","zurich","vatican","rome","madrid",
    "amsterdam","new york","san francisco","chicago","los angeles","la","seattle"
}

_PUNC_TABLE = str.maketrans({c: " " for c in (string.punctuation + "—–“”‘’")})
_WH = {"what","when","how","who","where","why","which"}

def _now() -> float:
    return time.time()

def _is_fresh_cache() -> bool:
    return bool(CACHE["data"]) and (_now() - (CACHE["ts"] or 0)) <= CACHE_TTL_SEC

def _normalize_str(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.translate(_PUNC_TABLE)
    s = " ".join(s.lower().split())
    return s

def _normalize_messages(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        if isinstance(raw, dict):
            for k in ("messages", "data", "items", "results"):
                if isinstance(raw.get(k), list):
                    raw = raw[k]
                    break
        if not isinstance(raw, list):
            raise HTTPException(status_code=502, detail="Upstream payload is not a list of messages")
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = item.get("message") or item.get("text") or item.get("content")
        if not text:
            continue
        member = item.get("member_name") or item.get("user_name") or item.get("member") or item.get("name") or item.get("user")
        out.append({"raw": item, "text": text, "member": member})
    return out

def find_name_in_question(q: str) -> List[str]:
    toks = q.strip().split()
    cands: List[str] = []
    for i, t in enumerate(toks):
        if t and t[0].isupper() and t.lower() not in _WH and len(t) > 1:
            cands.append(t)
            if i + 1 < len(toks) and toks[i+1] and toks[i+1][0].isupper() and toks[i+1].lower() not in _WH:
                cands.append(t + " " + toks[i+1])
    return list(dict.fromkeys(cands))

def score_message(msg_text: str, question: str) -> int:
    q_words = set(question.split())
    m_words = set(msg_text.split())
    return len(q_words & m_words)

def try_extract_date(text: str) -> Optional[str]:
    if not text:
        return None
    m = DATE_RX.search(text)
    if m:
        return m.group(0)
    m2 = ISO_RX.search(text)
    if m2:
        return m2.group(0)
    m3 = REL_RXC.search(text)
    if m3:
        return m3.group(0)
    return None

def try_extract_number(text: str) -> Optional[str]:
    if not text:
        return None
    m = NUM_DIGIT_RX.search(text)
    if m:
        return m.group(0)
    mw = NUM_WORD_RX.search(text)
    if mw:
        w = mw.group(1).lower()
        return NUMBER_WORDS.get(w)
    return None

async def fetch_messages_with_retry() -> List[Dict[str, Any]]:
    attempts = 3
    last_error_detail: Optional[str] = None
    for i in range(attempts):
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                resp = await client.get(MESSAGES_URL)
                if resp.status_code == 200:
                    data = resp.json()
                    CACHE["data"] = data
                    CACHE["ts"] = _now()
                    return _normalize_messages(data)
                else:
                    last_error_detail = f"Upstream returned {resp.status_code}"
        except httpx.RequestError as e:
            last_error_detail = f"Upstream request error: {e!s}"
        await asyncio.sleep(0.5 + 0.5 * i)
    if _is_fresh_cache():
        return _normalize_messages(CACHE["data"])
    raise HTTPException(status_code=502, detail=last_error_detail or "Upstream unreachable")

@app.get("/")
async def root():
    return {"status": "ok", "docs": "/docs"}

@app.get("/healthz")
async def healthz():
    age = _now() - CACHE["ts"] if CACHE["ts"] else None
    return {"ok": True, "cache_age_sec": age}

@app.get("/debug/upstream")
async def debug_upstream():
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(MESSAGES_URL)
            info: Dict[str, Any] = {"status_code": resp.status_code, "ok": resp.status_code == 200}
            if resp.status_code == 200:
                payload = resp.json()
                info["type"] = type(payload).__name__
                if isinstance(payload, list) and payload:
                    info["first_item_type"] = type(payload[0]).__name__
                    if isinstance(payload[0], dict):
                        info["first_item_keys"] = list(payload[0].keys())
                elif isinstance(payload, dict):
                    info["payload_keys"] = list(payload.keys())
            return info
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {e!s}")

@app.get("/debug/sample")
async def debug_sample():
    try:
        msgs = await fetch_messages_with_retry()
        if not msgs:
            return {"note": "no messages after normalization"}
        first = msgs[0].copy()
        if isinstance(first.get("raw"), dict):
            first["raw"] = {k: first["raw"].get(k) for k in list(first["raw"].keys())[:10]}
        return first
    except HTTPException as e:
        return JSONResponse({"debug": "normalization failed", "detail": e.detail}, status_code=e.status_code)

@app.get("/inspect")
async def inspect(query: str = Query(..., min_length=1), require_date: bool = Query(False)):
    msgs = await fetch_messages_with_retry()
    out = []
    q = _normalize_str(query)
    for m in msgs:
        t = _normalize_str(m.get("text") or "")
        if q in t:
            if not require_date or try_extract_date(m.get("text") or ""):
                out.append({"text": m.get("text") or "", "member": m.get("member"), "has_date": bool(try_extract_date(m.get("text") or "")), "raw": m.get("raw")})
    return {"count": len(out), "matches": out[:50]}

@app.post("/ask", response_model=AskResponse)
async def ask(body: AskBody):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Field 'question' must be a non-empty string.")
    try:
        messages = await fetch_messages_with_retry()
    except HTTPException as e:
        raise e
    except Exception:
        raise HTTPException(status_code=502, detail="Unexpected upstream handling error")
    if not messages:
        return JSONResponse({"answer": "I couldn't find an answer in the member messages.", "matched_message": None})

    q_norm = _normalize_str(question)
    q_names = find_name_in_question(question)
    wants_when = ("when" in q_norm) or ("date" in q_norm) or ("what date" in q_norm) or ("on what date" in q_norm) or ("schedule" in q_norm) or ("planning" in q_norm)
    wants_number = ("how many" in q_norm) or ("number" in q_norm) or ("tickets" in q_norm)
    wants_meeting = ("meeting" in q_norm) or ("meet" in q_norm)
    asked_broadway = "broadway" in q_norm
    asked_ticket_domain = any(t in q_norm for t in TICKET_TERMS)
    asked_travel_domain = any(t in q_norm for t in TRAVEL_TERMS)

    asked_cities = {c for c in CITY_TERMS if c in q_norm}

    candidate = messages

    if q_names:
        tokens = [_normalize_str(t) for n in q_names for t in n.split()]
        by_name = []
        for mm in candidate:
            field = _normalize_str((mm.get("member") or "") + " " + (mm.get("text") or ""))
            if all(t in field for t in tokens):
                by_name.append(mm)
        if not by_name:
            return {"answer": f"I couldn't find any message mentioning {q_names[0]}.", "matched_message": None, "evidence": []}
        candidate = by_name

    if asked_cities:
        by_city = []
        for mm in candidate:
            t = _normalize_str(mm.get("text") or "")
            if any(c in t for c in asked_cities):
                by_city.append(mm)
        if not by_city:
            city_str = ", ".join(sorted(asked_cities))
            return {"answer": f"I couldn't find any message mentioning {city_str}.", "matched_message": None, "evidence": []}
        candidate = by_city

    if wants_meeting:
        by_meeting = []
        for mm in candidate:
            t = _normalize_str(mm.get("text") or "")
            if any(term in t for term in MEETING_TERMS):
                by_meeting.append(mm)
        if not by_meeting:
            if asked_cities:
                city_str = ", ".join(sorted(asked_cities))
                return {"answer": f"No messages about a meeting found for {city_str}.", "matched_message": None, "evidence": []}
            return {"answer": "No messages about a meeting found.", "matched_message": None, "evidence": []}
        candidate = by_meeting

    def bonus_score(mm: Dict[str, Any]) -> int:
        text = mm.get("text") or ""
        t = _normalize_str(text)
        base = score_message(t, q_norm)
        s = base
        if wants_when:
            s += 6 * (1 if try_extract_date(text) else 0)
        if wants_number:
            s += 5 * (1 if try_extract_number(text) else 0)
            if asked_broadway:
                s += 3 * (1 if "broadway" in t else 0)
            if asked_ticket_domain:
                s += 2 * (1 if any(term in t for term in TICKET_TERMS) else 0)
        if wants_meeting:
            s += 3 * (1 if any(term in t for term in MEETING_TERMS) else 0)
        for c in asked_cities:
            s += 2 * (1 if c in t else 0)
        if asked_travel_domain:
            s += 2 * (1 if any(term in t for term in TRAVEL_TERMS) else 0)
        return s

    ranked = sorted(candidate, key=bonus_score, reverse=True)
    if not ranked:
        return {"answer": "No relevant messages found.", "matched_message": None, "evidence": []}

    evidence = [{"text": (mm.get("text") or ""), "member": mm.get("member")} for mm in ranked[:5]]

    if wants_number:
        filtered = ranked
        if asked_ticket_domain:
            fb = [mm for mm in filtered if any(term in _normalize_str(mm.get("text") or "") for term in TICKET_TERMS)]
            if fb:
                filtered = fb
        if asked_broadway:
            fb2 = [mm for mm in filtered if "broadway" in _normalize_str(mm.get("text") or "")]
            if fb2:
                filtered = fb2
        for mm in filtered:
            num = try_extract_number(mm.get("text") or "")
            if num:
                return {"answer": num, "matched_message": mm.get("raw"), "evidence": evidence}
        return {"answer": "No explicit number mentioned in the relevant messages.", "matched_message": None, "evidence": evidence}

    if wants_when:
        for mm in ranked:
            txt = mm.get("text") or ""
            dt = try_extract_date(txt)
