import re
import time
import asyncio
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
    version="1.6.0",
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

def _now() -> float:
    return time.time()

def _is_fresh_cache() -> bool:
    return bool(CACHE["data"]) and (_now() - (CACHE["ts"] or 0)) <= CACHE_TTL_SEC

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
            item = {"value": item}
        text = item.get("message") or item.get("text") or item.get("content") or str(item)
        member = item.get("member_name") or item.get("member") or item.get("name") or item.get("user")
        out.append({"raw": item, "text": text, "member": member})
    return out

def find_name_in_question(q: str) -> List[str]:
    tokens = q.strip().split()
    candidates: List[str] = []
    for i, t in enumerate(tokens):
        if t.istitle():
            candidates.append(t)
            if i + 1 < len(tokens) and tokens[i + 1].istitle():
                candidates.append(t + " " + tokens[i + 1])
    return list(dict.fromkeys(candidates))

def score_message(msg_text: str, question: str) -> int:
    q_words = set(question.lower().split())
    m_words = set(msg_text.lower().split())
    return len(q_words & m_words)

def try_extract_date(text: str) -> Optional[str]:
    if not text:
        return None
    m1 = re.search(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,\s*\d{4})?", text, re.IGNORECASE)
    if m1:
        return m1.group(0)
    m2 = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
    if m2:
        return m2.group(0)
    m3 = re.search(r"\b(?:tomorrow|today|tonight|this weekend|next week|next month|this week)\b", text, re.IGNORECASE)
    if m3:
        return m3.group(0)
    return None

def try_extract_number(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"\b\d+\b", text)
    if m:
        return m.group(0)
    return None

async def fetch_messages_with_retry() -> List[Dict[str, Any]]:
    attempts = 3
    last_error_detail: Optional[str] = None
    for i in range(attempts):
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                resp = await client.get(MESSAGES_URL)
                print(f"[upstream] attempt={i+1} status={resp.status_code}")
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
        print("[upstream] using cached messages")
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
    q = query.lower()
    for m in msgs:
        t = (m.get("text") or "")
        if q in t.lower():
            if not require_date or try_extract_date(t):
                out.append({"text": t, "member": m.get("member"), "has_date": bool(try_extract_date(t)), "raw": m.get("raw")})
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
    except Exception as e:
        print(f"[ask] unexpected error during fetch: {e!r}")
        raise HTTPException(status_code=502, detail="Unexpected upstream handling error")
    if not messages:
        return JSONResponse({"answer": "I couldn't find an answer in the member messages.", "matched_message": None})

    q_lower = question.lower()
    q_names = find_name_in_question(question)
    wants_when = ("when" in q_lower) or ("date" in q_lower) or ("planning" in q_lower)
    travel_words = ("plan" in q_lower) or ("planning" in q_lower) or ("trip" in q_lower) or ("travel" in q_lower) or ("visit" in q_lower) or ("fly" in q_lower) or ("flying" in q_lower)
    asked_london = "london" in q_lower

    candidate_msgs = messages
    if q_names:
        filtered_by_name = [mm for mm in messages if mm.get("member") and any(n.lower() in mm["member"].lower() for n in q_names)]
        if filtered_by_name:
            candidate_msgs = filtered_by_name
    if asked_london:
        filtered_by_city = [mm for mm in candidate_msgs if "london" in (mm.get("text") or "").lower()]
        if filtered_by_city:
            candidate_msgs = filtered_by_city

    def bonus_score(mm: Dict[str, Any]) -> int:
        text = (mm.get("text") or "")
        t = text.lower()
        base = score_message(text, question)
        date_present = 1 if try_extract_date(text) else 0
        travel_present = 1 if ("plan" in t or "planning" in t or "trip" in t or "travel" in t or "visit" in t or "fly" in t or "flying" in t) else 0
        london_present = 1 if ("london" in t) else 0
        return base + (5 * date_present if wants_when else 0) + (2 * london_present if asked_london else 0) + (2 * travel_present if travel_words else 0)

    ranked = sorted(candidate_msgs, key=bonus_score, reverse=True)
    if not ranked:
        return JSONResponse({"answer": "I couldn't find an answer in the member messages.", "matched_message": None})

    def first_with_date(items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for mm in items:
            if try_extract_date(mm.get("text") or ""):
                return mm
        return None

    top = ranked[0]
    top_text = top.get("text") or ""
    answer: Optional[str] = None
    ev: List[Dict[str, Any]] = [{"text": (mm.get("text") or ""), "member": mm.get("member")} for mm in ranked[:5]]

    if wants_when:
        dt = try_extract_date(top_text)
        if dt:
            answer = dt
        else:
            alt = first_with_date(ranked[1:15])
            if alt:
                top = alt
                top_text = alt.get("text") or ""
                answer = try_extract_date(top_text)

    if not answer:
        if "how many" in q_lower or "number" in q_lower or "cars" in q_lower:
            num = try_extract_number(top_text)
            if num:
                answer = num
    if not answer and ("favorite" in q_lower or "favourite" in q_lower):
        answer = top_text
    if not answer:
        if wants_when:
            answer = "No explicit date mentioned. Closest relevant message: " + top_text
        else:
            answer = top_text or "I couldn't extract an answer from the messages."

    return {"answer": answer, "matched_message": top.get("raw"), "evidence": ev}
