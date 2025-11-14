import re
import time
from typing import Any, Dict, Optional, List

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

MESSAGES_URL = "https://november7-730026606190.europe-west1.run.app/messages"

app = FastAPI(
    title="Member Messages QA Service",
    description="Ask natural-language questions about member messages.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


CACHE: Dict[str, Any] = {"data": None, "ts": 0.0}
CACHE_TTL_SEC = 180  

class AskBody(BaseModel):
    question: str = Field(..., description="Natural-language question to answer")
    model_config = {
        "json_schema_extra": {
            "example": {"question": "When is Layla planning her trip to London?"}
        }
    }

class AskResponse(BaseModel):
    answer: str
    matched_message: Optional[Dict[str, Any]] = None

async def fetch_messages_with_retry() -> List[Dict[str, Any]]:
    """Try upstream up to 3 times; on total failure, use warm cache if present."""
    attempts = 3
    last_exc: Optional[Exception] = None

    for i in range(attempts):
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(MESSAGES_URL)
                if resp.status_code == 200:
                    data = resp.json()
                  
                    CACHE["data"] = data
                    CACHE["ts"] = time.time()
                    return data
                else:
                    last_exc = HTTPException(status_code=502, detail=f"Upstream returned {resp.status_code}")
        except httpx.RequestError as e:
            last_exc = e

        
        time.sleep(0.5 + 0.5 * i)

    if CACHE["data"] and (time.time() - CACHE["ts"] <= CACHE_TTL_SEC):
        return CACHE["data"]

    
    if isinstance(last_exc, HTTPException):
        raise last_exc
    raise HTTPException(status_code=502, detail="Upstream messages API unreachable or timed out")

def find_name_in_question(q: str):
    tokens = q.strip().split()
    candidates = []
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

def try_extract_date(text: str):
    m = re.search(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,\s*\d{4})?",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(0)
    m2 = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
    if m2:
        return m2.group(0)
    return None

def try_extract_number(text: str):
    m = re.search(r"\b\d+\b", text)
    if m:
        return m.group(0)
    return None

@app.get("/")
async def root():
    return {"status": "ok", "docs": "/docs"}

@app.get("/healthz")
async def healthz():
    
    age = time.time() - CACHE["ts"] if CACHE["ts"] else None
    return {"ok": True, "cache_age_sec": age}

@app.get("/debug/upstream")
async def debug_upstream():
    """Fetch upstream once (no cache) to see current status, for debugging."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(MESSAGES_URL)
            return {"status_code": resp.status_code, "ok": resp.status_code == 200}
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {e!s}")

@app.post("/ask", response_model=AskResponse)
async def ask(body: AskBody):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Field 'question' must be a non-empty string.")

    messages = await fetch_messages_with_retry()

    normalized = []
    for m in messages:
        text = m.get("message") or m.get("text") or m.get("content") or str(m)
        member = m.get("member_name") or m.get("member") or m.get("name") or m.get("user")
        normalized.append({"raw": m, "text": text, "member": member})

    q_names = find_name_in_question(question)
    if q_names:
        filtered = [
            mm for mm in normalized
            if mm["member"] and any(n.lower() in mm["member"].lower() for n in q_names)
        ]
        candidate_msgs = filtered or normalized
    else:
        candidate_msgs = normalized

    ranked = sorted(candidate_msgs, key=lambda mm: score_message(mm["text"], question), reverse=True)
    if not ranked:
        return JSONResponse({"answer": "I couldn't find an answer in the member messages.", "matched_message": None})

    top = ranked[0]
    q_lower = question.lower()
    answer: Optional[str] = None

    if ("when" in q_lower) or ("date" in q_lower) or ("planning" in q_lower):
        date = try_extract_date(top["text"])
        if date:
            answer = date
    elif ("how many" in q_lower) or ("number" in q_lower) or ("cars" in q_lower):
        num = try_extract_number(top["text"])
        if num:
            answer = num
    elif ("favorite" in q_lower) or ("favourite" in q_lower):
        answer = top["text"]

    if not answer:
        answer = top["text"]

    return {"answer": answer, "matched_message": top["raw"]}
