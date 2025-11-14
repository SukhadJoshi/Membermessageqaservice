-- Member Messages QA Service

Ask natural-language questions about a stream of “member messages” and get precise answers (dates, counts, or the most relevant message), with strict filtering by person name, city, and intent (e.g., meeting/tickets). Built with FastAPI and deployable to Render.


-- Live URLs (example)

Your service (FastAPI + Swagger UI):
https://membermessageqaservice.onrender.com/docs

Health check:
https://membermessageqaservice.onrender.com/healthz

Upstream messages API used by this service:
https://november7-730026606190.europe-west1.run.app/messages



-- Features

Natural-language Q&A over messages

Date extraction: full month names, ISO (YYYY-MM-DD), and relative (“tonight”, “next week”)

Number extraction: digits and number-words (“two”, “three”)

Strict gating by name, city (diacritic-safe), and intent (meeting/tickets/travel) to reduce irrelevant answers

Helpful debug endpoints to verify upstream connectivity and normalization
