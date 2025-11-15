# Member Messages QA Service

Ask natural-language questions about a stream of **member messages** and get precise answers (dates, counts, or the most relevant message). Built with **FastAPI** and deployable on **Render**.

<p align="center">
  <a href="https://membermessageqaservice.onrender.com/docs"><img alt="Swagger UI" src="https://img.shields.io/badge/docs-Swagger%20UI-3fb950"></a>
  <a href="https://membermessageqaservice.onrender.com/healthz"><img alt="Health" src="https://img.shields.io/badge/health-OK-brightgreen"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
</p>

## Live Links
- **API Docs (Swagger UI):** <https://membermessageqaservice.onrender.com/docs>  
- **Health Check:** <https://membermessageqaservice.onrender.com/healthz>  
- **Upstream Messages API (used by this service):** <https://november7-730026606190.europe-west1.run.app/messages>


## Features
- Natural-language Q&A over messages
- **Date extraction:** full month names, ISO `YYYY-MM-DD`, relative terms (“tonight”, “next week”)
- **Number extraction:** digits and number-words (“two”, “three”)
- **Noise control:** filters by person **name**, **city** (diacritic-safe), and intent (meeting/tickets/travel)
- Debug endpoints to verify upstream connectivity and normalization

