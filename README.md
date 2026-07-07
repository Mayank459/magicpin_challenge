# Vera AI Bot — Submission README

## Team
Solo submission — built by Antigravity

## Model
Google Gemini 1.5 Flash (temperature=0 for determinism, `application/json` response type)

## Approach

### Core Architecture
Single FastAPI server with 5 endpoints. All context stored in-memory per session. Stateful conversation tracking.

### 4-Context Handling
- `POST /v1/context` stores category, merchant, customer, trigger separately with version control
- At `/v1/tick`, contexts are assembled per trigger: `(category, merchant, trigger, customer?)` → `compose()`
- Full context payload is passed to the LLM — no truncation

### LLM Composition Strategy
**Trigger-kind routing**: 17 distinct trigger kinds each have their own prompt template:
- `research_digest` → clinical/peer framing with source citation
- `regulation_change` → urgency + loss aversion + deadline
- `recall_due` → customer-facing, slot-offering, Hindi-English
- `perf_dip` → loss aversion + social proof
- `competitor_opened` → curiosity + competitive framing
- `active_planning_intent` → immediate action (no re-qualifying)
- ...and 11 more

**Compulsion levers used per message (from brief §10)**:
- Always: Specificity (numbers, dates, sources)
- Per-kind: Loss aversion, Social proof, Effort externalization, Curiosity

### Multi-Turn Handling
- **Auto-reply detection**: Regex patterns for 10+ common WA Business auto-reply templates. On detection: try once more (different angle), then gracefully exit.
- **Intent routing**: When merchant says "yes"/"ok"/"go ahead" → switch immediately to action mode, no re-qualifying.
- **Exit detection**: On "not interested"/"stop"/"band karo" → graceful exit with warm closing.
- **Anti-repetition**: Tracks all sent message bodies per merchant; never sends exact duplicate.

### Language Matching
Checks `merchant.identity.languages`. If `"hi"` is present → Hindi-English Hinglish mix. Regional languages (te/kn/ta/mr) → English with warmth.

### What I'd improve with more time
1. **Retrieval**: Embed digest items and retrieve only the most relevant 2-3 per compose call (reduces token waste)
2. **Semantic auto-reply detection**: Use LLM to classify instead of regex
3. **Conversation memory persistence**: Redis instead of in-memory
4. **A/B variant tracking**: Log which prompt variant produced each message for offline analysis
5. **Rate limiting**: Backpressure handling for high-volume tick scenarios

## Public URL
See deployment instructions below.

## Deployment

### Option 1: Render.com (recommended)
1. Connect this repo to Render
2. Set `GEMINI_API_KEY` environment variable
3. Deploy — auto-detects `render.yaml`

### Option 2: Local + ngrok
```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set GEMINI_API_KEY
uvicorn bot:app --host 0.0.0.0 --port 8080
# In another terminal:
ngrok http 8080
```

### Option 3: Railway.app
```bash
railway init
railway add
railway up
```

## Testing locally
```bash
export BOT_URL=http://localhost:8080
python ../task\ plan/judge_simulator.py
```
