"""
bot.py — Vera AI Bot server.

Implements all 5 required endpoints for the magicpin AI Challenge:
  POST /v1/context
  POST /v1/tick
  POST /v1/reply
  GET  /v1/healthz
  GET  /v1/metadata

Run:  uvicorn bot:app --host 0.0.0.0 --port 8080
Env:  GEMINI_API_KEY=<your key>
"""

import os
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import composer as composer_module

# ─── Setup ────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("vera-bot")

app = FastAPI(
    title="Vera AI Bot",
    description="magicpin merchant engagement AI — challenge submission",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()

# ─── In-memory state ──────────────────────────────────────────────────────────

# (scope, context_id) → {version, payload, stored_at}
contexts: dict[tuple[str, str], dict] = {}

# conversation_id → {merchant_id, customer_id, turns: [{from, body, ts}], suppression_keys: set}
conversations: dict[str, dict] = {}

# suppression_key → True (global dedup)
sent_suppression_keys: set[str] = set()

VALID_SCOPES = {"category", "merchant", "customer", "trigger"}

# ─── Models ───────────────────────────────────────────────────────────────────

class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = Field(default_factory=list)


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _get_payload(scope: str, context_id: str) -> Optional[dict]:
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None


def _counts() -> dict:
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        if scope in counts:
            counts[scope] += 1
    return counts


def _resolve_trigger_contexts(trigger: dict) -> tuple[Optional[dict], Optional[dict], Optional[dict]]:
    """Resolve category, merchant, customer from a trigger payload."""
    merchant_id = trigger.get("merchant_id") or trigger.get("payload", {}).get("merchant_id")
    customer_id = trigger.get("customer_id") or trigger.get("payload", {}).get("customer_id")

    merchant = _get_payload("merchant", merchant_id) if merchant_id else None
    if not merchant:
        # Try searching all merchants for this id
        for (scope, mid), entry in contexts.items():
            if scope == "merchant" and mid == merchant_id:
                merchant = entry["payload"]
                break

    category = None
    if merchant:
        cat_slug = merchant.get("category_slug")
        if cat_slug:
            category = _get_payload("category", cat_slug)

    customer = _get_payload("customer", customer_id) if customer_id else None

    return category, merchant, customer


def _is_trigger_expired(trigger: dict) -> bool:
    expires_at = trigger.get("expires_at")
    if not expires_at:
        return False
    try:
        exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        return datetime.now(timezone.utc) > exp
    except Exception:
        return False


def _conversation_bodies(conv_id: str) -> list[str]:
    conv = conversations.get(conv_id, {})
    return [t.get("body", "") for t in conv.get("turns", []) if t.get("from") == "vera"]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": _counts(),
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Vera Challenger",
        "team_members": ["Antigravity"],
        "model": "gemini-2.5-flash",
        "approach": (
            "Trigger-kind routing with per-kind LLM prompt templates. "
            "Auto-reply detection, intent routing, anti-repetition, "
            "language-aware composition (Hindi-English mix). "
            "Temperature=0 for determinism."
        ),
        "contact_email": "challenger@example.com",
        "version": "1.0.0",
        "submitted_at": "2026-07-07T05:00:00Z",
    }


@app.post("/v1/context")
async def push_context(body: ContextBody):
    if body.scope not in VALID_SCOPES:
        return JSONResponse(
            status_code=400,
            content={"accepted": False, "reason": "invalid_scope", "details": f"scope must be one of {list(VALID_SCOPES)}"},
        )

    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] >= body.version:
        return JSONResponse(
            status_code=409,
            content={"accepted": False, "reason": "stale_version", "current_version": cur["version"]},
        )

    stored_at = _now_iso()
    contexts[key] = {
        "version": body.version,
        "payload": body.payload,
        "stored_at": stored_at,
    }
    logger.info(f"Context stored: {body.scope}/{body.context_id} v{body.version}")

    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": stored_at,
    }


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []
    now_str = body.now

    for trg_id in body.available_triggers:
        trigger = _get_payload("trigger", trg_id)
        if not trigger:
            logger.warning(f"Trigger not found in context store: {trg_id}")
            continue

        # Skip expired triggers
        if _is_trigger_expired(trigger):
            logger.info(f"Trigger expired, skipping: {trg_id}")
            continue

        # Skip already-sent suppression keys
        supp_key = trigger.get("suppression_key", "")
        if supp_key and supp_key in sent_suppression_keys:
            logger.info(f"Suppression key already sent, skipping: {supp_key}")
            continue

        # Resolve contexts
        category, merchant, customer = _resolve_trigger_contexts(trigger)
        if not merchant:
            logger.warning(f"No merchant found for trigger {trg_id}")
            continue
        if not category:
            logger.warning(f"No category found for merchant {trigger.get('merchant_id')}")
            continue

        merchant_id = trigger.get("merchant_id") or merchant.get("merchant_id", "")
        customer_id = trigger.get("customer_id")

        # Generate unique conversation ID
        conv_id = f"conv_{merchant_id}_{trg_id}_{uuid.uuid4().hex[:8]}"

        logger.info(f"Composing for trigger {trg_id} (kind={trigger.get('kind')}, merchant={merchant_id})")

        try:
            result = composer_module.compose(category, merchant, trigger, customer)
        except Exception as e:
            logger.error(f"Composition error for {trg_id}: {e}")
            continue

        body_text = result.get("body", "").strip()
        if not body_text:
            logger.warning(f"Empty body from composer for {trg_id}")
            continue

        # Anti-repetition: check if we've sent this body before
        # (across all conversations for this merchant)
        is_repeat = False
        for c_id, conv in conversations.items():
            if conv.get("merchant_id") == merchant_id:
                if body_text in _conversation_bodies(c_id):
                    is_repeat = True
                    break
        if is_repeat:
            logger.info(f"Anti-repetition: skipping duplicate body for {merchant_id}")
            continue

        # Store conversation
        conversations[conv_id] = {
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "trigger_id": trg_id,
            "turns": [{"from": "vera", "body": body_text, "ts": now_str}],
            "suppression_keys": {supp_key} if supp_key else set(),
        }

        # Mark suppression key
        if supp_key:
            sent_suppression_keys.add(supp_key)

        send_as = result.get("send_as", "vera")
        cta = result.get("cta", "open_ended")

        # Build template params (name + up to 2 key facts from body)
        merchant_name = merchant.get("identity", {}).get("name", "")
        template_params = [merchant_name, trigger.get("kind", ""), body_text[:50]]

        action = {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": send_as,
            "trigger_id": trg_id,
            "template_name": f"vera_{trigger.get('kind', 'generic')}_v1",
            "template_params": template_params,
            "body": body_text,
            "cta": cta,
            "suppression_key": supp_key,
            "rationale": result.get("rationale", ""),
        }
        actions.append(action)

        logger.info(f"Action added: conv={conv_id}, kind={trigger.get('kind')}, cta={cta}")

    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv_id = body.conversation_id
    merchant_id = body.merchant_id
    customer_id = body.customer_id
    message = body.message
    turn_number = body.turn_number

    logger.info(f"Reply received: conv={conv_id}, turn={turn_number}, from={body.from_role}, msg={message[:80]!r}")

    # Store incoming message
    conv = conversations.setdefault(conv_id, {
        "merchant_id": merchant_id,
        "customer_id": customer_id,
        "trigger_id": None,
        "turns": [],
        "suppression_keys": set(),
    })
    conv["turns"].append({
        "from": body.from_role,
        "body": message,
        "msg": message,
        "ts": body.received_at,
    })

    # Resolve contexts
    merchant = None
    if merchant_id:
        merchant = _get_payload("merchant", merchant_id)
    if not merchant:
        # Try to find it
        for (scope, mid), entry in contexts.items():
            if scope == "merchant":
                merchant = entry["payload"]
                break

    customer = _get_payload("customer", customer_id) if customer_id else None

    category = None
    if merchant:
        cat_slug = merchant.get("category_slug")
        if cat_slug:
            category = _get_payload("category", cat_slug)
    if not category:
        # Use first available category
        for (scope, _), entry in contexts.items():
            if scope == "category":
                category = entry["payload"]
                break
    if not category:
        category = {"slug": "general", "voice": {}, "display_name": "General"}

    # Compose reply
    try:
        result = composer_module.compose_reply(
            category=category,
            merchant=merchant or {},
            customer=customer,
            conversation_history=conv["turns"],
            new_message=message,
            turn_number=turn_number,
        )
    except Exception as e:
        logger.error(f"Reply composition error for {conv_id}: {e}")
        result = {
            "action": "send",
            "body": "Got it — let me follow up on that right away.",
            "cta": "open_ended",
            "rationale": f"Fallback reply due to error: {str(e)[:80]}",
        }

    action = result.get("action", "send")

    if action == "send":
        body_text = result.get("body", "").strip()
        if not body_text:
            body_text = "Understood — I'll get right on that."

        # Anti-repetition: don't send same body twice in this conversation
        prev_bodies = _conversation_bodies(conv_id)
        if body_text in prev_bodies:
            # Slightly modify it
            body_text = body_text + " 🙏"

        # Store Vera's reply
        conv["turns"].append({
            "from": "vera",
            "body": body_text,
            "ts": body.received_at,
        })

        response = {
            "action": "send",
            "body": body_text,
            "cta": result.get("cta", "open_ended"),
            "rationale": result.get("rationale", ""),
        }

    elif action == "wait":
        response = {
            "action": "wait",
            "wait_seconds": result.get("wait_seconds", 1800),
            "rationale": result.get("rationale", "Backing off as requested."),
        }

    else:  # end
        # Store graceful exit
        conv["turns"].append({
            "from": "vera",
            "body": "[conversation ended]",
            "ts": body.received_at,
        })
        response = {
            "action": "end",
            "rationale": result.get("rationale", "Gracefully ending conversation."),
        }

    logger.info(f"Reply sent: conv={conv_id}, action={action}")
    return response


@app.post("/v1/teardown")
async def teardown():
    """Optional: wipe all state at end of test (per spec §11)."""
    contexts.clear()
    conversations.clear()
    sent_suppression_keys.clear()
    logger.info("Teardown complete — all state cleared.")
    return {"status": "cleared"}


# ─── Root ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "Vera AI Bot",
        "version": "1.0.0",
        "endpoints": [
            "POST /v1/context",
            "POST /v1/tick",
            "POST /v1/reply",
            "GET /v1/healthz",
            "GET /v1/metadata",
        ],
        "status": "live",
    }
