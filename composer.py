"""
composer.py — LLM-based message composer for Vera bot.

Dispatches on trigger.kind → selects prompt template → calls Gemini API.
All calls use temperature=0 for determinism.
"""

import os
import json
import hashlib
import re
from typing import Optional
import google.generativeai as genai

# Configure Gemini
_api_key = os.getenv("GEMINI_API_KEY", "")
if _api_key:
    genai.configure(api_key=_api_key)

MODEL_NAME = "gemini-1.5-flash"

# Auto-reply patterns (common WhatsApp Business auto-replies)
AUTO_REPLY_PATTERNS = [
    r"thank\s*you\s*for\s*contact",
    r"aapki\s*jaankari\s*ke\s*liye.*shukriya",
    r"automated\s*assistant",
    r"main\s*ek\s*automated",
    r"i am an automated",
    r"we will get back",
    r"our team will",
    r"hamare team",
    r"madad ke liye shukriya",
    r"for (?:more )?info(?:rmation)?,?\s*please (?:visit|contact|call)",
]

# Phrases indicating merchant wants to disengage
EXIT_PATTERNS = [
    r"not interested",
    r"nahi chahiye",
    r"band karo",
    r"stop",
    r"unsubscribe",
    r"remove me",
    r"don't message",
    r"mat karo",
    r"mujhe nahi",
    r"no thanks",
]

# Phrases indicating merchant is ready to act
INTENT_ACTION_PATTERNS = [
    r"\byes\b",
    r"\bhaan\b",
    r"\bha\b",
    r"let's do it",
    r"go ahead",
    r"kar do",
    r"kar dena",
    r"ok please",
    r"bilkul",
    r"zaroor",
    r"send me",
    r"share karo",
    r"theek hai",
    r"acha",
]


def detect_auto_reply(message: str) -> bool:
    msg_lower = message.lower()
    return any(re.search(p, msg_lower) for p in AUTO_REPLY_PATTERNS)


def detect_exit_intent(message: str) -> bool:
    msg_lower = message.lower()
    return any(re.search(p, msg_lower) for p in EXIT_PATTERNS)


def detect_action_intent(message: str) -> bool:
    msg_lower = message.lower()
    return any(re.search(p, msg_lower) for p in INTENT_ACTION_PATTERNS)


def _lang_instruction(merchant: dict) -> str:
    langs = merchant.get("identity", {}).get("languages", ["en"])
    if "hi" in langs:
        return (
            "Use natural Hindi-English code-mix (Hinglish). "
            "Short Hindi phrases are welcome: 'Kya aap chahenge?', 'Main kar sakti hoon', etc. "
            "Match the merchant's casual bilingual style."
        )
    elif "te" in langs or "kn" in langs or "ta" in langs or "mr" in langs:
        return "Use English primarily; occasional regional warmth is fine. Keep professional."
    return "Use clear English."


def _voice_instruction(category: dict) -> str:
    voice = category.get("voice", {})
    tone = voice.get("tone", "professional")
    taboos = voice.get("vocab_taboo", [])
    taboo_str = ", ".join(f'"{t}"' for t in taboos[:5]) if taboos else "none"
    return (
        f"Tone: {tone}. "
        f"Taboo words (never use): {taboo_str}. "
        "Use service+price format (e.g., 'Haircut @ ₹99'), not generic discounts ('20% off'). "
        "Peer/colleague voice, not promotional."
    )


def _merchant_name(merchant: dict) -> str:
    return merchant.get("identity", {}).get("name", "there")


def _owner_name(merchant: dict) -> str:
    fname = merchant.get("identity", {}).get("owner_first_name", "")
    name = merchant.get("identity", {}).get("name", "")
    if fname:
        return fname
    # Try to extract first word
    return name.split()[0] if name else "there"


def _build_system_prompt() -> str:
    return """You are Vera, magicpin's merchant AI assistant. You talk to Indian merchants over WhatsApp.

Rules:
- Be concise. No long preambles. No "I hope you're doing well."
- Do NOT re-introduce yourself after the first message.
- Use ONE primary CTA at the end (Reply YES / Reply STOP, or a single open question).
- Never use taboo words from the category voice profile.
- Never hallucinate data not in the context. If unsure, skip that detail.
- Use specific numbers, dates, citations — "6,777 searches", "38% better", "JIDA Oct p.14".
- For merchant-facing: peer/colleague tone, not promotional.
- For customer-facing (send_as = merchant_on_behalf): warm, clinical, from merchant's voice.
- Length: 2-4 sentences is ideal. Max 6 sentences.
- CTA must be the LAST sentence.
- Return ONLY a JSON object with these keys:
  body, cta, send_as, suppression_key, rationale

cta values: "yes_stop" | "open_ended" | "none"
send_as values: "vera" | "merchant_on_behalf"
"""


def _call_llm(prompt: str) -> dict:
    """Call Gemini with the prompt and parse JSON response."""
    if not _api_key:
        # Fallback if no API key
        return {
            "body": "Hi, Vera here with an update for your business.",
            "cta": "open_ended",
            "send_as": "vera",
            "suppression_key": "fallback",
            "rationale": "No API key configured — fallback response",
        }

    try:
        model = genai.GenerativeModel(
            MODEL_NAME,
            system_instruction=_build_system_prompt(),
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        response = model.generate_content(prompt)
        text = response.text.strip()
        # Strip markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return json.loads(text)
    except Exception as e:
        return {
            "body": f"Vera here — let's connect on an update for your business.",
            "cta": "open_ended",
            "send_as": "vera",
            "suppression_key": "error_fallback",
            "rationale": f"LLM error: {str(e)[:100]}",
        }


# ─── Prompt builders per trigger kind ────────────────────────────────────────

def _prompt_research_digest(category: dict, merchant: dict, trigger: dict) -> str:
    merchant_name = _owner_name(merchant)
    cat_name = category.get("display_name", category.get("slug", ""))
    digest = category.get("digest", [])
    top_item_id = trigger.get("payload", {}).get("top_item_id", "")
    top_item = next((d for d in digest if d.get("id") == top_item_id), digest[0] if digest else {})

    peer_stats = category.get("peer_stats", {})
    merchant_ctr = merchant.get("performance", {}).get("ctr", None)
    peer_ctr = peer_stats.get("avg_ctr", None)
    ctr_below = merchant_ctr and peer_ctr and merchant_ctr < peer_ctr

    customer_agg = merchant.get("customer_aggregate", {})
    signals = merchant.get("signals", [])

    return f"""Compose a WhatsApp message from Vera to {merchant_name} (category: {cat_name}).

TRIGGER KIND: research_digest — a new research/digest item just dropped.

RESEARCH ITEM:
{json.dumps(top_item, indent=2, ensure_ascii=False)}

MERCHANT STATE:
- Name: {merchant.get('identity', {}).get('name')}
- CTR: {merchant_ctr} (peer median: {peer_ctr}) {'← BELOW PEER' if ctr_below else ''}
- Customer aggregate: {json.dumps(customer_agg, ensure_ascii=False)}
- Signals: {signals}
- Last conversation: {json.dumps(merchant.get('conversation_history', [])[-2:], ensure_ascii=False)}

VOICE: {_voice_instruction(category)}
LANGUAGE: {_lang_instruction(merchant)}

COMPULSION LEVERS TO USE (pick 2+):
1. Specificity — cite the exact stat/source from the research item
2. Merchant fit — reference their specific patient cohort or signals
3. Curiosity — "want me to pull it + draft a patient-ed message?"
4. Effort externalization — "I've drafted it — just say go"

suppression_key must be: {trigger.get('suppression_key', 'research:default')}
send_as: "vera"

Return JSON only."""


def _prompt_regulation_change(category: dict, merchant: dict, trigger: dict) -> str:
    merchant_name = _owner_name(merchant)
    digest = category.get("digest", [])
    top_item_id = trigger.get("payload", {}).get("top_item_id", "")
    item = next((d for d in digest if d.get("id") == top_item_id), {})
    deadline = trigger.get("payload", {}).get("deadline_iso", "")

    return f"""Compose a WhatsApp message from Vera to {merchant_name} about a regulatory/compliance change.

TRIGGER KIND: regulation_change — compliance update, deadline approaching.

COMPLIANCE ITEM:
{json.dumps(item, indent=2, ensure_ascii=False)}

DEADLINE: {deadline}

MERCHANT: {merchant.get('identity', {}).get('name')} in {merchant.get('identity', {}).get('city')}
SIGNALS: {merchant.get('signals', [])}

VOICE: {_voice_instruction(category)}
LANGUAGE: {_lang_instruction(merchant)}

COMPULSION LEVERS:
1. Loss aversion — specific deadline, what happens if missed
2. Specificity — exact regulation name, date, what changes
3. Effort externalization — "I can help you audit in 5 min"

CTA: Single YES/STOP binary.
suppression_key: {trigger.get('suppression_key', 'compliance:default')}
send_as: "vera"

Return JSON only."""


def _prompt_recall_due(category: dict, merchant: dict, trigger: dict, customer: Optional[dict]) -> str:
    merchant_name = merchant.get("identity", {}).get("name", "")
    cust_name = customer.get("identity", {}).get("name", "there") if customer else "Patient"
    cust_lang = customer.get("identity", {}).get("language_pref", "en") if customer else "en"
    state = customer.get("state", "lapsed_soft") if customer else "lapsed_soft"
    prefs = customer.get("preferences", {}) if customer else {}
    payload = trigger.get("payload", {})
    slots = payload.get("available_slots", [])
    slots_str = " ya ".join(s.get("label", "") for s in slots[:2]) if slots else "next available slot"
    active_offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
    offer_str = active_offers[0].get("title", "") if active_offers else ""

    lang_note = "hi-en mix (Hinglish)" if "hi" in (cust_lang or "") else "English"

    return f"""Compose a WhatsApp message ON BEHALF OF the merchant to their customer.

TRIGGER KIND: recall_due — customer's service recall window just opened.

CUSTOMER:
- Name: {cust_name}
- State: {state}
- Language pref: {cust_lang} → use {lang_note}
- Preferred slots: {prefs.get('preferred_slots', 'flexible')}

MERCHANT (sender):
- Name: {merchant_name}
- Active offer: {offer_str}
- Available slots: {slots_str}

TRIGGER PAYLOAD:
{json.dumps(payload, indent=2, ensure_ascii=False)}

RULES:
- send_as = "merchant_on_behalf" (message comes FROM the merchant's number)
- Warm but clinical — no overclaims, no "guaranteed"
- Include specific slot options
- Include service price if known
- Language: {lang_note}

CTA: slot booking choice (Reply 1/2 or tell us a time)
suppression_key: {trigger.get('suppression_key', 'recall:default')}

Return JSON only."""


def _prompt_perf_dip(category: dict, merchant: dict, trigger: dict) -> str:
    merchant_name = _owner_name(merchant)
    payload = trigger.get("payload", {})
    metric = payload.get("metric", "views")
    delta = payload.get("delta_pct", 0)
    delta_pct = int(abs(delta) * 100)
    peer_stats = category.get("peer_stats", {})

    return f"""Compose a WhatsApp message from Vera to {merchant_name} about a performance dip.

TRIGGER KIND: perf_dip — their {metric} dropped {delta_pct}% this week.

MERCHANT PERFORMANCE:
{json.dumps(merchant.get('performance', {}), indent=2, ensure_ascii=False)}

PEER STATS (benchmark):
{json.dumps(peer_stats, indent=2, ensure_ascii=False)}

SIGNALS: {merchant.get('signals', [])}
ACTIVE OFFERS: {[o['title'] for o in merchant.get('offers', []) if o.get('status') == 'active']}

VOICE: {_voice_instruction(category)}
LANGUAGE: {_lang_instruction(merchant)}

COMPULSION LEVERS:
1. Loss aversion — what they're missing (specific number)
2. Social proof — what peers are doing
3. Effort externalization — specific fix I can do right now

CTA: YES/STOP binary.
suppression_key: {trigger.get('suppression_key', 'perf_dip:default')}
send_as: "vera"

Return JSON only."""


def _prompt_perf_spike(category: dict, merchant: dict, trigger: dict) -> str:
    merchant_name = _owner_name(merchant)
    payload = trigger.get("payload", {})
    metric = payload.get("metric", "views")
    delta = payload.get("delta_pct", 0)
    delta_pct = int(abs(delta) * 100)
    driver = payload.get("likely_driver", "")

    return f"""Compose a WhatsApp message from Vera to {merchant_name} about a positive performance spike.

TRIGGER KIND: perf_spike — their {metric} jumped {delta_pct}% this week. Likely driver: {driver}.

MERCHANT PERFORMANCE:
{json.dumps(merchant.get('performance', {}), indent=2, ensure_ascii=False)}

VOICE: {_voice_instruction(category)}
LANGUAGE: {_lang_instruction(merchant)}

COMPULSION LEVERS:
1. Reciprocity — I noticed this, thought you'd want to know
2. Curiosity — "want to see what's driving it?"
3. Momentum — "let's lock in this growth"

CTA: open_ended (no binary here — this is celebratory + curious)
suppression_key: {trigger.get('suppression_key', 'perf_spike:default')}
send_as: "vera"

Return JSON only."""


def _prompt_renewal_due(category: dict, merchant: dict, trigger: dict) -> str:
    merchant_name = _owner_name(merchant)
    payload = trigger.get("payload", {})
    days = payload.get("days_remaining", 14)
    plan = payload.get("plan", "Pro")
    amount = payload.get("renewal_amount", "")
    perf = merchant.get("performance", {})

    return f"""Compose a WhatsApp message from Vera to {merchant_name} about subscription renewal.

TRIGGER KIND: renewal_due — {days} days left on their {plan} plan.

MERCHANT STATE:
- Performance (30d): views={perf.get('views')}, calls={perf.get('calls')}, CTR={perf.get('ctr')}
- Signals: {merchant.get('signals', [])}
- Renewal amount: ₹{amount if amount else 'as per plan'}

VOICE: {_voice_instruction(category)}
LANGUAGE: {_lang_instruction(merchant)}

COMPULSION LEVERS:
1. Loss aversion — what stops when subscription lapses (profile visibility, active offers)
2. Specificity — their actual numbers that will be lost
3. Single binary CTA

CTA: "Reply YES to renew" / STOP
suppression_key: {trigger.get('suppression_key', 'renewal:default')}
send_as: "vera"

Return JSON only."""


def _prompt_milestone_reached(category: dict, merchant: dict, trigger: dict) -> str:
    merchant_name = _owner_name(merchant)
    payload = trigger.get("payload", {})
    metric = payload.get("metric", "review_count")
    value_now = payload.get("value_now", "")
    milestone = payload.get("milestone_value", "")
    imminent = payload.get("is_imminent", False)

    return f"""Compose a WhatsApp message from Vera to {merchant_name} about a milestone.

TRIGGER KIND: milestone_reached — they are {'approaching' if imminent else 'at'} {milestone} {metric}. Currently at {value_now}.

MERCHANT:
- Name: {merchant.get('identity', {}).get('name')}
- Signals: {merchant.get('signals', [])}

VOICE: {_voice_instruction(category)}
LANGUAGE: {_lang_instruction(merchant)}

COMPULSION LEVERS:
1. Social proof — what other merchants do at this milestone
2. Curiosity — "want to make the most of this milestone?"
3. Reciprocity — "I spotted this for you"

CTA: open_ended
suppression_key: {trigger.get('suppression_key', 'milestone:default')}
send_as: "vera"

Return JSON only."""


def _prompt_competitor_opened(category: dict, merchant: dict, trigger: dict) -> str:
    merchant_name = _owner_name(merchant)
    payload = trigger.get("payload", {})
    comp_name = payload.get("competitor_name", "a new competitor")
    dist_km = payload.get("distance_km", "")
    their_offer = payload.get("their_offer", "")
    active_offers = [o['title'] for o in merchant.get('offers', []) if o.get('status') == 'active']

    return f"""Compose a WhatsApp message from Vera to {merchant_name} about a new competitor.

TRIGGER KIND: competitor_opened — {comp_name} opened {dist_km}km away with offer: "{their_offer}".

MERCHANT:
- Active offers: {active_offers}
- Performance: {json.dumps(merchant.get('performance', {}), ensure_ascii=False)}
- Signals: {merchant.get('signals', [])}

VOICE: {_voice_instruction(category)}
LANGUAGE: {_lang_instruction(merchant)}

IMPORTANT: Do NOT name the competitor (privacy). Reference their offer tier only.
COMPULSION LEVERS:
1. Loss aversion — potential patient leakage
2. Curiosity — "want to see how you compare?"
3. Effort externalization — "I can update your GBP offer in 5 min"

CTA: YES/STOP binary
suppression_key: {trigger.get('suppression_key', 'competitor:default')}
send_as: "vera"

Return JSON only."""


def _prompt_festival_upcoming(category: dict, merchant: dict, trigger: dict) -> str:
    merchant_name = _owner_name(merchant)
    payload = trigger.get("payload", {})
    festival = payload.get("festival", "upcoming festival")
    days_until = payload.get("days_until", "")
    active_offers = [o['title'] for o in merchant.get('offers', []) if o.get('status') == 'active']
    cat_offers = category.get("offer_catalog", [])[:3]

    return f"""Compose a WhatsApp message from Vera to {merchant_name} about an upcoming festival.

TRIGGER KIND: festival_upcoming — {festival} is {days_until} days away.

MERCHANT:
- Active offers: {active_offers}
- Category offers available: {[o.get('title') for o in cat_offers]}

VOICE: {_voice_instruction(category)}
LANGUAGE: {_lang_instruction(merchant)}

COMPULSION LEVERS:
1. Specificity — festival name, days until, specific offer to run
2. Social proof — "merchants who run festival offers see X% more footfall"
3. Effort externalization — "I can set it up in 2 min"

CTA: YES/STOP binary
suppression_key: {trigger.get('suppression_key', 'festival:default')}
send_as: "vera"

Return JSON only."""


def _prompt_curious_ask(category: dict, merchant: dict, trigger: dict) -> str:
    merchant_name = _owner_name(merchant)
    payload = trigger.get("payload", {})
    ask_template = payload.get("ask_template", "what_service_in_demand_this_week")

    questions_map = {
        "what_service_in_demand_this_week": "which service has been most in demand this week?",
        "what_customers_are_asking": "what are customers asking about most?",
        "what_would_help_most": "what would help your business most this month?",
    }
    question = questions_map.get(ask_template, "what's top of mind for your business this week?")

    return f"""Compose a short curiosity-driven WhatsApp message from Vera to {merchant_name}.

TRIGGER KIND: curious_ask_due — time to ask the merchant an engaging question (no agenda).

QUESTION TO ASK: {question}

MERCHANT:
- Name: {merchant.get('identity', {}).get('name')}
- Recent signals: {merchant.get('signals', [])}

VOICE: {_voice_instruction(category)}
LANGUAGE: {_lang_instruction(merchant)}

Keep this SHORT (1-2 sentences max). Casual, curious, peer-like. No sales pitch.
CTA: open_ended (the question IS the CTA)
suppression_key: {trigger.get('suppression_key', 'curious:default')}
send_as: "vera"

Return JSON only."""


def _prompt_winback(category: dict, merchant: dict, trigger: dict) -> str:
    merchant_name = _owner_name(merchant)
    payload = trigger.get("payload", {})
    days_since = payload.get("days_since_expiry", payload.get("days_since_last_merchant_message", ""))
    perf_dip = payload.get("perf_dip_pct", None)
    lapsed = payload.get("lapsed_customers_added_since_expiry", None)

    return f"""Compose a WhatsApp message from Vera to a dormant/lapsed merchant.

TRIGGER KIND: winback/dormant — merchant has been inactive for {days_since} days.

MERCHANT:
- Name: {merchant.get('identity', {}).get('name')}
- Subscription: {json.dumps(merchant.get('subscription', {}), ensure_ascii=False)}
- Performance dip since inactivity: {f'{int(abs(perf_dip)*100)}%' if perf_dip else 'not measured'}
- New lapsed customers since expiry: {lapsed if lapsed else 'unknown'}
- Signals: {merchant.get('signals', [])}

VOICE: {_voice_instruction(category)}
LANGUAGE: {_lang_instruction(merchant)}

COMPULSION LEVERS:
1. Loss aversion — what's been happening to their profile/customers while away
2. Reciprocity — "I noticed X about your account"
3. Low-friction re-entry — "one quick thing to restart"

CTA: YES/STOP binary
suppression_key: {trigger.get('suppression_key', 'winback:default')}
send_as: "vera"

Return JSON only."""


def _prompt_gbp_unverified(category: dict, merchant: dict, trigger: dict) -> str:
    merchant_name = _owner_name(merchant)
    payload = trigger.get("payload", {})
    uplift = payload.get("estimated_uplift_pct", 0.30)
    verification_path = payload.get("verification_path", "postcard")

    return f"""Compose a WhatsApp message from Vera about an unverified Google Business Profile.

TRIGGER KIND: gbp_unverified — merchant's GBP is not verified. Verified = ~{int(uplift*100)}% more visibility.

MERCHANT:
- Name: {merchant.get('identity', {}).get('name')}
- Performance: {json.dumps(merchant.get('performance', {}), ensure_ascii=False)}
- Verification method: {verification_path}

VOICE: {_voice_instruction(category)}
LANGUAGE: {_lang_instruction(merchant)}

COMPULSION LEVERS:
1. Loss aversion — {int(uplift*100)}% more views they're missing
2. Effort externalization — "I can walk you through verification in 5 min"
3. Specificity — their actual views number + uplift potential

CTA: YES/STOP binary
suppression_key: {trigger.get('suppression_key', 'gbp_unverified:default')}
send_as: "vera"

Return JSON only."""


def _prompt_cde_opportunity(category: dict, merchant: dict, trigger: dict) -> str:
    merchant_name = _owner_name(merchant)
    digest = category.get("digest", [])
    item_id = trigger.get("payload", {}).get("digest_item_id", "")
    item = next((d for d in digest if d.get("id") == item_id), {})
    credits = trigger.get("payload", {}).get("credits", 0)
    fee = trigger.get("payload", {}).get("fee", "")

    return f"""Compose a WhatsApp message from Vera about a continuing education / webinar opportunity.

TRIGGER KIND: cde_opportunity — relevant webinar/course for this merchant's category.

CDE ITEM:
{json.dumps(item, indent=2, ensure_ascii=False)}

Credits: {credits} CDE credits | Fee: {fee}

MERCHANT: {merchant.get('identity', {}).get('name')} in {merchant.get('identity', {}).get('city')}

VOICE: {_voice_instruction(category)}
LANGUAGE: {_lang_instruction(merchant)}

COMPULSION LEVERS:
1. Specificity — exact date, speaker, credits, fee
2. Social proof — "IDA members are registering"
3. Loss aversion — "closes soon"

CTA: open_ended ("Want the link?")
suppression_key: {trigger.get('suppression_key', 'cde:default')}
send_as: "vera"

Return JSON only."""


def _prompt_supply_alert(category: dict, merchant: dict, trigger: dict) -> str:
    merchant_name = _owner_name(merchant)
    payload = trigger.get("payload", {})
    molecule = payload.get("molecule", "")
    batches = payload.get("affected_batches", [])
    manufacturer = payload.get("manufacturer", "")

    return f"""Compose a WhatsApp message from Vera about an urgent supply/recall alert.

TRIGGER KIND: supply_alert — URGENT. Product recall affecting their inventory.

ALERT DETAILS:
- Molecule: {molecule}
- Affected batches: {batches}
- Manufacturer: {manufacturer}

MERCHANT: {merchant.get('identity', {}).get('name')}
SIGNALS: {merchant.get('signals', [])}

VOICE: {_voice_instruction(category)}
LANGUAGE: {_lang_instruction(merchant)}

This is URGENT (urgency 5). Be direct. Lead with the alert. 
COMPULSION LEVERS:
1. Loss aversion — legal/patient safety risk
2. Specificity — batch numbers
3. Effort externalization — "I can filter your customer list"

CTA: YES/STOP binary
suppression_key: {trigger.get('suppression_key', 'supply_alert:default')}
send_as: "vera"

Return JSON only."""


def _prompt_active_planning(category: dict, merchant: dict, trigger: dict) -> str:
    merchant_name = _owner_name(merchant)
    payload = trigger.get("payload", {})
    topic = payload.get("intent_topic", "")
    last_msg = payload.get("merchant_last_message", "")

    return f"""Compose a WhatsApp message from Vera continuing an active planning conversation.

TRIGGER KIND: active_planning_intent — merchant expressed intent to plan {topic}.

MERCHANT'S LAST MESSAGE: "{last_msg}"

MERCHANT:
- Name: {merchant.get('identity', {}).get('name')}
- Offers: {[o['title'] for o in merchant.get('offers', []) if o.get('status') == 'active']}
- Performance: {json.dumps(merchant.get('performance', {}), ensure_ascii=False)}

CATEGORY: {category.get('slug')}
VOICE: {_voice_instruction(category)}
LANGUAGE: {_lang_instruction(merchant)}

IMPORTANT: The merchant already said yes to the idea. Do NOT ask another qualifying question.
Provide a CONCRETE plan/draft immediately. Use specifics (price, timeline, format).

CTA: open_ended or "Reply YES to publish"
suppression_key: {trigger.get('suppression_key', 'planning:default')}
send_as: "vera"

Return JSON only."""


def _prompt_generic(category: dict, merchant: dict, trigger: dict, customer: Optional[dict]) -> str:
    merchant_name = _owner_name(merchant)
    kind = trigger.get("kind", "unknown")

    return f"""Compose a WhatsApp message from Vera to {merchant_name}.

TRIGGER KIND: {kind}
TRIGGER PAYLOAD: {json.dumps(trigger.get('payload', {}), indent=2, ensure_ascii=False)}

MERCHANT:
{json.dumps({k: v for k, v in merchant.items() if k != 'conversation_history'}, indent=2, ensure_ascii=False)}

CATEGORY: {category.get('slug')} ({category.get('display_name', '')})
VOICE: {_voice_instruction(category)}
LANGUAGE: {_lang_instruction(merchant)}

{"CUSTOMER:" + json.dumps(customer, indent=2, ensure_ascii=False) if customer else ""}

Follow Vera's rules: specific, concise, single CTA, no hallucination.
suppression_key: {trigger.get('suppression_key', f'{kind}:default')}
send_as: "{'merchant_on_behalf' if customer else 'vera'}"

Return JSON only."""


# ─── Main dispatch ────────────────────────────────────────────────────────────

def compose(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
) -> dict:
    """
    Main composition entry point. Dispatches on trigger.kind.
    Returns: {body, cta, send_as, suppression_key, rationale}
    """
    kind = trigger.get("kind", "")

    dispatch = {
        "research_digest": lambda: _call_llm(_prompt_research_digest(category, merchant, trigger)),
        "regulation_change": lambda: _call_llm(_prompt_regulation_change(category, merchant, trigger)),
        "recall_due": lambda: _call_llm(_prompt_recall_due(category, merchant, trigger, customer)),
        "perf_dip": lambda: _call_llm(_prompt_perf_dip(category, merchant, trigger)),
        "seasonal_perf_dip": lambda: _call_llm(_prompt_perf_dip(category, merchant, trigger)),
        "perf_spike": lambda: _call_llm(_prompt_perf_spike(category, merchant, trigger)),
        "renewal_due": lambda: _call_llm(_prompt_renewal_due(category, merchant, trigger)),
        "milestone_reached": lambda: _call_llm(_prompt_milestone_reached(category, merchant, trigger)),
        "competitor_opened": lambda: _call_llm(_prompt_competitor_opened(category, merchant, trigger)),
        "festival_upcoming": lambda: _call_llm(_prompt_festival_upcoming(category, merchant, trigger)),
        "curious_ask_due": lambda: _call_llm(_prompt_curious_ask(category, merchant, trigger)),
        "winback_eligible": lambda: _call_llm(_prompt_winback(category, merchant, trigger)),
        "dormant_with_vera": lambda: _call_llm(_prompt_winback(category, merchant, trigger)),
        "gbp_unverified": lambda: _call_llm(_prompt_gbp_unverified(category, merchant, trigger)),
        "cde_opportunity": lambda: _call_llm(_prompt_cde_opportunity(category, merchant, trigger)),
        "supply_alert": lambda: _call_llm(_prompt_supply_alert(category, merchant, trigger)),
        "active_planning_intent": lambda: _call_llm(_prompt_active_planning(category, merchant, trigger)),
    }

    fn = dispatch.get(kind)
    if fn:
        return fn()
    return _call_llm(_prompt_generic(category, merchant, trigger, customer))


def compose_reply(
    category: dict,
    merchant: dict,
    customer: Optional[dict],
    conversation_history: list,
    new_message: str,
    turn_number: int,
) -> dict:
    """
    Compose a reply to a merchant/customer message in an ongoing conversation.
    Returns: {action: send|wait|end, body?, cta?, rationale}
    """
    # Check for exit intent first
    if detect_exit_intent(new_message):
        name = _owner_name(merchant)
        lang = merchant.get("identity", {}).get("languages", ["en"])
        if "hi" in lang:
            body = f"Samajh gayi. Koi baat nahi, {name}! Jab bhi kuch chahiye, main yahan hoon. 🙂"
        else:
            body = f"No worries, {name}! I'll be here whenever you need. 🙂"
        return {"action": "end", "rationale": "Merchant signaled disinterest; gracefully exiting conversation."}

    # Check for auto-reply
    is_auto = detect_auto_reply(new_message)

    # Check for action intent
    is_action = detect_action_intent(new_message)

    # Build conversation context for the reply
    hist_str = "\n".join(
        f"[{h.get('from', '?').upper()}] {h.get('body', h.get('msg', ''))}"
        for h in conversation_history[-6:]
    )

    merchant_name = _owner_name(merchant)
    lang = merchant.get("identity", {}).get("languages", ["en"])
    lang_note = "Hindi-English mix (Hinglish)" if "hi" in lang else "English"

    # Auto-reply handling
    if is_auto:
        if turn_number <= 3:
            # Try once more with a different angle
            prompt = f"""Vera received what looks like an auto-reply from {merchant_name}.

CONVERSATION SO FAR:
{hist_str}

LATEST MESSAGE (likely auto-reply): "{new_message}"

Auto-reply detected. Make ONE gentle attempt to reach the real merchant/owner.
Short, human, curious. 1-2 sentences. Don't re-explain why you're messaging.
Language: {lang_note}

Return JSON: {{action: "send", body: "...", cta: "open_ended", rationale: "..."}}"""
        else:
            # Give up gracefully
            return {
                "action": "end",
                "rationale": f"Auto-reply detected {turn_number} times. Gracefully exiting to avoid spam.",
            }
    elif is_action and turn_number <= 4:
        prompt = f"""Vera got a positive/action response from {merchant_name}. Take immediate action.

CONVERSATION SO FAR:
{hist_str}

MERCHANT'S MESSAGE: "{new_message}"

The merchant said YES or showed clear intent to proceed. Do NOT ask another qualifying question.
Deliver the promised action immediately (draft content, confirm what you're doing, next concrete step).
Language: {lang_note}

Return JSON: {{action: "send", body: "...", cta: "open_ended", rationale: "..."}}"""
    else:
        prompt = f"""Continue this WhatsApp conversation as Vera.

MERCHANT: {merchant_name} (category: {category.get('slug', '')})
TURN: {turn_number}

CONVERSATION SO FAR:
{hist_str}

LATEST MESSAGE: "{new_message}"

Rules:
- Respond naturally and helpfully
- Be concise (2-3 sentences)
- If turn >= 4 and no engagement, consider graceful exit
- Language: {lang_note}
- Do NOT re-introduce yourself

Return JSON: {{action: "send"|"wait"|"end", body: "...", cta: "open_ended"|"none"|"yes_stop", rationale: "..."}}
If action is "wait": include wait_seconds field.
If action is "end": no body needed."""

    result = _call_llm(prompt)
    # Ensure action field exists
    if "action" not in result:
        result["action"] = "send"
    return result
