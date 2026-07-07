"""
local_test.py -- Quick local test of all 5 endpoints.
Run: python local_test.py (after the server is running on localhost:8080)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import httpx

BASE = "http://localhost:8080"
HEADERS = {"Content-Type": "application/json"}

def ok(label, resp):
    try:
        data = resp.json()
    except Exception:
        data = resp.text
    icon = "[OK]" if resp.status_code < 400 else "[ERR]"
    print(f"{icon} [{resp.status_code}] {label}: {json.dumps(data, indent=2, ensure_ascii=False)[:300]}")
    return data


def run():
    print("\n=== Vera Bot - Local Tests ===\n")

    # 1. healthz
    r = httpx.get(f"{BASE}/v1/healthz")
    ok("GET /v1/healthz", r)

    # 2. metadata
    r = httpx.get(f"{BASE}/v1/metadata")
    ok("GET /v1/metadata", r)

    # 3. Push category context
    category_payload = {
        "slug": "dentists",
        "display_name": "Dentists",
        "voice": {
            "tone": "peer_clinical",
            "vocab_taboo": ["guaranteed", "cure", "miracle"],
        },
        "offer_catalog": [
            {"id": "den_001", "title": "Dental Cleaning @ ₹299", "value": "299"},
        ],
        "peer_stats": {"avg_ctr": 0.030, "avg_rating": 4.4, "avg_reviews": 62},
        "digest": [
            {
                "id": "d_2026W17_jida_fluoride",
                "kind": "research",
                "title": "3-month fluoride varnish recall outperforms 6-month for high-risk adult caries",
                "source": "JIDA Oct 2026, p.14",
                "trial_n": 2100,
                "patient_segment": "high_risk_adults",
                "summary": "38% lower caries recurrence with 3-month vs 6-month recall.",
            }
        ],
        "patient_content_library": [],
        "seasonal_beats": [],
        "trend_signals": [],
    }
    r = httpx.post(f"{BASE}/v1/context", json={
        "scope": "category", "context_id": "dentists", "version": 1,
        "payload": category_payload, "delivered_at": "2026-04-26T10:00:00Z"
    })
    ok("POST /v1/context (category=dentists)", r)

    # 4. Push merchant context
    merchant_payload = {
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "category_slug": "dentists",
        "identity": {
            "name": "Dr. Meera's Dental Clinic",
            "city": "Delhi",
            "locality": "Lajpat Nagar",
            "languages": ["en", "hi"],
            "owner_first_name": "Meera",
            "verified": True,
        },
        "subscription": {"status": "active", "plan": "Pro", "days_remaining": 82},
        "performance": {"window_days": 30, "views": 2410, "calls": 18, "ctr": 0.021},
        "offers": [{"id": "o_meera_001", "title": "Dental Cleaning @ ₹299", "status": "active"}],
        "conversation_history": [],
        "customer_aggregate": {"total_unique_ytd": 540, "lapsed_180d_plus": 78, "high_risk_adult_count": 124},
        "signals": ["stale_posts:22d", "ctr_below_peer_median", "high_risk_adult_cohort"],
    }
    r = httpx.post(f"{BASE}/v1/context", json={
        "scope": "merchant", "context_id": "m_001_drmeera_dentist_delhi", "version": 1,
        "payload": merchant_payload, "delivered_at": "2026-04-26T10:00:00Z"
    })
    ok("POST /v1/context (merchant=Dr.Meera)", r)

    # 5. Push trigger context
    trigger_payload = {
        "id": "trg_001_research_digest_dentists",
        "scope": "merchant", "kind": "research_digest", "source": "external",
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "customer_id": None,
        "payload": {"category": "dentists", "top_item_id": "d_2026W17_jida_fluoride"},
        "urgency": 2,
        "suppression_key": "research:dentists:2026-W17",
        "expires_at": "2027-05-03T00:00:00Z",
    }
    r = httpx.post(f"{BASE}/v1/context", json={
        "scope": "trigger", "context_id": "trg_001_research_digest_dentists", "version": 1,
        "payload": trigger_payload, "delivered_at": "2026-04-26T10:00:00Z"
    })
    ok("POST /v1/context (trigger=research_digest)", r)

    # 6. Tick — should compose a message
    r = httpx.post(f"{BASE}/v1/tick", json={
        "now": "2026-04-26T10:30:00Z",
        "available_triggers": ["trg_001_research_digest_dentists"],
    }, timeout=35.0)
    tick_data = ok("POST /v1/tick", r)
    
    actions = tick_data.get("actions", []) if isinstance(tick_data, dict) else []
    if not actions:
        print("[WARN] No actions returned from tick (may need GEMINI_API_KEY set)")
        conv_id = "conv_test_001"
    else:
        conv_id = actions[0].get("conversation_id", "conv_test_001")
        print(f"\n   >> Composed body: {actions[0].get('body', '')}")

    # 7. Simulate merchant reply
    r = httpx.post(f"{BASE}/v1/reply", json={
        "conversation_id": conv_id,
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "customer_id": None,
        "from_role": "merchant",
        "message": "Yes, send me the abstract",
        "received_at": "2026-04-26T10:45:00Z",
        "turn_number": 2,
    }, timeout=35.0)
    reply_data = ok("POST /v1/reply (merchant says 'yes')", r)
    if isinstance(reply_data, dict) and reply_data.get("body"):
        print(f"\n   >> Vera reply: {reply_data.get('body', '')}")

    # 8. Test auto-reply detection
    r = httpx.post(f"{BASE}/v1/reply", json={
        "conversation_id": conv_id,
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "customer_id": None,
        "from_role": "merchant",
        "message": "Thank you for contacting us. Our team will get back to you shortly.",
        "received_at": "2026-04-26T10:46:00Z",
        "turn_number": 3,
    }, timeout=35.0)
    ok("POST /v1/reply (auto-reply test)", r)

    # 9. Test idempotency — same version should return 409
    r = httpx.post(f"{BASE}/v1/context", json={
        "scope": "category", "context_id": "dentists", "version": 1,
        "payload": category_payload, "delivered_at": "2026-04-26T10:01:00Z"
    })
    ok("POST /v1/context (idempotency test — same version → 409)", r)

    # 10. Healthz after loading — should show loaded counts
    r = httpx.get(f"{BASE}/v1/healthz")
    ok("GET /v1/healthz (after loading contexts)", r)

    print("\n=== Tests complete ===\n")


if __name__ == "__main__":
    run()
