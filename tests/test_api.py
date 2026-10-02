"""HTTP tests for the FastAPI triage service (no LLM key needed)."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from triage_agent.api import app

client = TestClient(app)

SINGLE_TICKET = {
    "id": "ticket-1-billing",
    "customer_id": "cust_free_001",
    "messages": [
        {"timestamp": "1h ago", "text": "I have THREE charges of $29.99, none refunded."},
        {"timestamp": "just now", "text": "Presentation in 2 hours, will dispute."},
    ],
}


def test_health_ok() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "llm_provider" in body
    assert "llm_configured" in body


def test_root_lists_endpoints() -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "POST /triage" in r.json()["endpoints"]


def test_samples_shape() -> None:
    r = client.get("/tickets/samples")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    assert all("messages" in t and "customer_id" in t for t in body["tickets"])


def test_triage_fallback_escalates() -> None:
    """No LLM key -> 200 with fail-closed escalate, never 500."""
    with patch("triage_agent.utils.nodes.get_llm", return_value=None):
        r = client.post("/triage", json=SINGLE_TICKET)
    assert r.status_code == 200
    body = r.json()
    assert body["ticket_id"] == "ticket-1-billing"
    assert body["result"]["next_action"] == "escalate to human"
    assert body["result"]["confidence"] == 0.5


def test_triage_rejects_empty_messages() -> None:
    r = client.post(
        "/triage",
        json={"id": "t-empty", "customer_id": "c1", "messages": []},
    )
    assert r.status_code == 422


def test_triage_batch_fallback() -> None:
    payload = {
        "tickets": [
            SINGLE_TICKET,
            {
                "id": "ticket-3-darkmode",
                "customer_id": "cust_pro_123",
                "messages": [{"timestamp": "today", "text": "Dark mode bug?"}],
            },
        ]
    }
    with patch("triage_agent.utils.nodes.get_llm", return_value=None):
        r = client.post("/triage/batch", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert body["ok"] == 2
    assert body["errors"] == 0
    assert all(x["result"]["next_action"] == "escalate to human" for x in body["results"])


def test_triage_batch_partial_failure() -> None:
    """One bad ticket must not abort the batch — its entry carries 'error'."""
    from triage_agent.api import run_ticket as real_run_ticket

    def flaky(ticket: dict):  # type: ignore[no-untyped-def]
        if ticket["id"] == "ticket-bad":
            raise RuntimeError("boom")
        return real_run_ticket(ticket)

    payload = {
        "tickets": [
            SINGLE_TICKET,
            {
                "id": "ticket-bad",
                "customer_id": "c1",
                "messages": [{"timestamp": "now", "text": "kaboom"}],
            },
        ]
    }
    with (
        patch("triage_agent.utils.nodes.get_llm", return_value=None),
        patch("triage_agent.api.run_ticket", side_effect=flaky),
    ):
        r = client.post("/triage/batch", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert body["ok"] == 1
    assert body["errors"] == 1
    by_id = {x["ticket_id"]: x for x in body["results"]}
    assert "result" in by_id["ticket-1-billing"]
    assert "boom" in by_id["ticket-bad"]["error"]
    # Order preserved
    assert [x["ticket_id"] for x in body["results"]] == ["ticket-1-billing", "ticket-bad"]


def test_triage_rejects_blank_text() -> None:
    r = client.post(
        "/triage",
        json={"id": "t-blank", "customer_id": "c1", "messages": [{"text": "   "}]},
    )
    assert r.status_code == 422


def test_triage_batch_rejects_oversize() -> None:
    payload = {"tickets": [SINGLE_TICKET] * 21}
    r = client.post("/triage/batch", json=payload)
    assert r.status_code == 422


def test_request_id_header_present() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.headers.get("x-request-id")


def test_rate_limit_429(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from triage_agent import api as api_module

    api_module._rate_hits.clear()
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "2")
    with patch("triage_agent.utils.nodes.get_llm", return_value=None):
        assert client.post("/triage", json=SINGLE_TICKET).status_code == 200
        assert client.post("/triage", json=SINGLE_TICKET).status_code == 200
        r = client.post("/triage", json=SINGLE_TICKET)
    assert r.status_code == 429
    assert "rate limit" in r.json()["detail"]
    api_module._rate_hits.clear()


def test_triage_batch_rejects_empty() -> None:
    r = client.post("/triage/batch", json={"tickets": []})
    assert r.status_code == 422


def test_mock_tool_endpoints_no_key() -> None:
    assert "plan=Free" in client.get("/tools/profile?customer_id=cust_free_001").json()["output"]
    assert "pending" in client.get("/tools/billing?customer_id=cust_free_001").json()["output"].lower()
    assert "degraded" in client.get("/tools/status?region=asia").json()["output"].lower()
    assert "kb-" in client.get("/tools/kb?query=duplicate+charge").json()["output"]
