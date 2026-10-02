"""Unit tests for mock tools."""

from __future__ import annotations

from triage_agent.utils import tools


def test_get_customer_profile_found() -> None:
    result = tools.get_customer_profile.invoke({"customer_id": "cust_free_001"})
    assert "plan=Free" in result
    assert "tenure_months=4" in result


def test_get_customer_profile_not_found() -> None:
    result = tools.get_customer_profile.invoke({"customer_id": "nonexistent"})
    assert "Customer not found" in result
    assert "nonexistent" in result


def test_search_knowledge_base_hit() -> None:
    result = tools.search_knowledge_base.invoke({"query": "payment failed upgrade to Pro"})
    assert "kb-billing-duplicate" in result or "kb-upgrade-pro-access" in result


def test_search_knowledge_base_no_tokens() -> None:
    result = tools.search_knowledge_base.invoke({"query": "a b"})
    assert "No matching articles" in result


def test_search_knowledge_base_no_matches() -> None:
    result = tools.search_knowledge_base.invoke({"query": "xylophone zebra astronaut"})
    assert "No matching articles" in result


def test_search_knowledge_base_thai_outage() -> None:
    result = tools.search_knowledge_base.invoke({"query": "เซิร์ฟเวอร์ล่มเข้าไม่ได้เลย"})
    assert "kb-outage-500" in result


def test_search_knowledge_base_thai_billing() -> None:
    result = tools.search_knowledge_base.invoke({"query": "ถูกตัดเงินซ้ำขอคืนเงิน"})
    assert "kb-billing-duplicate" in result


def test_search_knowledge_base_thai_darkmode() -> None:
    result = tools.search_knowledge_base.invoke({"query": "โหมดมืดไม่ทำงาน"})
    assert "kb-dark-mode" in result


def test_check_system_status_known_region() -> None:
    result = tools.check_system_status.invoke({"region": "asia"})
    assert "Region asia: degraded" in result
    assert "Global page says:" in result


def test_check_system_status_unknown_region() -> None:
    result = tools.check_system_status.invoke({"region": "mars"})
    assert "Unknown region 'mars'" in result
    assert "Global page says:" in result


def test_check_billing_with_charges() -> None:
    result = tools.check_billing.invoke({"customer_id": "cust_free_001"})
    assert "3 charge(s)" in result
    assert "$89.97 pending/unrefunded" in result


def test_check_billing_no_charges() -> None:
    result = tools.check_billing.invoke({"customer_id": "cust_ent_th_045"})
    assert "No charges on file" in result


def test_as_langchain_tools() -> None:
    all_tools = tools.as_langchain_tools()
    assert len(all_tools) == 4
    names = {t.name for t in all_tools}
    assert names == {
        "get_customer_profile",
        "search_knowledge_base",
        "check_system_status",
        "check_billing",
    }
