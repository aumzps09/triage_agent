"""Static mock data covering the 3 sample tickets.

No network access. All tools read from these in-memory tables so the
LangGraph graph runs without an API key.
"""

CUSTOMERS = {
    "cust_free_001": {
        "customer_id": "cust_free_001",
        "plan": "Free",
        "tenure_months": 4,
        "tickets_before": 0,
        "notes": "Attempted upgrade to Pro; first support contact.",
    },
    "cust_ent_th_045": {
        "customer_id": "cust_ent_th_045",
        "plan": "Enterprise",
        "region": "Thailand/Asia",
        "seats": 45,
        "tenure_months": 8,
        "tickets_before": 0,
        "notes": "First critical issue; demo with major client this afternoon.",
    },
    "cust_pro_123": {
        "customer_id": "cust_pro_123",
        "plan": "Pro",
        "tenure_months": 5,
        "tickets_before": 0,
        "notes": "Daily active user; never opened a ticket before.",
    },
}

BILLING_CHARGES = {
    "cust_free_001": [
        {"amount": 29.99, "currency": "USD", "status": "pending", "refunded": False},
        {"amount": 29.99, "currency": "USD", "status": "pending", "refunded": False},
        {"amount": 29.99, "currency": "USD", "status": "pending", "refunded": False},
    ],
}

SYSTEM_STATUS = {
    "global_page": "all systems operational (stale)",
    "regions": {
        "asia": {"status": "degraded", "detail": "HTTP 500 on app servers; multi-browser, multi-user reports."},
        "us": {"status": "operational", "detail": ""},
        "eu": {"status": "operational", "detail": ""},
    },
}

KB_ENTRIES = [
    {
        "id": "kb-billing-duplicate",
        "title": "Duplicate charges after failed upgrade",
        "keywords": ["charge", "billing", "duplicate", "refund", "upgrade", "pending", "dispute", "ตัดเงิน", "คืนเงิน", "เรียกเก็บ"],
        "text": "If upgrade shows Free plan with pending charges: do NOT retry the card. "
        "Support reverses duplicate $29.99 pending charges and manually grants Pro access pending settlement.",
    },
    {
        "id": "kb-upgrade-pro-access",
        "title": "Upgrade failed but charged — Pro export features",
        "keywords": ["upgrade", "pro", "export", "access", "payment failed", "free plan"],
        "text": "Failed upgrade keeps the account on Free until billing settles. "
        "For urgent deadlines, support can grant temporary Pro export access while refund is processed.",
    },
    {
        "id": "kb-outage-500",
        "title": "Error 500 across browsers and coworkers",
        "keywords": ["500", "error", "outage", "cannot access", "browser", "chrome", "safari", "firefox", "server", "ล่ม", "ใช้งานไม่ได้", "เข้าไม่ได้"],
        "text": "Same error on Chrome/Safari/Firefox for multiple users means a server/region "
        "outage, not a client issue. Escalate with region and seat count.",
    },
    {
        "id": "kb-status-page",
        "title": "Status page says operational but region is down",
        "keywords": ["status", "operational", "region", "asia", "status.company.com", "สถานะ"],
        "text": "The global status page can lag region incidents. Always check per-region "
        "status (especially Asia) when customers report an outage the page denies.",
    },
    {
        "id": "kb-dark-mode",
        "title": "Dark mode in Settings > Appearance",
        "keywords": ["dark mode", "appearance", "settings", "theme", "system default", "mac", "light", "โหมดมืด", "ธีม"],
        "text": "Dark mode lives in Settings > Appearance. If 'System Default' on macOS "
        "does not follow the OS theme, it is a known display bug — report app/OS versions.",
    },
    {
        "id": "kb-feature-request",
        "title": "Feature requests (e.g. scheduled dark mode)",
        "keywords": ["feature", "schedule", "auto-switch", "request", "dark mode", "ฟีเจอร์"],
        "text": "Schedule/auto-switch requests are feature requests, not bugs. "
        "Auto-respond with the FAQ answer and file the request for the roadmap.",
    },
]
