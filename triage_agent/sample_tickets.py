"""The 3 sample tickets, condensed from docs/assignment-summary.md.

Each ticket keeps the full thread (timestamps + original wording, Thai kept for
Ticket 2) because triage must judge temporal escalation, not just the last message.
"""

SAMPLE_TICKETS = [
    {
        "id": "ticket-1-billing",
        "customer_id": "cust_free_001",
        "messages": [
            ("3h ago", "My payment failed when I tried to upgrade to Pro. Can you check what's wrong?"),
            ("2h ago", "I tried again with a different card. Now I see TWO pending charges but my account still shows Free plan??"),
            ("1h ago", "Okay this is getting ridiculous. Just checked my bank app - I have THREE charges of $29.99 now. None of them refunded. And I STILL don't have Pro access."),
            ("just now", "HELLO?? Is anyone there??? I need this fixed NOW. I have a presentation in 2 hours and I need the Pro export features. If these charges aren't reversed by end of day I'm disputing all of them with my bank."),
        ],
    },
    {
        "id": "ticket-2-outage",
        "customer_id": "cust_ent_th_045",
        "messages": [
            ("2h ago", "ระบบเข้าไม่ได้ครับ ขึ้น error 500 (Can't access the system, showing error 500)"),
            ("1.5h ago", "ลองหลายเครื่องแล้ว ทั้ง Chrome, Safari, Firefox ผลเหมือนกันหมด เพื่อนร่วมงานก็เข้าไม่ได้เหมือนกัน (Tried multiple machines, same result. Coworkers also can't access)"),
            ("45m ago", "ตอนนี้ลูกค้าโวยเข้ามาเยอะมาก เรามี demo กับลูกค้ารายใหญ่บ่ายนี้ ถ้าระบบไม่กลับมา deal นี้อาจจะหลุด (Customers flooding in. We have a demo with a major client this afternoon.)"),
            ("just now", "เช็ค status.company.com แล้ว บอกว่า all systems operational แต่เราใช้งานไม่ได้จริงๆ ช่วยเช็คให้หน่อยได้ไหมครับ region Asia มีปัญหาหรือเปล่า? (Status page says operational but we really can't use it. Is there an issue with the Asia region?)"),
        ],
    },
    {
        "id": "ticket-3-darkmode",
        "customer_id": "cust_pro_123",
        "messages": [
            ("2d ago", "Hey, just wondering if you support dark mode? No rush 😊"),
            ("1d ago", "Thanks for the reply! Oh nice, so it's in Settings > Appearance. Found it! But hmm I'm on Pro plan and I only see 'Light' and 'System Default' options. No dark mode toggle?"),
            ("1d ago +3h", "Okay so I switched to 'System Default' and my Mac is set to dark mode, but your app still shows light theme. Is this a bug or am I missing something?"),
            ("today", "Also random question while I have you - is there a way to schedule dark mode? Like auto-switch at 6pm? Some apps have that. Would be cool if you guys added it 👀"),
        ],
    },
]
