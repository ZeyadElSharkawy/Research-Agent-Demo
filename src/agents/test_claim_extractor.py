# File: test_claim_extractor.py
"""
Test the claim extractor with the actual answer from our pipeline
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from claim_extractor_agent import extract_claims

# Use the actual answer that was generated
test_answer = """
A workflow status check script for chatbots, specifically designed for L0 (low-complexity) diagnostics on the NexaCore In-App Assistant platform, aims to resolve basic issues or collect data for L1 escalation. This script, referred to as 'Workflow Status Check (Pre-Escalation)', follows a structured question-and-answer flow:

### Workflow Status Check Script (L0 Diagnostics)

**Objective:** Resolve L0 (low-complexity) issues and efficiently triage/collect data for L1 escalation concerning workflow status.

**Platform:** NexaCore In-App Assistant (Powered by Gemini) [Source: Document 1]

**Example Flow:**

*   **User Input:** "My expense approval workflow (ID: 789) hasn't executed for 3 hours." [Source: Document 1]

*   **Chatbot Initial Response (L0):**
    *   Acknowledges the request: "Hello! I can certainly help check the status of Workflow ID 789." [Source: Document 1]
    *   Seeks to confirm the core issue by asking:
        1.  "Is the workflow currently stuck (In Progress, but not moving)?" [Source: Document 1]
        2.  "Did the workflow fail with an error message?" [Source: Document 1]
        3.  "Did the workflow fail to start at all (Trigger not activating)?" [Source: Document 1]

*   **User Selection Example:** User selects: "3. Did the workflow fail to start at all." [Source: Document 1]

*   **Chatbot Follow-up Response (L0):**
    *   Identifies potential cause: "Thank you. That usually points to a trigger configuration issue." [Source: Document 1]
    *   Before escalation, requests further confirmation:
        *   "Is the workflow trigger based on a Scheduled Time or an API Event?" [Source: Document 1, Document 2]
        *   "Have you checked the API Key status for the connecting service in the Integrations section?" [Source: Document 2]

*   **Action based on User Confirmation (Example):** If the user confirms they checked the API Key:
    *   **Chatbot Response (L0):** "I see. I have logged these diagnostic steps. I am now opening a P2 ticket for you: [TICKET_ID]. I have automatically attached your workflow ID and the confirmed check of the API key status. An L1 specialist will take over shortly." [Source: Document 2]

This script is part of a set of chatbot script samples for L0 diagnostics, which also includes scripts for general information/documentation requests and multilingual triage. [Source: Document 1, Document 2, Document 3]
"""

print("🧪 Testing Claim Extractor with actual pipeline answer...")
claims = extract_claims(test_answer)

print(f"\n✅ Extracted {len(claims)} claims:")
for i, claim in enumerate(claims, 1):
    print(f"{i}. {claim}")