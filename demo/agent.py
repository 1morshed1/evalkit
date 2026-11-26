"""Synthetic agent traces for the demo, in OpenAI chat-completion format.

These are the same traces embedded in ``demo/agent_suite.yaml``; they live here
too so the OpenAI adapter (``AgentTrace.from_openai``) can be exercised directly
in code and tests. Each is a list of OpenAI ``messages``.
"""

from __future__ import annotations

from typing import Any

# A clean run: two tool calls, both succeed, correct final answer, 2 LLM steps.
REFUND_LOOKUP: list[dict[str, Any]] = [
    {"role": "user", "content": "How many days do I have to return a defective product?"},
    {
        "role": "assistant",
        "content": None,
        "cost_usd": 0.0009,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "search_kb",
                    "arguments": '{"query": "return window defective"}',
                },
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "Defective products: 90-day return window.",
    },
    {
        "role": "assistant",
        "content": "You have 90 days to return a defective product.",
        "cost_usd": 0.0011,
    },
]

# A run that hits a tool error and recovers by retrying the same tool.
ORDER_STATUS_WITH_RETRY: list[dict[str, Any]] = [
    {"role": "user", "content": "What's the status of order 5591?"},
    {
        "role": "assistant",
        "content": None,
        "cost_usd": 0.0008,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_order", "arguments": '{"order_id": "5591"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "Error: upstream timeout"},
    {
        "role": "assistant",
        "content": None,
        "cost_usd": 0.0008,
        "tool_calls": [
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "get_order", "arguments": '{"order_id": "5591"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_2", "content": "Order 5591: shipped, arriving Tuesday."},
    {
        "role": "assistant",
        "content": "Order 5591 has shipped and arrives Tuesday.",
        "cost_usd": 0.0012,
    },
]

# A failing run: wrong/extra tool, wanders for many steps, wrong answer.
PRICING_WRONG: list[dict[str, Any]] = [
    {"role": "user", "content": "How much is the enterprise plan?"},
    {
        "role": "assistant",
        "content": None,
        "cost_usd": 0.001,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "web_search", "arguments": '{"q": "enterprise plan"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "No results."},
    {
        "role": "assistant",
        "content": None,
        "cost_usd": 0.001,
        "tool_calls": [
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "web_search", "arguments": '{"q": "pricing"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_2", "content": "No results."},
    {"role": "assistant", "content": "I'm not sure of the price.", "cost_usd": 0.001},
]

TRACES = {
    "refund-lookup": REFUND_LOOKUP,
    "order-status-retry": ORDER_STATUS_WITH_RETRY,
    "pricing-wrong": PRICING_WRONG,
}
