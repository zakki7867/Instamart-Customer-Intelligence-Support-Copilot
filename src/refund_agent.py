"""refund_agent.py -- ReAct Refund Agent for Part D.
Implements five tools, ReAct execution loop, graceful limits (stop safely),
state persistence & resume, and scenario evaluation over data/raw/agent_scenarios.csv.
Produces outputs/agent_runs.jsonl, outputs/refund_ledger.jsonl, and checkpoints in state/.
"""
import os
import sys
import json
import sqlite3
import time
import datetime
import pandas as pd

# Ensure root is in path for mock_llm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import mock_llm
from src.policy_rag import chunk_documents, PolicyRetriever

DB_PATH = os.path.join("data", "instamart.db")
SCENARIOS_PATH = os.path.join("data", "raw", "agent_scenarios.csv")
AGENT_RUNS_PATH = os.path.join("outputs", "agent_runs.jsonl")
REFUND_LEDGER_PATH = os.path.join("outputs", "refund_ledger.jsonl")
STATE_DIR = os.path.join("state")

os.makedirs("outputs", exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)


class SimulatedAgentCrashError(Exception):
    """Exception used to simulate a runtime crash after a specific tool execution."""
    pass


# Initialize Policy Retriever for search_policy tool
_chunks, _ = chunk_documents()
_retriever = PolicyRetriever(_chunks)


# Tool definitions
def get_order(order_id):
    """Retrieves order details from SQLite data/instamart.db."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?", (order_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return {"error": "order not found"}
    return {
        "order_id": row[0],
        "status": row[1],
        "order_amount": float(row[2]),
        "delivery_minutes": None if row[3] is None else int(row[3])
    }


def search_policy(query):
    """Searches policy documents using PolicyRetriever and returns top 3 chunks."""
    results = _retriever.retrieve(query, k=3)
    return {"results": results}


def calculate_refund(order_amount, delivery_minutes):
    """Computes refund amount based on delivery time (> 45 min -> 100%, <= 45 min -> 0)."""
    if delivery_minutes is not None and delivery_minutes > 45:
        return {"refund_inr": float(order_amount)}
    return {"refund_inr": 0.0}


def issue_refund(order_id, refund_inr):
    """Issues an approved refund (up to INR 1000) and logs to outputs/refund_ledger.jsonl."""
    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tool": "issue_refund",
        "order_id": order_id,
        "refund_inr": float(refund_inr)
    }
    with open(REFUND_LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"refund_inr": float(refund_inr)}


def escalate_to_human(order_id, refund_inr):
    """Escalates high-value refunds (> INR 1000) to a human lead and logs to ledger."""
    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tool": "escalate_to_human",
        "order_id": order_id,
        "refund_inr": float(refund_inr)
    }
    with open(REFUND_LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"refund_inr": float(refund_inr)}


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Reads order status, amount, and delivery time from instamart.db.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"]
        }
    },
    {
        "name": "search_policy",
        "description": "Searches store policy documents for refund eligibility and delivery rules.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    },
    {
        "name": "calculate_refund",
        "description": "Calculates the eligible refund amount based on delivery delay.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": "number"}
            },
            "required": ["order_amount", "delivery_minutes"]
        }
    },
    {
        "name": "issue_refund",
        "description": "Appends an approved refund up to INR 1000 to the refund ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Appends a high-value refund (> INR 1000) escalation to the ledger for human approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    }
]


def save_checkpoint(run_id, scenario_id, messages, tokens_used, step_number, tools_executed):
    """Persists intermediate state to state/<run_id>.json."""
    ckpt = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "step_number": step_number,
        "tools_executed": tools_executed
    }
    ckpt_path = os.path.join(STATE_DIR, f"{run_id}.json")
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, indent=2)


def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after_tool_count=None
):
    """Runs the ReAct refund agent loop with safety limits and state checkpointing."""
    if run_id is None:
        run_id = f"run_{scenario_id}_{int(time.time() * 1000)}"

    messages = [
        {"role": "system", "content": "You are an AGENT that processes refunds using available tools."},
        {"role": "user", "content": user_message}
    ]

    tokens_used = 0
    llm_calls = 0
    steps = []
    tools_executed = []
    start_time = time.monotonic()

    final_answer = ""
    status = ""

    for iteration in range(max_iterations):
        # Time limit check (measured at top of iteration)
        elapsed = time.monotonic() - start_time
        if elapsed >= time_limit_seconds:
            status = "FALLBACK_TIMEOUT"
            final_answer = f"Handed over to human lead: execution time exceeded {time_limit_seconds}s limit."
            break

        # LLM Call
        resp = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += resp.usage.total_tokens

        # Token limit check
        if tokens_used > token_limit:
            status = "FALLBACK_TOKEN_LIMIT"
            final_answer = f"Handed over to human lead: cumulative token limit ({token_limit}) exceeded."
            break

        msg = resp.choices[0].message

        # Case 1: Final answer returned (no tool calls)
        if not msg.tool_calls:
            final_answer = msg.content
            if ":" in final_answer:
                status = final_answer.split(":")[0].strip()
            else:
                status = "UNKNOWN"
            break

        # Case 2: Tool call requested
        tc = msg.tool_calls[0]
        # Append assistant message with tool call
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]
        })

        # Execute tool
        tool_fn = TOOL_MAP[tc.name]
        tool_res = tool_fn(**tc.arguments)
        tools_executed.append(tc.name)

        # Append tool observation
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "name": tc.name,
            "content": json.dumps(tool_res)
        })

        step_record = {
            "step": iteration + 1,
            "tool": tc.name,
            "arguments": tc.arguments,
            "observation": tool_res
        }
        steps.append(step_record)

        # Save checkpoint after every tool execution
        save_checkpoint(run_id, scenario_id, messages, tokens_used, iteration + 1, tools_executed)

        # Simulate crash if requested
        if crash_after_tool_count is not None and len(tools_executed) == crash_after_tool_count:
            raise SimulatedAgentCrashError(f"Simulated crash after executing tool #{len(tools_executed)}: {tc.name}")

    else:
        # Loop reached max_iterations without breaking
        status = "FALLBACK_MAX_ITERATIONS"
        final_answer = f"Handed over to human lead: reached maximum {max_iterations} reasoning iterations."

    # Parse refund amount if present in answer
    refund_amount = 0.0
    if status == "REFUND":
        # Example: "REFUND: INR 128.25 issued."
        parts = final_answer.replace("INR", "").split()
        for p in parts:
            try:
                refund_amount = float(p.rstrip("."))
                break
            except ValueError:
                continue
    elif status == "ESCALATE":
        # Extract escalated refund amount from calculate_refund step
        for s in steps:
            if s["tool"] == "calculate_refund":
                refund_amount = float(s["observation"].get("refund_inr", 0.0))

    return {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "status": status,
        "refund_amount": refund_amount,
        "final_answer": final_answer,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
        "steps": steps,
        "tools_executed": tools_executed
    }


def resume_agent(
    run_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10
):
    """Loads state from state/<run_id>.json and resumes the agent execution."""
    ckpt_path = os.path.join(STATE_DIR, f"{run_id}.json")
    with open(ckpt_path, "r", encoding="utf-8") as f:
        ckpt = json.load(f)

    scenario_id = ckpt["scenario_id"]
    messages = ckpt["messages"]
    tokens_used = ckpt["tokens_used"]
    tools_executed = list(ckpt["tools_executed"])
    completed_steps = ckpt["step_number"]

    llm_calls = completed_steps  # number of calls made prior to resume
    steps = []
    # Reconstruct steps from messages
    for i in range(len(messages)):
        if messages[i].get("role") == "assistant" and messages[i].get("tool_calls"):
            tc = messages[i]["tool_calls"][0]
            # Next message is tool
            if i + 1 < len(messages) and messages[i + 1].get("role") == "tool":
                obs = json.loads(messages[i + 1]["content"])
                steps.append({"tool": tc["name"], "arguments": tc["arguments"], "observation": obs})

    start_time = time.monotonic()
    final_answer = ""
    status = ""

    for iteration in range(completed_steps, max_iterations):
        elapsed = time.monotonic() - start_time
        if elapsed >= time_limit_seconds:
            status = "FALLBACK_TIMEOUT"
            final_answer = f"Handed over to human lead: execution time exceeded limit."
            break

        resp = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += resp.usage.total_tokens

        if tokens_used > token_limit:
            status = "FALLBACK_TOKEN_LIMIT"
            final_answer = f"Handed over to human lead: token limit exceeded."
            break

        msg = resp.choices[0].message
        if not msg.tool_calls:
            final_answer = msg.content
            status = final_answer.split(":")[0].strip() if ":" in final_answer else "UNKNOWN"
            break

        tc = msg.tool_calls[0]
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]
        })

        tool_fn = TOOL_MAP[tc.name]
        tool_res = tool_fn(**tc.arguments)
        tools_executed.append(tc.name)

        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "name": tc.name,
            "content": json.dumps(tool_res)
        })

        step_record = {
            "step": iteration + 1,
            "tool": tc.name,
            "arguments": tc.arguments,
            "observation": tool_res
        }
        steps.append(step_record)
        save_checkpoint(run_id, scenario_id, messages, tokens_used, iteration + 1, tools_executed)
    else:
        status = "FALLBACK_MAX_ITERATIONS"
        final_answer = "Handed over to human lead: maximum iterations exceeded."

    refund_amount = 0.0
    if status == "REFUND":
        parts = final_answer.replace("INR", "").split()
        for p in parts:
            try:
                refund_amount = float(p.rstrip("."))
                break
            except ValueError:
                continue

    return {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "status": status,
        "refund_amount": refund_amount,
        "final_answer": final_answer,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
        "steps": steps,
        "tools_executed": tools_executed
    }


def main():
    print("=" * 60)
    print("REFUND AGENT & SCENARIO EVALUATION (PART D)")
    print("=" * 60)

    # Clean ledger and runs files for fresh execution
    if os.path.exists(REFUND_LEDGER_PATH):
        os.remove(REFUND_LEDGER_PATH)
    if os.path.exists(AGENT_RUNS_PATH):
        os.remove(AGENT_RUNS_PATH)

    scenarios = pd.read_csv(SCENARIOS_PATH)
    scenario_results = []

    print("Running 7 Simulated Agent Scenarios:")
    for _, sc in scenarios.iterrows():
        sid = sc["scenario_id"]
        cid = sc["customer_id"]
        msg = sc["message"]
        exp_oc = sc["expected_outcome"]
        exp_amt = float(sc["expected_refund_inr"])

        res = run_agent(msg, scenario_id=sid, max_iterations=6, token_limit=6000, time_limit_seconds=10)

        # Append to outputs/agent_runs.jsonl
        run_record = {
            "scenario_id": res["scenario_id"],
            "status": res["status"],
            "final_answer": res["final_answer"],
            "llm_calls": res["llm_calls"],
            "tokens_used": res["tokens_used"],
            "steps": res["steps"]
        }
        with open(AGENT_RUNS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(run_record) + "\n")

        scenario_results.append({
            "Scenario ID": sid,
            "Expected Outcome": exp_oc,
            "Actual Status": res["status"],
            "Expected Refund (INR)": f"{exp_amt:.2f}",
            "Actual Refund (INR)": f"{res['refund_amount']:.2f}",
            "Model Calls": res["llm_calls"],
            "Tokens Used": res["tokens_used"]
        })

        print(f"  {sid}: status={res['status']} (expected={exp_oc}), refund=INR {res['refund_amount']:.2f}, calls={res['llm_calls']}, tokens={res['tokens_used']}")
        if exp_oc in ["REFUND", "ESCALATE", "DENIED", "NOT_FOUND"]:
            assert res["status"] == exp_oc, f"{sid} status mismatch: expected {exp_oc}, got {res['status']}"
            assert abs(res["refund_amount"] - exp_amt) < 0.05, f"{sid} refund mismatch: expected {exp_amt}, got {res['refund_amount']}"
        elif exp_oc == "FALLBACK":
            assert res["status"] == "FALLBACK_MAX_ITERATIONS", f"{sid} fallback status mismatch: got {res['status']}"

    # Scenario Table
    df_sc = pd.DataFrame(scenario_results)
    print("\nScenario Results Table:")
    print(df_sc.to_string(index=False))

    # Task 7: Stop Safely Demonstrations
    print("\nTask 7: Stop-Safely Demonstrations:")
    s1_msg = scenarios.iloc[0]["message"]

    # Demo 1: Token limit exceeded
    demo_token = run_agent(s1_msg, scenario_id="S1_TOKEN_LIMIT", max_iterations=6, token_limit=200, time_limit_seconds=10)
    print(f"  Demo 1 (token_limit=200): status = {demo_token['status']} (Reference: FALLBACK_TOKEN_LIMIT)")
    assert demo_token["status"] == "FALLBACK_TOKEN_LIMIT"

    # Demo 2: Timeout
    demo_timeout = run_agent(s1_msg, scenario_id="S1_TIMEOUT", max_iterations=6, token_limit=6000, time_limit_seconds=0)
    print(f"  Demo 2 (time_limit_seconds=0): status = {demo_timeout['status']} (Reference: FALLBACK_TIMEOUT)")
    assert demo_timeout["status"] == "FALLBACK_TIMEOUT"

    # Task 8: Crash and Resume Demonstration
    print("\nTask 8: Crash-and-Resume Demonstration (S1):")
    crash_run_id = "s1_crash_resume_test"

    # Step 1: Run S1 and trigger crash after 2nd tool execution
    print("  Executing Part 1 of S1 (simulated crash after 2nd tool)...")
    try:
        run_agent(
            s1_msg,
            scenario_id="S1",
            max_iterations=6,
            token_limit=6000,
            time_limit_seconds=10,
            run_id=crash_run_id,
            crash_after_tool_count=2
        )
    except SimulatedAgentCrashError as crash_err:
        print(f"  Simulated crash caught successfully: {crash_err}")

    # Inspect checkpoint
    ckpt_path = os.path.join(STATE_DIR, f"{crash_run_id}.json")
    with open(ckpt_path, "r", encoding="utf-8") as f:
        pre_ckpt = json.load(f)
    print(f"  Tools executed before crash: {pre_ckpt['tools_executed']}")
    assert pre_ckpt["tools_executed"] == ["get_order", "search_policy"]

    # Step 2: Resume agent
    print("  Resuming agent from checkpoint...")
    resume_res = resume_agent(crash_run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10)
    print(f"  Resumed Run Status: {resume_res['status']} with refund INR {resume_res['refund_amount']:.2f}")
    print(f"  Complete sequence of executed tools across both parts: {resume_res['tools_executed']}")

    expected_tools = ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert resume_res["tools_executed"] == expected_tools, f"Expected {expected_tools}, got {resume_res['tools_executed']}"
    assert resume_res["status"] == "REFUND"
    assert abs(resume_res["refund_amount"] - 128.25) < 0.05
    print("  Assertion passed: Exactly four tools executed in order with no duplicates, ending with REFUND 128.25.")


if __name__ == "__main__":
    main()
