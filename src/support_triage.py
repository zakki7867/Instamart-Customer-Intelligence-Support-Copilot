"""support_triage.py -- Support ticket triage with guardrails, retries, PII redaction,
zero-shot vs few-shot comparison, token budgeting, and segment-based routing for Part C.
Generates outputs/triage_results.csv and outputs/triage_budget_demo.csv.
"""
import os
import sys
import re
import json
import random
import time
import pandas as pd

# Ensure root is in path for mock_llm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import mock_llm

TICKETS_PATH = os.path.join("data", "raw", "tickets.csv")
SEGMENTS_PATH = os.path.join("outputs", "customer_segments.csv")
RESULTS_OUTPUT_PATH = os.path.join("outputs", "triage_results.csv")
BUDGET_DEMO_OUTPUT_PATH = os.path.join("outputs", "triage_budget_demo.csv")

ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}
BASE_DELAY = 0.05
MAX_INPUT_TOKENS = 400

# Redaction patterns
RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
RE_PHONE = re.compile(r"\b[6-9]\d{9}\b")
RE_ORDER = re.compile(r"ORD-\d{6}")

# System prompts
ZERO_SHOT_PROMPT = (
    "You are a CLASSIFY assistant. Classify the customer support ticket into exactly one of "
    "these four categories: billing, delivery, technical, fraud. "
    'Return your response strictly as valid JSON with format: {"category": "...", "reason": "..."}.'
)

FEW_SHOT_PROMPT = (
    "You are a CLASSIFY assistant. Classify the customer support ticket into exactly one of "
    "these four categories: billing, delivery, technical, fraud. "
    'Return your response strictly as valid JSON with format: {"category": "...", "reason": "..."}.\n'
    "EXAMPLES:\n"
    "Ticket: I was charged twice for order [ORDER_ID]. Please refund the extra payment.\n"
    '{"category": "billing", "reason": "duplicate charge refund"}\n\n'
    "Ticket: There is an unauthorized transaction on my account. Someone hacked my wallet.\n"
    '{"category": "fraud", "reason": "unauthorized security breach"}'
)


def redact_pii(text):
    """Applies email, phone, and order id redactions in exact order."""
    text_sub, n_email = RE_EMAIL.subn("[EMAIL]", text)
    text_sub, n_phone = RE_PHONE.subn("[PHONE]", text_sub)
    text_sub, n_order = RE_ORDER.subn("[ORDER_ID]", text_sub)
    return text_sub, n_email, n_phone, n_order


def call_mock_with_retry(messages, ticket_id, max_retries=3, verbose=False):
    """Executes a chat call with exponential backoff and jitter."""
    for n in range(max_retries + 1):
        try:
            response = mock_llm.chat(messages)
            return response, n
        except mock_llm.TransientLLMError as e:
            if n < max_retries:
                wait = BASE_DELAY * (2 ** n) * random.uniform(0.9, 1.15)
                if verbose:
                    print(f"  [RETRY] Ticket {ticket_id}: retry {n}, waiting {wait:.4f}s ({e})")
                time.sleep(wait)
            else:
                raise


def parse_and_validate(content):
    """Validates model response according to strict hierarchical rules."""
    # Step 1: Try JSON parse
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            category = data.get("category", "")
            if category in ALLOWED_CATEGORIES:
                return category, "classified_json"
            else:
                return None, "flagged_invalid_output"
    except Exception:
        pass

    # Step 2: Fallback keyword word extraction
    words = re.findall(r"[a-z]+", content.lower())
    found_cats = [w for w in set(words) if w in ALLOWED_CATEGORIES]
    if len(found_cats) == 1:
        return found_cats[0], "classified_keyword"

    # Step 3: Flagged invalid output
    return None, "flagged_invalid_output"


def run_triage_pipeline(prompt, token_budget=None, log_retries=False):
    """Runs the triage pipeline over all 164 tickets in data/raw/tickets.csv."""
    tickets = pd.read_csv(TICKETS_PATH, keep_default_na=False)
    mock_llm.reset()

    results = []
    cumulative_tokens = 0
    total_retries = 0

    for _, row in tickets.iterrows():
        tid = row["ticket_id"]
        cid = row["customer_id"]
        raw_text = row["text"]
        true_cat = row["true_category"]

        # Input guardrail (a): empty or whitespace
        if not raw_text.strip():
            results.append({
                "ticket_id": tid,
                "customer_id": cid,
                "status": "skipped_empty",
                "category": "",
                "true_category": true_cat,
                "tokens_used": 0
            })
            continue

        # Input guardrail (b): input token limit
        est = mock_llm.est_tokens(raw_text)
        if est > MAX_INPUT_TOKENS:
            results.append({
                "ticket_id": tid,
                "customer_id": cid,
                "status": "flagged_long",
                "category": "",
                "true_category": true_cat,
                "tokens_used": 0
            })
            continue

        # Token budget check
        if token_budget is not None and cumulative_tokens >= token_budget:
            results.append({
                "ticket_id": tid,
                "customer_id": cid,
                "status": "budget_exhausted",
                "category": "",
                "true_category": true_cat,
                "tokens_used": 0
            })
            continue

        # PII Redaction
        redacted_text, _, _, _ = redact_pii(raw_text)
        # Verify no PII remains
        assert not RE_EMAIL.search(redacted_text)
        assert not RE_PHONE.search(redacted_text)
        assert not RE_ORDER.search(redacted_text)

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": redacted_text}
        ]

        # Resilient LLM Call
        resp, retries_taken = call_mock_with_retry(messages, tid, verbose=log_retries)
        total_retries += retries_taken

        tokens = resp.usage.total_tokens
        cumulative_tokens += tokens

        content = resp.choices[0].message.content
        category, status = parse_and_validate(content)

        results.append({
            "ticket_id": tid,
            "customer_id": cid,
            "status": status,
            "category": category if category else "",
            "true_category": true_cat,
            "tokens_used": tokens
        })

    df_out = pd.DataFrame(results)
    return df_out, cumulative_tokens, total_retries


def apply_routing(df_results, segments_path):
    """Routes tickets using Part B customer segments and classification status."""
    segments = pd.read_csv(segments_path)
    # Mapping customer_id -> (segment_name, is_premium)
    seg_map = {row["customer_id"]: (row["segment_name"], bool(row["is_premium"])) for _, row in segments.iterrows()}

    routes = []
    priorities = []
    seg_names = []

    for _, row in df_results.iterrows():
        status = row["status"]
        cat = row["category"]
        cid = row["customer_id"]

        seg_name, is_prem = seg_map.get(cid, ("", False))

        if status in ["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"]:
            route = "human_review"
            priority = "NORMAL"
        elif cat == "fraud":
            route = "security_team"
            priority = "HIGH"
        elif is_prem:
            route = "premium_desk"
            priority = "HIGH"
        else:
            route = "support_queue"
            priority = "NORMAL"

        routes.append(route)
        priorities.append(priority)
        seg_names.append(seg_name if seg_name else "")

    df_results["segment_name"] = seg_names
    df_results["priority"] = priorities
    df_results["route"] = routes

    output_cols = [
        "ticket_id", "customer_id", "status", "category",
        "segment_name", "priority", "route", "tokens_used"
    ]
    return df_results[output_cols]


def main():
    print("=" * 60)
    print("SUPPORT TICKET TRIAGE (PART C)")
    print("=" * 60)

    # 1. PII Redaction Audit across all 164 tickets
    tickets_raw = pd.read_csv(TICKETS_PATH, keep_default_na=False)
    e_tot, p_tot, o_tot = 0, 0, 0
    empty_cnt, long_cnt = 0, 0

    for text in tickets_raw["text"]:
        if not text.strip():
            empty_cnt += 1
        elif mock_llm.est_tokens(text) > MAX_INPUT_TOKENS:
            long_cnt += 1
        _, ne, np, no = redact_pii(text)
        e_tot += ne
        p_tot += np
        o_tot += no

    print("Task 1 & 2: Guardrails & Redaction Audit:")
    print(f"  Empty tickets skipped: {empty_cnt} (Reference: 4)")
    print(f"  Long tickets flagged: {long_cnt} (Reference: 4, ~734 tokens)")
    print(f"  Emails redacted: {e_tot} (Reference: 55)")
    print(f"  Phones redacted: {p_tot} (Reference: 91)")
    print(f"  Order IDs redacted: {o_tot} (Reference: 132)")
    print(f"  Total substitutions: {e_tot + p_tot + o_tot} (Reference: 278)")

    # 2. Zero-shot run
    print("\nRunning Zero-Shot Pipeline (no token budget)...")
    df_zs, tokens_zs, retries_zs = run_triage_pipeline(ZERO_SHOT_PROMPT, token_budget=None, log_retries=False)
    zs_classified = df_zs[df_zs["status"].isin(["classified_json", "classified_keyword"])]
    zs_acc = (zs_classified["category"] == zs_classified["true_category"]).mean() * 100
    print(f"  Status distribution: {df_zs['status'].value_counts().to_dict()}")
    print(f"  Classified tickets: {len(zs_classified)} (Reference: 147)")
    print(f"  Zero-shot accuracy: {zs_acc:.1f}% (Reference: 85.7%)")
    print(f"  Total tokens: {tokens_zs}, Tokens per classified ticket: {tokens_zs / len(zs_classified):.1f}")
    print(f"  Transient LLM errors recovered: {retries_zs} (Reference: 19)")

    print("\nZero-Shot Confusion Matrix (classified tickets):")
    ct = pd.crosstab(zs_classified["true_category"], zs_classified["category"], rownames=["True"], colnames=["Predicted"])
    print(ct.to_string())

    # 3. Few-shot run
    print("\nRunning Few-Shot Pipeline (no token budget)...")
    df_fs, tokens_fs, retries_fs = run_triage_pipeline(FEW_SHOT_PROMPT, token_budget=None, log_retries=False)
    fs_classified = df_fs[df_fs["status"].isin(["classified_json", "classified_keyword"])]
    fs_acc = (fs_classified["category"] == fs_classified["true_category"]).mean() * 100
    print(f"  Status distribution: {df_fs['status'].value_counts().to_dict()}")
    print(f"  Classified tickets: {len(fs_classified)} (Reference: 147)")
    print(f"  Few-shot accuracy: {fs_acc:.1f}% (Reference: 100.0%)")
    print(f"  Total tokens: {tokens_fs}, Tokens per classified ticket: {tokens_fs / len(fs_classified):.1f}")
    print(f"  Transient LLM errors recovered: {retries_fs} (Reference: 19)")

    # 4. Budget Demo (Zero-shot with TOKEN_BUDGET = 6000)
    print("\nRunning Token Budget Demo (Zero-shot with TOKEN_BUDGET = 6000)...")
    df_budget, tokens_budget, _ = run_triage_pipeline(ZERO_SHOT_PROMPT, token_budget=6000, log_retries=False)
    exhausted_count = (df_budget["status"] == "budget_exhausted").sum()
    print(f"  Tickets ended budget_exhausted: {exhausted_count}")
    df_budget_routed = apply_routing(df_budget, SEGMENTS_PATH)
    df_budget_routed.to_csv(BUDGET_DEMO_OUTPUT_PATH, index=False)
    print(f"  Saved budget demo to {BUDGET_DEMO_OUTPUT_PATH}")

    # 5. Final Few-shot routing with TOKEN_BUDGET = 100000
    print("\nGenerating Final Triage Results (Few-shot with TOKEN_BUDGET = 100000)...")
    df_final, tokens_final, _ = run_triage_pipeline(FEW_SHOT_PROMPT, token_budget=100000, log_retries=False)
    df_final_routed = apply_routing(df_final, SEGMENTS_PATH)
    df_final_routed.to_csv(RESULTS_OUTPUT_PATH, index=False)
    print(f"  Saved final triage results ({len(df_final_routed)} rows) to {RESULTS_OUTPUT_PATH}")
    print("\nFinal Routing Summary:")
    print(df_final_routed["route"].value_counts().to_string())
    print("\nPriority Summary:")
    print(df_final_routed["priority"].value_counts().to_string())


if __name__ == "__main__":
    main()
