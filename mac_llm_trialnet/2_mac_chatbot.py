"""
2_mac_chatbot.py — TrialNet Chatbot with RAG Memory + LLM-as-Judge

Commands:
  /correct [answer]   — manually log model's last mistake + your correction
  /clear              — clear conversation history
  /memory             — show mistake count
  /stats              — show adapter, memory, history stats
  /judge on|off       — toggle auto-judge (default: on)
  quit                — exit
"""

from __future__ import annotations
import json
import os
import re
from mlx_lm import load, generate
from memory.chroma_bank import ChromaMemoryBank
from memory.judge import judge_response
from tools.executor import execute_tool, TOOL_SCHEMAS

MODEL_ID     = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR  = "./mac_trialnet_3b_v4_adapter"
LEGACY_JSONL = "./mac_mistakes_memory.jsonl"

MAX_HISTORY_TURNS = 8   # sliding context window (32K ctx, 1.5B model)
JUDGE_ENABLED     = True
JUDGE_BAD_THRESH  = 5   # score <= this → auto-log as mistake
JUDGE_GOOD_THRESH = 8   # score >= this → log as positive example

CODE_KEYWORDS = {
    "write", "implement", "create", "build", "make", "code", "function",
    "class", "script", "program", "def ", "algorithm", "fix", "debug",
    "refactor", "add", "generate",
}

BASE_SYSTEM = (
    "You are TrialNet, a continuously self-improving AI assistant running on Apple Silicon. "
    "You learn from your mistakes. Answer accurately and concisely.\n"
    "When writing code: always provide complete, runnable code. Never truncate or say 'rest omitted'. "
    "Use Python unless another language is requested.\n"
    "Tool use rules: use 'calculator' for arithmetic/algebra only. "
    "Use 'python_exec' only when asked to RUN or TEST code — never when asked to WRITE or IMPLEMENT code. "
    "For coding tasks (write/implement/create/fix/explain), respond with code directly — no tool calls."
)


def needs_more_tokens(prompt: str) -> bool:
    return any(kw in prompt.lower() for kw in CODE_KEYWORDS)


# ── Startup ───────────────────────────────────────────────────────────────────

print("Initializing Error Memory Bank (ChromaDB)...")
memory = ChromaMemoryBank()

if os.path.exists(LEGACY_JSONL) and memory.count() == 0:
    n = memory.migrate_jsonl(LEGACY_JSONL)
    if n:
        print(f"  Migrated {n} legacy mistake(s) → ChromaDB")

print(f"Loading TrialNet [{os.path.basename(ADAPTER_DIR)}] via MLX...")
model, tokenizer = load(MODEL_ID, adapter_path=ADAPTER_DIR)

print("\n" + "=" * 58)
print(f"  TrialNet — adapter : {os.path.basename(ADAPTER_DIR)}")
print(f"  Memory bank        : {memory.count()} mistake(s)")
print(f"  Auto-judge         : {'ON' if JUDGE_ENABLED else 'OFF'}")
print("  /correct [ans]  /clear  /judge on|off  /stats  quit")
print("=" * 58 + "\n")

# ── State ─────────────────────────────────────────────────────────────────────

conversation_history: list[dict] = []
last_prompt   = ""
last_response = ""


# ── Chat ──────────────────────────────────────────────────────────────────────

def build_system_prompt(user_prompt: str) -> str:
    injection = memory.build_system_injection(user_prompt)
    return BASE_SYSTEM + ("\n\n" + injection if injection else "")


_TOOL_CALL_RE = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL)

MAX_TOOL_ROUNDS = 3  # prevent infinite loops


def _generate(messages: list[dict], max_tok: int) -> str:
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        tools=TOOL_SCHEMAS,
    )
    return generate(model, tokenizer, prompt=prompt_text, max_tokens=max_tok,
                    verbose=False)


def chat(user_prompt: str) -> str:
    system   = build_system_prompt(user_prompt)
    window   = conversation_history[-(MAX_HISTORY_TURNS * 2):]
    messages = [{"role": "system", "content": system}] + window + [
        {"role": "user", "content": user_prompt}
    ]
    max_tok = 1200 if needs_more_tokens(user_prompt) else 500

    for _ in range(MAX_TOOL_ROUNDS):
        response = _generate(messages, max_tok)
        match    = _TOOL_CALL_RE.search(response)
        if not match:
            return response  # no tool call — done

        # Parse and execute the tool
        try:
            call = json.loads(match.group(1))
            tool_name = call.get("name", "")
            tool_args = call.get("arguments", {})
        except json.JSONDecodeError:
            return response  # malformed — return as-is

        tool_result = execute_tool(tool_name, tool_args)
        print(f"  [Tool: {tool_name}({tool_args}) → {tool_result}]")

        # Append assistant tool-call turn + tool response, then loop
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "tool",      "content": tool_result, "name": tool_name})
        max_tok = 300  # short follow-up for final answer

    # Fallback: return last response if rounds exhausted
    return response


def run_judge(question: str, response: str) -> dict | None:
    """Score response, auto-log if bad, return verdict."""
    if not JUDGE_ENABLED:
        return None
    try:
        verdict = judge_response(question, response, model, tokenizer)
        score   = verdict["score"]

        if verdict.get("is_bad") and score <= JUDGE_BAD_THRESH:
            # is_bad=True AND low score — both required to avoid parse-error false positives
            memory.add_mistake(
                prompt=question,
                bad_generation=response,
                human_correction=f"[AUTO-FLAGGED] {verdict.get('reason', '')}",
            )
        elif score >= JUDGE_GOOD_THRESH:
            # Future: log as chosen pair for DPO
            pass

        return verdict
    except Exception:
        return None


# ── Main loop ─────────────────────────────────────────────────────────────────

while True:
    try:
        user_input = input("\nYou: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        break

    if not user_input:
        continue

    # ── Commands ──
    if user_input.lower() == "quit":
        break

    if user_input.lower() == "/clear":
        conversation_history.clear()
        last_prompt = last_response = ""
        print("  Conversation history cleared.")
        continue

    if user_input.lower() == "/memory":
        print(f"  Memory bank: {memory.count()} mistake(s) in ChromaDB")
        if memory.count() >= 10:
            print("  Ready for self-correction → run: bash run_self_correction.sh")
        continue

    if user_input.lower() in ("/judge on", "/judge off"):
        JUDGE_ENABLED = user_input.endswith("on")
        print(f"  Auto-judge: {'ON' if JUDGE_ENABLED else 'OFF'}")
        continue

    if user_input.lower() == "/stats":
        print(f"  Adapter  : {ADAPTER_DIR}")
        print(f"  Model    : {MODEL_ID}")
        print(f"  Memory   : {memory.count()} mistake(s)")
        print(f"  History  : {len(conversation_history) // 2} turn(s) in context")
        print(f"  Judge    : {'ON' if JUDGE_ENABLED else 'OFF'}")
        continue

    if user_input.startswith("/correct "):
        correction = user_input[len("/correct "):].strip()
        if not last_prompt:
            print("  Nothing to correct yet.")
        elif not correction:
            print("  Usage: /correct [the right answer]")
        else:
            memory.add_mistake(last_prompt, last_response, correction)
            print(f"  Logged. Memory bank: {memory.count()} mistake(s).")
            if memory.count() >= 10:
                print("  Ready → run: bash run_self_correction.sh")
        continue

    # ── Normal turn ──
    print("\nTrialNet: ", end="", flush=True)
    response = chat(user_input)
    display = re.sub(r'<thinking>.*?</thinking>\s*', '', response, flags=re.DOTALL).strip()
    print(display)

    # Accumulate history before judging
    conversation_history.append({"role": "user",      "content": user_input})
    conversation_history.append({"role": "assistant", "content": response})
    last_prompt   = user_input
    last_response = display  # store stripped version for /correct and judge

    # Judge runs after response is visible to user
    if JUDGE_ENABLED:
        print("  ", end="", flush=True)
        verdict = run_judge(user_input, display)
        if verdict:
            score  = verdict["score"]
            reason = verdict.get("reason", "")
            if verdict.get("is_bad") and score <= JUDGE_BAD_THRESH:
                print(f"[ Judge: {score}/10 ⚠  auto-logged — {reason} ]")
                print(f"  Use /correct [answer] to provide the real fix.")
            elif score <= JUDGE_BAD_THRESH:
                print(f"[ Judge: {score}/10 ⚠  — {reason} ]")
                print(f"  Use /correct [answer] if this is wrong.")
            elif score >= JUDGE_GOOD_THRESH:
                print(f"[ Judge: {score}/10 ✓ ]")
            else:
                print(f"[ Judge: {score}/10 ]")
