"""Quick coding eval for TrialNet v6. 6 focused prompts."""
import os, sys, re, json
from mlx_lm import load, generate
from tools.executor import execute_tool, TOOL_SCHEMAS

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER  = "./mac_trialnet_v8_adapter"

PROMPTS = [
    {"id": "recursive_fib",       "q": "Write a Python function `fib(n)` that returns the nth Fibonacci number using recursion with memoization."},
    {"id": "reverse_linked_list", "q": "Write a Python class `ListNode` and a function `reverse(head)` that reverses a singly linked list in-place. Return the new head."},
    {"id": "palindrome_ignore",   "q": "Write `is_palindrome(s: str) -> bool` that ignores case, spaces, and punctuation. Include 2 test cases."},
    {"id": "bug_fix",             "q": "Fix this buggy code: `def avg(xs): return sum(xs) / len(xs)` — make it handle an empty list gracefully and return None."},
    {"id": "binary_search",       "q": "Write iterative binary search `bsearch(arr, target) -> int` that returns the index or -1. Include edge cases in comments."},
    {"id": "oop_stack",           "q": "Implement a `Stack` class with push, pop, peek, is_empty, size. Raise IndexError on pop/peek from empty stack."},
]

SYS = "You are TrialNet. Answer accurately. When writing code: provide complete, runnable code. Use Python."
_TOOL_RE = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL)

def run(model, tok, messages, max_tok=800):
    for _ in range(3):
        pt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=TOOL_SCHEMAS)
        out = generate(model, tok, prompt=pt, max_tokens=max_tok, verbose=False)
        m = _TOOL_RE.search(out)
        if not m: return out
        try:
            call = json.loads(m.group(1))
            res = execute_tool(call.get("name", ""), call.get("arguments", {}))
        except Exception:
            return out
        messages.append({"role": "assistant", "content": out})
        messages.append({"role": "tool", "content": res, "name": call.get("name", "")})
    return out

def main():
    print(f"Loading {ADAPTER}...", flush=True)
    model, tok = load(MODEL_ID, adapter_path=ADAPTER)
    results = []
    for p in PROMPTS:
        print(f"\n=== {p['id']} ===")
        print(f"Q: {p['q']}")
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": p["q"]}]
        ans = run(model, tok, msgs)
        clean = re.sub(r'<thinking>.*?</thinking>\s*', '', ans, flags=re.DOTALL).strip()
        print(f"A: {clean}")
        results.append({"id": p["id"], "q": p["q"], "a": clean})
    adapter_name = os.path.basename(ADAPTER.rstrip("/"))
    os.makedirs("eval_results", exist_ok=True)
    out_path = f"eval_results/coding_test_{adapter_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")

if __name__ == "__main__":
    main()
