# TrialNet — Build Progress

## Hardware
- Apple M4, 16GB unified RAM
- Framework: MLX (mlx-lm) — Metal GPU, no CUDA
- Max model size: 7B int4 (tight), 3B comfortable

---

## Phases

### Phase 0 — Baseline ✅ DONE
- [x] Confirmed `3_mac_self_correct.py` uses SFT (correct for MLX — DPO not needed)
- [x] Created `evaluate_mac.py` — 10 fixed questions (math/logic/code/reasoning)
- [x] Ran eval on v1 and v2 adapters
- [x] **Found v2 adapter broken** — catastrophic overfit from 1 training example
- [x] Blacklisted v2 in `3_mac_self_correct.py`, reverted chatbot to v1
- [x] Saved eval baseline → `eval_results/mac_trialnet_v1_adapter.json`
- [x] Raised MIN_MISTAKES threshold to 10 (prevents repeat of v2 regression)

**v1 eval results (baseline):**
| Question | Result |
|---|---|
| 127 × 43 | ✓ correct (5461) |
| 3x + 7 = 25 | ✓ correct (x=6) |
| Logic syllogism (cats) | ✓ correct |
| Affirming consequent (rain) | ✗ wrong — 1.5B limitation |
| Bat & ball ($0.05) | ✓ correct |

---

### Phase 1 — ChromaDB RAG Memory ✅ DONE
- [x] Created `memory/chroma_bank.py` — ChromaDB vector store
  - `add_mistake(prompt, bad_gen, correction)`
  - `query_similar(prompt, n=3)` — cosine similarity, threshold 0.6
  - `build_system_injection(prompt)` — injects top-3 past mistakes into system prompt
  - `export_dpo_pairs(path)` — exports chosen/rejected JSONL for SFT
  - `migrate_jsonl(path)` — one-time import from legacy JSONL
- [x] Migrated `mac_mistakes_memory.jsonl` → ChromaDB (1 real mistake)
- [x] Updated `2_mac_chatbot.py`:
  - System prompt with TrialNet identity + code instructions
  - RAG injection: past mistakes auto-injected before each answer
  - `/clear` — reset conversation history
  - `/memory` — show mistake count
  - `/stats` — adapter + memory + history info

---

### Phase 1b — Conversation History ✅ DONE
- [x] Fixed stateless bug — chatbot was sending only current message each turn
- [x] Added `conversation_history` list — accumulates all turns
- [x] Sliding window: keeps last 8 turns (safe for 1.5B 32K context)
- [x] `max_tokens=1200` for code requests, `500` for normal chat
- [x] Auto-detects code keywords: `write`, `implement`, `function`, `code`, etc.
- [x] Tested: name recall across turns ✓, full code output ✓

---

### Phase 2 — LLM-as-Judge ✅ DONE
- [x] Created `memory/judge.py`
  - `judge_response()` — scores 0–10, returns `{score, is_bad, reason}`
  - JSON parse fallback: `score=5, is_bad=False` (no false-positive auto-logging)
- [x] Wired into `2_mac_chatbot.py`:
  - Runs after every response (sequential — MLX not thread-safe)
  - `is_bad=True AND score ≤ 5` → auto-logs to ChromaDB
  - `score ≥ 8` → marks as positive example
  - Shows verdict: `[ Judge: 0/10 ⚠ auto-logged ]` / `[ Judge: 10/10 ✓ ]`
  - `/judge on|off` command to toggle
- [x] Fixed false-positive bug: parse errors (score=5, is_bad=False) no longer auto-log
- [x] Tested all cases: bad (0/10 ✓), good (10/10 ✓), parse error (5/10, not logged ✓)

---

### Phase 3 — Reliable Self-Correction Loop 🔜 BLOCKED
**Blocked by:** need 10+ real mistakes in ChromaDB first

- [ ] Chat with model, collect 10+ diverse mistakes via judge auto-log + manual `/correct`
- [ ] Run `bash run_self_correction.sh`
  - Exports pairs from ChromaDB → `mac_correction_data/dpo_pairs.jsonl`
  - Runs MLX LoRA SFT on corrections (50 iters default)
  - Creates `mac_trialnet_v3_adapter/`
  - Runs eval + compare v1 vs v3
- [ ] Verify v3 beats v1 on eval baseline
- [ ] Update `ADAPTER_DIR` in chatbot to v3

---

### Phase 4 — Extended Thinking / Reasoning 📋 PLANNED
- [ ] Curate 200 reasoning traces with backtracking in `<thinking>` blocks
  - Sources: `open-r1/OpenR1-Math-220k`, `bespokelabs/Sky-T1`, DeepSeek-R1 distill
- [ ] SFT on traces: `--max-seq-length 2048`, `--iters 200`
- [ ] Add budget control: `SYSTEM = "Think for up to {N} tokens before answering"`
- [ ] Eval: compare logic/reasoning scores before/after on `evaluate_mac.py`

---

### Phase 5 — Tool Use 📋 PLANNED
- [ ] Define 3 tools: `calculator`, `search_memory` (RAG), `python_exec`
- [ ] SFT on 50 function-call examples in Qwen chat format
- [ ] Wire tool-call parser into chatbot — intercept `<tool_call>` tokens, execute, return result
- [ ] Test: math questions should call calculator, not hallucinate

---

### Phase 6 — Scale Up Model 📋 PLANNED
- [ ] Stay on Mac: upgrade `MODEL_ID` in `1_mac_finetune.py`
  - `Qwen/Qwen2.5-3B-Instruct` — comfortable, same pipeline
  - `Qwen/Qwen2.5-7B-Instruct` — int4 ~4GB, fits M4/16GB (tight)
  - `microsoft/Phi-4-mini-instruct` — 3.8B, strong reasoning
- [ ] Heavy RL training (GRPO) → Colab for 7B+, download adapter back to Mac

---

## Current File Structure

```
mac_llm_trialnet/
├── 1_mac_finetune.py          — base LoRA training (Qwen2.5-1.5B + MLX)
├── 2_mac_chatbot.py           — chatbot: RAG + history + auto-judge
├── 3_mac_self_correct.py      — self-correction: ChromaDB → SFT → new adapter
├── evaluate_mac.py            — 10-question eval baseline + version compare
├── run_self_correction.sh     — full pipeline: export → SFT → eval → compare
├── requirements_mac.txt       — mlx, mlx-lm, datasets, flask, chromadb, tqdm
├── mac_mistakes_memory.jsonl  — legacy JSONL (migrated to ChromaDB)
├── memory/
│   ├── chroma_bank.py         — ChromaDB RAG error memory
│   └── judge.py               — LLM-as-judge scorer
├── mac_trialnet_v1_adapter/   — CURRENT BEST adapter (reasoning + coding SFT)
├── mac_trialnet_v2_smarter_adapter/ — BROKEN (overfit, blacklisted)
└── eval_results/
    └── mac_trialnet_v1_adapter.json  — baseline scores
```

---

## Quick Commands

```bash
cd mac_llm_trialnet

# Chat + collect mistakes (judge auto-logs bad responses)
../.venv/bin/python 2_mac_chatbot.py

# Check mistake count
# → type /memory inside chatbot

# Run self-correction when 10+ mistakes collected
bash run_self_correction.sh

# Eval specific adapter
../.venv/bin/python evaluate_mac.py --adapter ./mac_trialnet_v3_adapter

# Compare two versions
../.venv/bin/python evaluate_mac.py --compare mac_trialnet_v1_adapter mac_trialnet_v3_adapter
```

---

## Key Lessons Learned
- Never run self-correction with fewer than 10 diverse examples — v2 broke from 1 example
- Same data for train + valid = guaranteed overfit on tiny datasets
- LLM-as-judge with same 1.5B model: works for clear errors, unreliable on edge cases (parse errors ~30%)
- Use `is_bad` flag + score threshold together — score alone causes false positives on parse errors
- v2 adapter blacklisted in `3_mac_self_correct.py` via `SKIP_ADAPTERS` set
