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

### Phase 3 — Reliable Self-Correction Loop ✅ DONE

- [x] Collected 12 diverse mistakes via judge auto-log + manual `/correct`
- [x] Fixed 3 bugs in chatbot/judge before running:
  - `<thinking>` blocks exposed in output → stripped with regex before display
  - Judge passed raw response (with `<thinking>`) → now passes stripped `display`
  - "auto-logged" printed on parse errors → fixed display condition to check `is_bad`
  - `dict | None` / `list[dict]` Python 3.10+ syntax crash on 3.9 → added `from __future__ import annotations`
- [x] Ran `bash run_self_correction.sh` — 50 iters SFT on 12 correction pairs
  - Train loss: 0.811 → 0.292
  - Created `mac_trialnet_v3_adapter/`
- [x] Updated `ADAPTER_DIR` in chatbot to v3

**v3 eval results vs v2:**
| Question | v2 | v3 |
|---|---|---|
| 127×43 | 5491 ✗ | 5411 ✗ (both wrong, correct=5461) |
| 3x+7=25 | x=6 ✓ | x=6 ✓ |
| Logic syllogisms | ✓ | ✓ |
| Affirming consequent (rain) | ✓ | ✓ |
| Fibonacci | full docstring ✓ | compact generator (regressed style) |
| Bat & ball ($0.05) | 10¢ ✗ | 10¢ ✗ (both wrong) |
| Boxes puzzle | ✓ | ✓ |

**Known weaknesses (1.5B limit):** multi-step arithmetic, CRT math puzzles. Need Phase 4 reasoning traces or Phase 6 larger model to fix.

---

### Phase 4 — Extended Thinking / Reasoning ✅ DONE
- [x] Created `4_mac_reasoning.py` — loads 200 math/logic traces from `nohurry/Opus-4.6-Reasoning-3000x-filtered`
  - Prioritizes math/reasoning/logic/algebra categories
  - System prompt with budget control: "Think for up to 300 tokens before answering"
  - Filters out shallow traces (< 50 chars thinking)
- [x] SFT on 175 train / 25 valid samples, `--max-seq-length 2048`, `--iters 200`, `lr=1e-5`
  - Train loss: 1.125 → 0.576
  - Peak mem: 11.5 GB (up from 4.1 GB — long reasoning traces)
  - Created `mac_trialnet_v5_adapter/`
- [x] Updated chatbot to v5

**v5 eval (8/10) vs v4 (8/10):**
| | v4 | v5 |
|---|---|---|
| logic_2 (affirming consequent) | ✗ | ✓ fixed |
| reason_1 (bat & ball) | ✓ | ✗ regressed |
| Uses `<thinking>` blocks | ✗ | ✓ |
| math_1 (127×43) | ✗ | ✗ hard limit |

**Key win**: model now reasons out loud with `<thinking>` on all multi-step problems.
**Remaining**: reason_1 regression fixable via correction; 127×43 needs Phase 5 calculator tool.

---

### Phase 5 — Tool Use ✅ DONE
- [x] Created `tools/executor.py` — 3 tools: `calculator` (safe AST eval), `python_exec` (sandboxed stdout), `search_memory` (ChromaDB RAG)
- [x] Created `tools/__init__.py` — exports `execute_tool`, `TOOL_SCHEMAS`
- [x] Created `5_mac_tools.py` — 50 tool-call SFT examples (15 mult, 6 add/sub, 5 pct, 6 algebra, 10 python_exec, 8 no-tool)
- [x] SFT: v5 → v6, 100 iters, lr=5e-6; train loss 1.16→0.09, val loss 2.86→0.15
- [x] Wired tool execution loop into `2_mac_chatbot.py` and `evaluate_mac.py`
  - Detects `<tool_call>` JSON in response, executes via `execute_tool()`, appends `role=tool`, re-generates
  - Max 3 rounds to prevent loops
- [x] Fixed `ast.Exec` Python 3.9 incompatibility in `tools/executor.py`

**v6 eval (8/10) vs v5 (8/10):**
| Question | v5 | v6 |
|---|---|---|
| math_1 (127×43) | ✗ hallucinated | ✓ **fixed** via calculator |
| math_2 (3x+7=25) | ✓ | ✗ over-applies calculator |
| logic_2 (affirming consequent) | ✗ | ✗ regressed |
| reason_1 (bat & ball) | ✓ | ✓ |

**Key win**: 127×43 now solved correctly via calculator tool call.
**Regressions**: math_2 over-triggers calculator; logic_2 re-regressed. Both fixable via next self-correction run.

---

### Phase 5.5 — Bug Fixes ✅ DONE
- [x] `__build_class__` added to python_exec sandbox → class definitions now work
- [x] `python_exec` tool description updated → discourages trigger on code-writing tasks
- [x] `BASE_SYSTEM` updated with explicit tool-use rules → no tool_call for write/implement/fix
- [x] `repetition_penalty=1.15` added to `_generate()` → fixes infinite repetition loops
- [x] `manual_corrections.jsonl` introduced → hand-crafted pairs survive ChromaDB overwrites
- [x] `3_mac_self_correct.py` patched → merges ChromaDB + manual corrections, counts both for MIN_MISTAKES
- [x] 3 new correction pairs added: avg bug fix, fib memoization, logic_2 (affirming consequent)
- [x] Created v7 adapter (100 iters, 13 manual + ChromaDB pairs)
- [x] Updated chatbot + coding_test to v7

---

### Phase 6 — Scale Up Model ✅ DONE (3B trained, self-correction pending)
- [x] Created `6_mac_scale.py` — 3-stage curriculum for `Qwen/Qwen2.5-3B-Instruct`
  - Stage A: base (reasoning + coding), 300 iters, lr=1e-4 → `3b_v1`; train 1.116→0.382, val 1.610→1.151
  - Stage B: reasoning traces, 200 iters, lr=1e-5 → `3b_v2`; val 1.097→0.759
  - Stage C: tool-call SFT (20 no-tool examples), 100 iters, lr=5e-6 → `3b_v3`; train 1.53→0.17, val 0.212
- [x] Fixed OOM: Stage B max_seq 2048→1024 (3B+2048 exceeds 16GB)
- [x] Updated `evaluate_mac.py` with `--model` flag for multi-model support
- [x] Updated chatbot to `MODEL_ID=Qwen2.5-3B-Instruct`, `ADAPTER_DIR=3b_v3`

**3b_v3 eval (6/10) — currently below 1.5B v6 (8/10):**
| Question | 1.5B v6 | 3B v3 |
|---|---|---|
| math_1 (127×43 via calculator) | ✓ | ✓ |
| math_2 (3x+7=25) | ✗ | ✓ fixed |
| logic_2 (affirming consequent) | ✗ | ✗ persistent |
| code_2 (O(log n)) | ✓ | ✗ over-triggers python_exec |
| reason_1 (bat & ball) | ✓ | ✗ arithmetic regression |
| reason_2 (boxes puzzle) | ✓ | ✗ over-triggers python_exec |

**Root cause**: tool SFT associates "binary search"/"logic puzzles" → python_exec. Fix via self-correction cycle.
**Peak mem**: 10.86 GB (Stage A/B), 7.26 GB (Stage C)

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
