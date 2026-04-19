# TrialNet Self-Learning LLM — Build Roadmap

Engineered plan to reach working self-learning **coding LLM**. Mac-first (M4 16GB, MLX), escalate to Colab (T4 15GB VRAM) when hardware caps hit.

**Domain:** code generation + bug-fix. Model writes code, runs in sandbox, learns from failing tests + human `/correct`.

**Rules of engagement**
- Each chunk is a PR-sized unit. Don't start next until current acceptance passes.
- Keep `mac_mistakes_memory.jsonl` as single source of truth for errors across both environments.
- Version every adapter: `v{n}_{stage}_adapter/` (e.g. `v3_dpo_adapter`).
- Log eval scores per version into `eval/scores.jsonl` before promoting.

---

## PHASE A — MAC (M4 16GB, MLX)

### Chunk 1 — Base Model Decision (Code-Tuned)
**Goal:** lock code-specialized base model for Mac phase.
**Steps**
- Swap `Qwen2.5-1.5B-Instruct` → **`Qwen2.5-Coder-1.5B-Instruct`** for smoke-test.
- Primary target: **`Qwen2.5-Coder-3B-Instruct`** (best size/quality for M4 16GB).
- Stretch: `Qwen2.5-Coder-7B-Instruct-4bit` (MLX-quant, ~5GB RAM, slower).
- Parameterize `MODEL_ID` via env var / CLI flag.
- Rationale: Coder variant pretrained on 5.5T code tokens → strong HumanEval/MBPP baseline, already knows syntax for 90+ langs.
**Deliverable:** `mac_llm_trialnet/config.py` with `MODEL_ID`, `ADAPTER_DIR`, `DATA_DIR`, `MAX_SEQ_LEN=4096` (code needs longer ctx), `STOP_TOKENS=["```\n", "</code>"]`.
**Accept:** `python 1_mac_finetune.py --model coder-3b` loads + runs 10-step dry train without OOM.

### Chunk 2 — Code Dataset Assembly
**Goal:** clean SFT dataset mixing reasoning + code instruction-following.
**Steps**
- Mix these open corpora (all permissive):
  - `ise-uiuc/Magicoder-OSS-Instruct-75K` — real OSS code + instruct pairs.
  - `ise-uiuc/Magicoder-Evol-Instruct-110K` — evolved harder problems.
  - `nickrosh/Evol-Instruct-Code-80k-v1` — algorithmic diversity.
  - `HuggingFaceH4/CodeAlpaca_20K` — short fast examples.
  - Keep small slice of `Opus-4.6-Reasoning` (10%) to preserve `<thinking>` habit.
- Target format: `{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"<thinking>plan</thinking>\n```python\n...\n```"}]}`.
- Filter: drop samples without fenced code block, drop > `MAX_SEQ_LEN` tokens, dedup by prompt hash.
- Split 90/5/5 → `train.jsonl` / `valid.jsonl` / `test.jsonl`.
- Reserve 164 HumanEval + 500 MBPP prompts as **held-out eval set** (never train on these).
**Deliverable:** `mac_data/{train,valid,test}.jsonl`, `mac_data/heldout_eval.jsonl`, `mac_data/STATS.md` (lang distribution, avg tokens, source mix).
**Accept:** `mlx_lm.lora --train --data mac_data` loads without schema errors; STATS shows ≥ 30% Python, ≥ 20% multi-lang.

### Chunk 3 — SFT Baseline (v1 code adapter)
**Goal:** train first code-specialized adapter + record metrics.
**Steps**
- LoRA config: rank **16** (bumped from 8 — code needs more capacity), alpha 32, dropout 0.05, lr 1e-5, 500 iters, warmup 50.
- Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (full attention + FFN).
- Save → `v1_code_sft_adapter/`.
- Capture train/val loss → `eval/v1_loss.png`.
- Smoke test: 20 prompts from test split + 10 from HumanEval, log outputs.
**Accept:** val loss < base model val loss; ≥ 70% outputs contain parseable fenced code; `<thinking>` appears ≥ 50%.

### Chunk 4 — Code Eval Harness (executable)
**Goal:** objective scoring via actual code execution, not token-match.
**Steps**
- Build `eval/run_eval.py`:
  - Loads adapter, runs held-out HumanEval (164) + MBPP subset (100).
  - Extracts fenced code block from model output (regex `\`\`\`(?:python)?\n(.*?)\`\`\``).
  - Executes in **subprocess sandbox** (`python -c`, 10s timeout, `resource.setrlimit` mem cap 512MB, no net).
  - Runs provided unit tests against generated code.
  - Metrics: **pass@1**, **pass@10** (k samples, temp 0.8), syntax-valid %, avg latency tok/s, avg reasoning length.
- Additional metrics: compile rate (AST parse %), mean function length.
- Append row to `eval/scores.jsonl`: `{adapter, pass@1, pass@10, syntax_ok, tok_s, date}`.
- Safety: sandbox must reject `import os`, `subprocess`, `socket`, file writes outside `/tmp/eval_sandbox/`.
**Deliverable:** `eval/run_eval.py`, `eval/sandbox.py`, `eval/humaneval.jsonl`, `eval/mbpp.jsonl`.
**Accept:** `python eval/run_eval.py v1_code_sft_adapter` produces row with pass@1 ≥ 20% (Coder-3B baseline is ~35%, expect drop initially post-SFT then recovery).

### Chunk 5 — Coding Chatbot + Auto Test Runner + /correct
**Goal:** live mistake logger driven by failing code executions.
**Steps**
- Harden `2_mac_chatbot.py`:
  - REPL accepts coding task. Model generates code. Chatbot auto-extracts + runs via sandbox from Chunk 4.
  - **Auto-fail capture:** if tests fail OR syntax error OR timeout → log as mistake automatically (no human needed).
  - Manual `/correct <fixed code>` for cases without tests.
  - Log entry: `{prompt, model_output, correction, test_output, error_type, lang, timestamp, adapter_version}`.
  - Commands: `/run` (re-execute), `/test <pytest snippet>` (attach user test), `/undo`, `/list`.
- Dedup on `hash(prompt + lang)`.
**Accept:** 20 test sessions → mistakes JSONL has auto-captured failures + manual corrections, zero malformed rows.

### Chunk 6 — Code DPO Pair Builder (test-verified)
**Goal:** turn failing-code mistakes into high-signal DPO pairs.
**Steps**
- Script `mac_llm_trialnet/build_dpo_pairs.py`:
  - Reads `mac_mistakes_memory.jsonl`.
  - Emits `{prompt, chosen, rejected}`: `rejected = broken code`, `chosen = corrected code`.
  - **Verification step:** re-run `chosen` in sandbox; drop pair if `chosen` also fails tests. Guarantees positive signal only.
  - Augment: for each verified pair, generate 2 paraphrases of prompt via template (preserves code, boosts count).
  - Filter: drop if AST of chosen == AST of rejected (no real change), or len < 10.
**Accept:** ≥ 100 verified pairs (test-passing `chosen`) before Chunk 7.

### Chunk 7 — DPO Self-Correction (v2 code adapter)
**Goal:** train smarter code adapter from verified mistake pairs.
**Steps**
- Use MLX-LM DPO if available; else port this one step to Colab (TRL `DPOTrainer`).
- Init from `v1_code_sft_adapter` weights, not base. Low lr (5e-6), 1-2 epochs, beta 0.1.
- Output → `v2_code_dpo_adapter/`.
- After train: run full eval harness (Chunk 4) on HumanEval + MBPP.
**Accept:** `pass@1` of v2 > v1 by ≥ 2 points; no regression > 3 points on syntax_ok; latency within 10% of v1.

### Chunk 8 — RAG Over Past Bug-Fixes
**Goal:** model recalls prior fixes before generating new code.
**Steps**
- Embed each mistake via code-aware embed model: `jinaai/jina-embeddings-v2-base-code` or `nomic-embed-text-v1.5` (local MLX).
- Store in ChromaDB (persistent, SQLite backend) with metadata: `{lang, error_type, adapter_version}`.
- On new prompt: top-k=3 similar past prompts → inject as `# Lessons from past mistakes:` prefix with before/after snippets.
- Filter retrieval by `lang` match when detectable.
- Gate behind `--use-memory-rag` flag.
**Accept:** on 10 known-repeat bug patterns, pass@1 with RAG ≥ pass@1 without RAG by ≥ 5 points.

### Chunk 9 — Test Executor + LLM-as-Judge (hybrid)
**Goal:** remove human from correction loop using tests first, judge second.
**Steps**
- **Primary signal: test execution** (free, deterministic). If tests pass → correct. If fail → mistake, log failing test output as error context.
- **Fallback judge** for prompts without tests: Claude Opus 4.7 API or Haiku 4.5 (cheap).
  - Prompt judge with: task, model code, judge returns `{correct: bool, issues: [...], ideal_code: str, test_cases: [...]}`.
  - Judge-generated test cases get saved + reused for future prompts.
- On `correct=false`: auto-append to mistakes JSONL with `source=test` or `source=judge`.
**Accept:** 100 auto-judged prompts, spot-check 20 → judge agreement ≥ 85%; test-driven signal agreement 100% (ground truth).

### Chunk 10 — Continuous Loop Orchestrator
**Goal:** daemon that generates coding tasks, auto-tests, self-trains.
**Steps**
- `4_mac_orchestrator.py`:
  - Task generator: pulls problems from `mac_data/heldout_eval.jsonl` slice + synthesizes variants via template.
  - Loop: generate code → sandbox run → log failures → when ≥ N new mistakes (default 50) → build DPO pairs → train → eval → **promote only if pass@1 improves**.
  - Keep last 3 adapters, prune older.
  - Cooldown: don't retrain within 30min of last promote.
- Log every cycle to `logs/orchestrator.jsonl`: `{cycle_id, new_mistakes, pairs_built, train_loss, pass@1_before, pass@1_after, promoted}`.
**Accept:** run 24h on live stream, produces ≥ 1 promoted adapter autonomously with measurable pass@1 gain.

---

## PHASE B — COLAB MIGRATION (Free T4, 15GB VRAM)

Escalate here when Mac hits ceiling: need 7B base, bigger LoRA, or longer context.

### Chunk 11 — Colab Env Bootstrap
**Steps**
- Notebook `colab/00_setup.ipynb`: install `bitsandbytes`, `peft`, `trl`, `transformers`, `accelerate`, `datasets`, `chromadb`.
- Mount Google Drive → `/content/drive/MyDrive/trialnet/` as persistent root.
- Copy `mac_mistakes_memory.jsonl` + `mac_data/` into Drive.
**Accept:** `import torch; torch.cuda.is_available()` = True + Drive mounted.

### Chunk 12 — 7B Code SFT on Colab
**Steps**
- Notebook `colab/01_sft.ipynb`: **`Qwen2.5-Coder-7B-Instruct`** 4-bit via `bitsandbytes`.
- LoRA rank 32, alpha 64, target `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.
- Train on same `train.jsonl` from Phase A (code dataset).
- Save adapter to Drive → `v1_colab_code_sft_adapter/`.
- Checkpoint every 100 steps (Colab kill risk).
**Accept:** pass@1 on HumanEval > Mac v2; VRAM stays < 14GB.

### Chunk 13 — DPO on Colab
**Steps**
- Notebook `colab/02_dpo.ipynb`: `DPOTrainer` from TRL.
- Load same pairs file as Mac phase.
- Train → `v2_colab_dpo_adapter/`.
**Accept:** eval harness (ported) shows improvement vs v1_colab.

### Chunk 14 — Unified Eval + Model Registry
**Steps**
- Single `eval/registry.json` tracks every adapter (Mac + Colab) with scores + metadata.
- Decide promotion rules: which adapter serves inference.
**Accept:** CLI `python eval/promote.py` prints winner + reason.

### Chunk 15 — Deployment Target
**Steps**
- Serve best adapter via:
  - Mac: MLX server on localhost.
  - Colab: `gradio` public URL for demo.
- Chatbot UI in `dashboard/` points to whichever is live.
**Accept:** End-to-end: user asks → inference → bad answer → `/correct` → auto-DPO trigger next cycle → improved answer on retry.

---

## Milestone Summary

| Milestone | Chunks | Outcome |
|-----------|--------|---------|
| **M1 — Clean Baseline** | 1–4 | Reproducible v1 SFT + eval harness |
| **M2 — Mistake Loop** | 5–7 | v2 DPO adapter from human corrections |
| **M3 — RAG + Auto-Judge** | 8–9 | No-human self-correction |
| **M4 — Autonomous** | 10 | Daemon promotes adapters on its own |
| **M5 — Colab Scale** | 11–13 | 7B base, same pipeline |
| **M6 — Shipped** | 14–15 | Registry + serving + demo UI |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| MLX lacks DPO support | Port only DPO step to Colab early, keep SFT on Mac |
| 1.5B too weak for `<thinking>` | Upgrade to 3B on Mac (Chunk 1), fallback 7B on Colab |
| Colab session kills mid-train | Checkpoint every 100 steps to Drive |
| Mistakes JSONL drift between envs | Single file in Drive; Mac syncs via `rclone` or manual copy |
| Eval score regression on v2 | Keep v1 frozen as fallback; promote only on beat |
| Adapter bloat | Prune: keep base + last 3 adapters |
| Sandbox escape via generated code | Strict subprocess isolation, seccomp/rlimit, no net, ephemeral dir, 10s timeout |
| Catastrophic forgetting (code SFT kills general reasoning) | Keep 10% Opus-Reasoning mix; eval reasoning benchmark alongside pass@1 |
| Data contamination (train leaks HumanEval) | Hash-filter: drop any train sample overlapping held-out by ≥ 10 tokens |

---

## What to do right now

1. Chunk 1 — swap to `Qwen2.5-Coder-3B-Instruct`, extract `config.py`.
2. Chunk 4 — code eval harness (HumanEval + sandbox). Without it, pass@1 can't be measured → no promotion signal.
3. Chunk 5 — chatbot with auto test runner (current JSONL = 1 line, loop untested).

**North-star metric:** `pass@1` on HumanEval. Every adapter promotion must improve it.
