# 🧠 Building a Self-Learning AI Model
### A Complete Technical Playbook — From Base Model to Autonomous Improvement

> *Inspired by techniques used in Claude, GPT, DeepSeek-R1, and frontier AI systems.*

---

## Table of Contents

1. [The Big Picture](#the-big-picture)
2. [Step 1 — Choose Your Base Model](#step-1--choose-your-base-model)
3. [Step 2 — Data Pipeline (Pre-Training Hygiene)](#step-2--data-pipeline-pre-training-hygiene)
4. [Step 3 — Fine-Tuning Techniques](#step-3--fine-tuning-techniques)
5. [Step 4 — Alignment Techniques](#step-4--alignment-techniques)
6. [Step 5 — Reasoning & Extended Thinking](#step-5--reasoning--extended-thinking)
7. [Step 6 — Tool Use & Agentic Training](#step-6--tool-use--agentic-training)
8. [Step 7 — Long Context Extension](#step-7--long-context-extension)
9. [Step 8 — Multimodal Capabilities](#step-8--multimodal-capabilities)
10. [Step 9 — Self-Learning Features](#step-9--self-learning-features)
11. [Step 10 — Inference-Time Scaling](#step-10--inference-time-scaling)
12. [Step 11 — Safety Stack](#step-11--safety-stack)
13. [Step 12 — Evaluation Harness](#step-12--evaluation-harness)
14. [Step 13 — Model Compression](#step-13--model-compression-run-on-low-spec-hardware)
15. [Step 14 — Serving & Deployment Optimizations](#step-14--serving--deployment-optimizations)
16. [Full Pipeline Summary](#full-pipeline-summary)
17. [Realistic Roadmap](#realistic-roadmap)
18. [Key Warnings & Pitfalls](#key-warnings--pitfalls)

---

## The Big Picture

```
Pretrained Base Model  (foundation — language, reasoning, world knowledge)
        ↓
   Fine-Tuning         (teach it YOUR domain and style)
        ↓
   Alignment           (make it behave the way you actually want)
        ↓
   Self-Learning Loop  (let it keep improving autonomously)
        ↓
   Compression         (make it run on small hardware)
        ↓
   Your Custom Model ✅
```

**Core insight:** You never train from scratch. You stand on the shoulders of giants — take a pretrained model, reshape its behaviour, then build mechanisms that help it improve over time.

---

## Step 1 — Choose Your Base Model

The base model already knows language, reasoning, and general knowledge from training on billions of tokens. Your job is to redirect that capability.

### Recommended Open-Source Options

| Model | Size Options | Best For | License |
|---|---|---|---|
| **Llama 3.1 / 3.3** (Meta) | 8B, 70B, 405B | General purpose, strong baseline | Custom (open) |
| **Mistral / Mixtral** | 7B, 8×7B MoE | Fast inference, efficient | Apache 2.0 |
| **Gemma 3** (Google) | 2B, 9B, 27B | Lightweight, good reasoning | Custom |
| **Qwen 2.5** (Alibaba) | 7B–72B | Multilingual + coding | Apache 2.0 |
| **Phi-4** (Microsoft) | 14B | Surprisingly strong for size | MIT |
| **DeepSeek-R1 / V3** | 7B–671B (MoE) | Strong reasoning, RL-trained | MIT |
| **Llama 4** (Meta) | 17B–400B MoE | Native multimodal, 1M+ context | Custom |
| **Qwen 2.5-VL / Qwen3** | 7B–72B | Vision + language, strong tool use | Apache 2.0 |

### Dense vs Mixture-of-Experts (MoE)

Frontier models (Claude 3.5+, DeepSeek-V3, Mixtral, Llama 4) are **MoE**: a router picks *k* expert FFN blocks per token out of *N* total. You get the capacity of a 400B model with the compute cost of a 30B dense forward pass.

```
Dense:    every token → every parameter (expensive, no specialization)
MoE:      every token → top-2 of 128 experts (cheap, specialized)
```

**Trade-offs:** MoE needs more VRAM (all experts resident), trickier training (load-balancing loss, expert collapse), but dominates on compute-efficiency. Use `Mixtral 8x22B`, `DeepSeek-V3`, or `Qwen3-MoE` if you want Opus-tier quality-per-FLOP.

### Choosing by Hardware

```
4GB  VRAM  →  3B–7B  model (int4 quantized)
8GB  VRAM  →  7B     model (int4 quantized)
16GB VRAM  →  13B    model (int4 quantized)
32GB RAM   →  7B     model on CPU via llama.cpp
64GB RAM   →  13B    model on CPU via llama.cpp
```

---

## Step 2 — Data Pipeline (Pre-Training Hygiene)

Frontier labs spend more engineering on data than on architecture. Model quality ≈ data quality. Before any fine-tuning:

### 2.1 Deduplication
Exact + near-duplicate removal via MinHash-LSH or SimHash. Duplicates cause memorization, wasted steps, and eval contamination. Target: <1% duplicate rate.

### 2.2 Decontamination
Scrub benchmark strings (MMLU, GSM8K, HumanEval, SWE-bench, GPQA, ARC) from training corpus using n-gram overlap (13-gram is standard). Contaminated models look great on benchmarks and fail in the wild.

### 2.3 Quality Filtering
- **Perplexity filtering** — small model scores each doc; drop high-perplexity (noise) and suspiciously low (boilerplate)
- **Classifier filtering** — fastText or small BERT trained on "good" examples (Wikipedia, curated books) vs CommonCrawl
- **Heuristics** — line-length distribution, symbol ratios, language ID

### 2.4 Synthetic Data Generation ⭐
Opus-tier models are trained heavily on synthetic data from stronger teachers. Pipelines worth copying:

| Pipeline | What it generates |
|---|---|
| **Evol-Instruct** | Iteratively rewrites seed prompts to increase difficulty/depth |
| **Self-Instruct** | Model generates its own instruction-response pairs from seed tasks |
| **Magpie** | Prompts aligned model with empty user turn → it hallucinates prompt + answer |
| **OSS-Instruct** | Seed with real code snippets, generate realistic coding problems |
| **Persona-Hub** | Condition generation on 1B+ personas for diversity |

Rule: **filter synthetic data harder than real data**. Use reward models, verifiers, and execution feedback to drop bad samples.

### 2.5 Data Mixing
Weight sources by quality, not volume. Curriculum: start with diverse web + books, shift toward code + math + reasoning + domain data in later epochs. Domain up-weighting in the final 10% of training ("annealing") is the frontier trick for targeted capability gains.

---

## Step 3 — Fine-Tuning Techniques

Fine-tuning takes the pretrained model and specializes it on your data.

---

### 2.1 Supervised Fine-Tuning (SFT)

Teach the model to respond the way you want by training on input-output pairs.

```
Input:  "Summarize this report in 3 bullet points."
Output: "• Point 1\n• Point 2\n• Point 3"
```

**When to use:** Domain specialization, style customization, task-specific behaviour.

**Tools:** Hugging Face `trl`, Axolotl, LlamaFactory

---

### 2.2 LoRA — Low-Rank Adaptation ⭐ (Most Practical)

Instead of updating all billions of parameters (expensive, risks forgetting), LoRA adds tiny trainable matrices alongside the frozen original weights.

```
Original weights (frozen):  W
LoRA adds:                  ΔW = A × B
                            (A and B are tiny low-rank matrices)
Final output:               W + ΔW
```

**You train only ~0.1–1% of parameters.** The base model's broad knowledge stays intact.

**Key benefit:** You can train multiple LoRA adapters for different tasks and swap them in/out on the same base model without storing multiple full models.

---

### 2.3 QLoRA — Quantized LoRA ⭐⭐ (Best for Consumer Hardware)

QLoRA = quantize the base model to 4-bit precision first, then apply LoRA on top.

```
Base model:    70B parameters × 4 bits = ~35GB  (fits on 2× A100 or large consumer GPU)
LoRA adapter:  ~100MB extra
Total:         ~35.1GB
```

**Result:** Fine-tune a 70B model on hardware that would normally only fit a 7B model. Quality loss is minimal.

---

### 2.4 Instruction Tuning

A specific type of SFT where you train the model to follow structured instructions.

```markdown
### Instruction:
Explain quantum entanglement to a 10-year-old.

### Response:
Imagine you have two magic coins...
```

This transforms a raw language model (which just predicts next tokens) into an assistant that follows commands. Popular datasets: Alpaca, OpenHermes, Dolly.

---

## Step 4 — Alignment Techniques

Fine-tuning makes the model capable. Alignment makes it **behave the way you actually want** — honest, helpful, safe, and not harmful.

---

### 3.1 RLHF — Reinforcement Learning from Human Feedback
*(Used by ChatGPT, Claude, and most frontier models)*

**Three stages:**

**Stage 1 — Collect human preferences:**
```
Same prompt → two model outputs → human picks which is better
Repeat thousands of times
```

**Stage 2 — Train a Reward Model:**
```
Reward Model learns to predict human preference scores
Input:  model output
Output: quality score (0.0 → 1.0)
```

**Stage 3 — PPO (Proximal Policy Optimization):**
```
Your model generates a response
Reward model scores it
PPO updates your model to maximize the score
KL penalty prevents gaming the reward model
```

> **KL penalty is critical** — without it the model learns to trick the reward model rather than actually improve.

---

### 3.2 Constitutional AI (CAI) — Anthropic's Approach ⭐
*(This is what makes Claude, Claude)*

Instead of relying only on expensive human labelers, CAI uses the model itself to critique outputs against a written set of principles — a "constitution."

**Phase 1 — Supervised (SL-CAI):**

```
1. Model generates a response
2. Model critiques it: "Is this response honest? Does it respect autonomy?"
3. Model revises the response
4. Train on the revised responses
```

**Phase 2 — RL from AI Feedback (RLAIF):**

```
Instead of humans comparing outputs →
AI feedback model (trained on the constitution) generates preference labels
Much more scalable and consistent than human labeling
```

**Your own constitution example:**
```
Principle 1: Always be honest, even when the truth is uncomfortable.
Principle 2: Prioritize user safety above task completion.
Principle 3: Never fabricate facts — say "I don't know" when uncertain.
Principle 4: Respect user autonomy — give information, not orders.
```

You define the principles. The model learns to follow them without human labelers for every example.

---

### 3.3 DPO — Direct Preference Optimization ⭐ (Simpler than RLHF)

RLHF is complex — training a separate reward model adds many moving parts. DPO eliminates the reward model entirely.

```
Training data format:
{
  "prompt":   "Explain climate change",
  "chosen":   "Climate change refers to...",   ← what you WANT
  "rejected": "Climate change is fake..."       ← what you DON'T want
}
```

DPO mathematically proves that directly optimizing on these preference pairs is equivalent to RLHF — but in one clean step. 

**Result:** Better alignment with ~10× less implementation complexity.

**Tools:** Hugging Face `trl` has DPO built in — ~20 lines of code.

---

### 3.4 RLEF — Reinforcement Learning from Environment Feedback

Instead of human feedback, the reward signal comes from the environment itself.

```
Code model:   reward = does the code compile and pass tests? (0 or 1)
Math model:   reward = is the numerical answer correct?
Agent model:  reward = did the agent complete the task successfully?
```

**Why this is powerful:** Feedback is automatic, instant, infinitely scalable, and perfectly accurate in verifiable domains. No humans needed at all.

This is how DeepSeek-R1 developed its remarkable reasoning — pure RL on math problems where correctness is objectively verifiable.

---

### 3.5 GRPO — Group Relative Policy Optimization
*(Used in DeepSeek-R1 — cutting edge)*

Generate a **group** of responses for the same prompt, score each, update the model to produce more responses like the high-scoring ones.

```
Prompt: "Solve: 2x + 5 = 13"
Generate 8 responses:
  Response A: x = 4  ✅ (score: 1.0)
  Response B: x = 4  ✅ (score: 1.0)
  Response C: x = 9  ❌ (score: 0.0)
  Response D: x = 4  ✅ (score: 1.0)
  ...
Update model: increase probability of responses like A, B, D
```

No separate reward model needed. Very efficient. Especially powerful for reasoning tasks.

---

### 3.6 Process Reward Models (PRM) ⭐

Outcome rewards (final answer right/wrong) give sparse signal. **PRMs score every reasoning step**, so the model learns *which thoughts* helped — not just whether the final answer was correct.

```
Step 1: "Let x = unknown"                → PRM score 0.9 ✅
Step 2: "Then 2x + 5 = 13"                → 0.9 ✅
Step 3: "So x = 9"                         → 0.1 ❌  (arithmetic wrong)
Step 4: "Therefore answer is 9"            → 0.2 ❌
```

Credit assignment at step-level fixes the "one good trajectory with one silly slip" problem that poisons outcome RL. Combine with GRPO → pick trajectories that dominate step-wise, not just at the end. This is the backbone of OpenAI o1 / DeepSeek-R1 style reasoners.

### 3.7 RLVR — RL with Verifiable Rewards

Generalization of RLEF: anything you can *verify programmatically* becomes a training signal.

- Code → unit tests, type checks, linters
- Math → symbolic solver agreement, numerical check
- Agents → task-completion predicate (file exists, API returns 200)
- Structured output → JSON schema validation, regex
- Translation → round-trip BLEU, back-translation consistency

Verifiers are cheap, fast, uncheatable (vs neural reward models). Build a zoo of verifiers — the more diverse, the more generalizable the reasoning.

### 3.8 KTO / IPO / SimPO — DPO Successors

DPO can over-fit to preference margin and collapse diversity. Consider:

- **KTO** (Kahneman-Tversky Optimization) — needs only binary good/bad labels, no pairs
- **IPO** — stabler objective, avoids reward over-optimization
- **SimPO** — reference-model-free, simpler, often matches DPO

Try KTO if you have thumbs-up/down logs (much easier to collect than chosen/rejected pairs).

---

## Step 5 — Reasoning & Extended Thinking

Opus 4.7's biggest unlock over 2023-era models is **deliberation**: the model generates a hidden chain-of-thought before committing to an answer. You can replicate this.

### 5.1 Supervised CoT Warmup

Collect long-form reasoning traces (from GPT-4o, Claude, DeepSeek-R1, or your own teacher) and SFT on:

```
<prompt>
<think>
[multi-paragraph reasoning, self-check, backtracking]
</think>
<answer>final response</answer>
```

Tag budget tokens (`<think>` ... `</think>`) so the model can learn variable-length thinking.

### 5.2 RL on Reasoning Traces (R1-Zero Recipe)

After SFT warmup, run GRPO + verifiable rewards on math/code. The trick: **reward only final answer correctness, let the model self-discover reasoning strategies**. DeepSeek-R1 showed this emerges: backtracking, self-verification, "aha moments", hypothesis testing — all without supervising the CoT content.

### 5.3 Budget-Controlled Thinking

Train the model to obey `<think budget=N>` directives so users trade latency for quality. Sample training with varied budgets (128, 1K, 8K, 32K tokens) so the policy learns to stop early when confident.

### 5.4 Self-Consistency / Reflection

Train with traces where the model **revisits and corrects itself**:

```
<think>
First attempt: x = 9
Wait, let me recheck: 2(9) + 5 = 23 ≠ 13
So x = 4. Verify: 2(4) + 5 = 13 ✓
</think>
```

Explicit "wait / let me reconsider / I made an error" tokens teach the habit of self-correction.

---

## Step 6 — Tool Use & Agentic Training

Opus 4.7 is fluent at calling tools, orchestrating multi-step workflows, and running long-horizon agent loops. A base chat model isn't — you must train it.

### 6.1 Function-Calling SFT

Train on conversations with structured tool-call turns:

```json
{"role": "user", "content": "What's the weather in Tokyo?"}
{"role": "assistant", "tool_calls": [{"name": "get_weather", "args": {"city": "Tokyo"}}]}
{"role": "tool", "name": "get_weather", "content": "{\"temp_c\": 18}"}
{"role": "assistant", "content": "Tokyo is 18°C."}
```

Datasets: ToolBench, xLAM, Glaive-function-calling, Hermes-Function-Calling. Mix in negative examples where the model **declines to call** when no tool is needed.

### 6.2 Multi-Turn Agent Trajectories

Long-horizon tasks (bug fix across repo, multi-step web navigation, planning). Training sources:

- **SWE-Gym / SWE-bench trajectories** — successful agent runs on real GitHub issues
- **Computer-Use traces** — screenshots + action sequences (xLAM, AgentTrek)
- **Web browsing** — Mind2Web, WebArena rollouts
- **Tau-bench** — customer-service style with tool chains

Reward: task completion predicate (tests pass, PR merged, UI goal reached).

### 6.3 Agent-Style RL

GRPO on full trajectories with sparse outcome reward. Add:

- **Tool-call format verifier** → dense reward on valid JSON/schema adherence
- **Efficiency penalty** → token cost per tool call discourages redundant calls
- **Memory/state coherence** → reward agents that don't forget earlier observations

### 6.4 Skill: "Know When to Stop"

Agentic models loop forever if not trained to terminate. Include trajectories with explicit `<done>` tokens and penalize useless extra steps. This is the hardest-earned competence.

---

## Step 7 — Long Context Extension

Base models are usually trained at 4K–8K context. Opus 4.7 handles 200K+. To extend:

### 7.1 RoPE Scaling

Rotary positional embeddings can be re-scaled at inference:

- **Position Interpolation (PI)** — linearly scale position IDs; crude but works
- **NTK-Aware / Dynamic NTK** — non-linear base frequency scaling, better at length
- **YaRN** ⭐ — best-in-class; scales different frequency bands differently, small amount of fine-tuning
- **LongRoPE** — searches optimal per-dim scaling factors, pushes to 2M tokens

Recipe: quantize → YaRN-extend to 128K → fine-tune 1–2B tokens of long-document data → evaluate on Needle-in-a-Haystack + RULER + LongBench.

### 7.2 Attention Variants for Long Context

- **Sliding Window Attention** (Mistral) — local attention + few global layers
- **Sink tokens / StreamingLLM** — keep first few tokens always attended, evict middle
- **Ring Attention / Blockwise** — distributes attention across devices for 1M+ contexts
- **Mamba / Hybrid SSM** — linear-cost alternatives to attention (Jamba, Zamba)

### 7.3 Long-Context Data

Hardest part. You need documents that genuinely **require** long context — multi-chapter QA, multi-file repos, book summarization. Synthetic: concatenate related docs + insert dependencies that force cross-document reasoning.

---

## Step 8 — Multimodal Capabilities

Opus 4.7 handles images natively. To add vision:

### 8.1 Vision Encoder + Projector (LLaVA-style)

```
Image → CLIP/SigLIP encoder → vision tokens
                              ↓ (MLP projector)
                          text embedding space → LLM
```

Freeze the LLM, train only the projector first (alignment). Then jointly fine-tune on VQA datasets (LLaVA-1.5, ShareGPT-4V, Cauldron).

### 8.2 Native Multimodal (Llama 4, Qwen-VL, Gemma 3)

Tokens and image patches share a unified vocabulary from pre-training. Higher ceiling, much more expensive. Use existing native-MM base if you want visual reasoning.

### 8.3 Audio / Speech

Add a Whisper-style encoder + projector. Or use Qwen2-Audio / Phi-4-multimodal as base. For output speech, layer a neural codec (EnCodec, SNAC) decoder.

### 8.4 Document / OCR

For Opus-level PDF understanding, fine-tune on DocVQA, ChartQA, InfographicVQA, ArXiv-figures. Layout-aware encoders (LayoutLMv3, Donut) help.

---

## Step 9 — Self-Learning Features

This is where the model goes from static to **continuously improving**.

---

### 4.1 RAG + Persistent Memory (Easiest — Start Here)

The model's responses improve because it accesses accumulated knowledge rather than relying on frozen weights.

```
New information arrives
        ↓
Embed → store in vector database
        ↓
User asks question
        ↓
Retrieve relevant stored knowledge
        ↓
Inject into prompt → model answers from YOUR knowledge
```

**Tools:** ChromaDB (local, free), Pinecone (cloud), Weaviate  
**Framework:** LlamaIndex or LangChain for orchestration

Think of this as giving your model an ever-growing, searchable external memory — like Graphify for your AI's own knowledge.

---

### 4.2 Self-Play / Self-Critique Loop (SPIN)

The model critiques and improves its own outputs, and you periodically fine-tune on the improved pairs.

```
Round 1:  Model answers a question
Round 2:  Model critiques its own answer
          "This is incomplete because..."
          "A better answer would include..."
Round 3:  Model generates improved answer
          ↓
Store: (original prompt → improved answer) as training data
          ↓
Fine-tune model on accumulated improved pairs
          ↓
Repeat → model keeps getting better
```

This mirrors how AlphaGo improved by playing against itself. For language models this technique is called **SPIN (Self-Play Fine-Tuning)**.

---

### 4.3 Online / Continual Learning

The model updates its weights from new data in rolling batches over time.

**The core challenge:** Catastrophic forgetting — when the model learns new things, it forgets old ones.

**Solutions:**

| Technique | How It Works |
|---|---|
| **Elastic Weight Consolidation (EWC)** | Penalizes changes to weights that were important for previous tasks |
| **Replay Buffers** | Mix old training data into new training batches |
| **LoRA Adapters Per Domain** | Train separate adapters, base stays frozen — no forgetting possible |
| **Progressive Neural Networks** | Add new network columns for new tasks, freeze old ones |

**Recommended approach:** LoRA adapters per domain — zero forgetting risk, clean separation.

### 4.3.1 Model Soups & Weight Averaging

Averaging weights of multiple fine-tunes (same base) often beats any single run — no inference cost, free quality.

- **Model Soup** — uniform average of hyperparameter sweep checkpoints
- **SLERP / TIES / DARE** — merge adapters with conflict resolution
- **WARP / WARM** — average multiple RL runs to stabilize preference optimization

Use `mergekit` for practical recipes.

### 4.3.2 Memory Beyond RAG

RAG is episodic text recall. Opus-tier agents also need:

- **Procedural memory** — learned skills stored as code/tool chains, retrieved and replayed
- **Semantic memory graph** — entities + relations (Graphify-style), queryable via Cypher
- **Hierarchical summaries** — MemGPT / Letta style: recent turns verbatim, older turns summarized, oldest archived
- **Skill library** (Voyager) — agent writes + stores reusable tool-chains, indexed by task embedding

---

### 4.4 The Complete Self-Learning Loop

```
┌─────────────────────────────────────────────────────────┐
│                   SELF-LEARNING LOOP                     │
│                                                         │
│  User interaction                                       │
│       ↓                                                 │
│  Model generates response                               │
│       ↓                                                 │
│  Self-critique (CAI-style)                              │
│       ↓                                                 │
│  Environment verifies (RLEF) if verifiable              │
│       ↓                                                 │
│  Store high-quality pairs in training buffer            │
│       ↓                                                 │
│  Periodic fine-tuning (QLoRA) on accumulated data       │
│       ↓                                                 │
│  Updated model deployed                                 │
│       ↓                                                 │
│  Repeat ──────────────────────────────────────────────► │
└─────────────────────────────────────────────────────────┘
```

---

## Step 10 — Inference-Time Scaling

Same weights, more compute at inference = smarter outputs. Opus 4.7's "extended thinking" is a productized version of this.

### 10.1 Best-of-N Sampling
Generate N candidates, pick best via reward model or majority vote. N=8–64 common. Diminishing returns past ~32.

### 10.2 Self-Consistency
For reasoning: sample N CoTs, take majority answer. Requires diverse sampling (temp ≥ 0.7).

### 10.3 Tree Search / MCTS
PRM scores intermediate steps; expand promising branches, prune losers. Used in rStar-Math, AlphaProof-style systems.

### 10.4 Verifier-Guided Decoding
Run a verifier (unit test, PRM, constraint checker) at each step; reject-and-resample bad prefixes. Combines beautifully with RLVR-trained models.

### 10.5 Speculative Decoding (for speed, not quality)
Small draft model proposes k tokens, big model verifies in one pass. 2–3× speedup, zero quality loss. Medusa / EAGLE heads go further.

---

## Step 11 — Safety Stack

Opus 4.7's real differentiator is layered safety. A capable model without this is a liability.

### 11.1 Red-Teaming
Adversarial probe for jailbreaks, prompt injections, harmful capabilities. Tools: `garak`, `PyRIT`, `HarmBench`. Automate — don't rely on eyeballing.

### 11.2 Constitutional Classifiers
Anthropic-style defense-in-depth: small classifier models trained on your constitution screen both input and output. Cheap, fast, patchable independently from the main model.

### 11.3 ASL-Style Capability Evals
Before deployment, test dangerous-capability benchmarks:
- **WMDP** — bio/chem/cyber hazardous knowledge
- **Cybench** — offensive cyber
- **SWE-bench-Verified** — autonomous code-edit capability
- **Agentic evals** — METR task suite

Gate releases on thresholds, not vibes.

### 11.4 Jailbreak Resistance Training
Fine-tune on adversarial prompts with refusal targets. Use DPO with `(jailbreak_prompt, harmful_response, safe_refusal)` tuples.

### 11.5 Honesty / Hallucination Reduction
- **Confidence calibration** — train the model to say "I don't know" with verified unknown-answer datasets (TriviaQA + truncation, SelfAware)
- **Citation / grounding** — reward answers with supporting quotes from retrieved context
- **Sycophancy mitigation** — DPO pairs where the chosen answer disagrees with an incorrect user premise

### 11.6 Interpretability
Sparse Autoencoders (SAEs) on residual stream → human-readable features. Lets you detect deception circuits, backdoor features, out-of-distribution activations before they harm users. Libraries: `sae_lens`, `TransformerLens`.

---

## Step 12 — Evaluation Harness

You cannot improve what you cannot measure. Build this **before** training.

### 12.1 Capability Benchmarks

| Domain | Benchmarks |
|---|---|
| General knowledge | MMLU-Pro, GPQA-Diamond |
| Math | MATH, AIME, OlympiadBench |
| Code | HumanEval+, MBPP+, LiveCodeBench, SWE-bench-Verified |
| Reasoning | ARC-AGI, BIG-Bench-Hard, MuSR |
| Long context | RULER, NIAH, InfiniteBench, LongBench |
| Tool use | BFCL, Tau-bench, ToolBench |
| Agentic | SWE-bench, WebArena, OSWorld |
| Multimodal | MMMU, MathVista, DocVQA, ChartQA |
| Instruction following | IFEval, InFoBench, MT-Bench |
| Safety | HarmBench, ToxiGen, TruthfulQA |

### 12.2 LLM-as-Judge
Pairwise comparison with a strong judge (GPT-4o, Claude, or your best internal model). Arena-Hard, AlpacaEval 2.0 LC. Watch for position bias, verbosity bias, self-preference bias — randomize and use multiple judges.

### 12.3 Domain Eval Set
Your golden eval. 100–500 hand-curated prompts from real user traffic, with rubric-based scoring. Run on every checkpoint. This is what tells you whether a training run helped *your users*, not some global benchmark.

### 12.4 Regression Suite
Track N past failure cases as locked tests. Every new model must pass all of them. Catches regressions benchmarks miss.

---

## Step 13 — Model Compression (Run on Low-Spec Hardware)

> *"The model was always smaller than it looked. You're just revealing it."*

---

### 5.1 Quantization — Reduce Precision

Neural network weights stored in high precision often don't need to be.

```
float32  → 4 bytes/weight  (original, 280GB for 70B model)
float16  → 2 bytes/weight  (~no quality loss)
int8     → 1 byte/weight   (slight loss, usually acceptable)
int4     → 0.5 bytes/weight (good with careful handling → 35GB for 70B!)
```

**GPTQ** — Smarter quantization: compensates for rounding error in each weight by adjusting neighbouring weights. Near float16 quality at int4 size.

**QAT (Quantization Aware Training)** — During fine-tuning, simulate quantization in the forward pass. The model learns to be robust to precision loss before you actually quantize it.

**Sweet spot:** `int4` weights + `float16` activations — used by GGUF format (llama.cpp). Run a 70B model on a MacBook.

---

### 5.2 Pruning — Remove What Doesn't Matter

Not all parameters contribute equally. Research shows 80% of intelligence concentrates in ~20% of parameters.

```
Unstructured pruning: remove individual near-zero weights
                      → sparse matrix, hard to speed up on GPU

Structured pruning:   remove entire attention heads or layers
                      → smaller dense matrix, GPU loves this ✅
```

**Attention Head Pruning:** A 70B model with 64 heads per layer — often only 10–20 heads are doing meaningful work. Pruning to 40 heads per layer removes 37% of attention compute with minimal quality loss.

**The Lottery Ticket Hypothesis:**
> Every large neural network contains a small "winning ticket" subnetwork that, when trained in isolation, can match the full network's performance.

The large model was mostly needed to find the winner. Once found, prune everything else.

---

### 5.3 Knowledge Distillation — Clone the Intelligence

Train a small model to imitate a large one. The small model learns not just answers but the **shape of the large model's uncertainty**.

```
Teacher (70B, slow):    "4" → 94%, "four" → 5%, "five" → 0.8%
Student (7B, fast):     learns these soft distributions, not just "4"

Result: 7B model trained via distillation >> 7B model trained from scratch
```

**Progressive distillation:**
```
70B → 30B → 13B → 7B → 3B
```
Each step is a smaller jump — more intelligence preserved at each stage.

---

### 5.4 Low-Rank Factorization — Compress Weight Matrices

Most weight matrices in trained transformers are mathematically low-rank — only a fraction of dimensions are actually being used.

```
Original matrix W: (4096 × 4096) = 16.7M parameters
Factorized:        A × B = (4096×256) × (256×4096) = 2.1M parameters
Savings:           8× reduction with near-identical outputs
```

This is the mathematical foundation of LoRA — and it works because trained models really do live in a much lower-dimensional space than their parameter count suggests.

---

### 5.5 The Cascade Strategy — Combine Everything

```
Stage 1: Structured pruning
         → Remove redundant attention heads and neurons
         → Target: 30–40% parameter reduction, <5% quality loss

Stage 2: Low-rank factorization
         → SVD on weight matrices, keep top-k singular values
         → Additional 20–30% reduction

Stage 3: QAT (Quantization Aware Training)
         → Fine-tune pruned+factorized model with simulated quantization
         → Then quantize to int4 weights / float16 activations

Stage 4: Knowledge Distillation
         → Use the original full model as teacher
         → Fine-tune compressed model to recover lost quality

Expected result: 70B model → effective 7–13B model
                 Fits on 8–16GB VRAM, performs like the 70B on your domain
```

---

## Step 14 — Serving & Deployment Optimizations

A frontier-quality model is useless if serving is slow/expensive. Opus 4.7-tier throughput needs:

### 14.1 Prompt Caching
Anthropic-style cache hits on repeated prefixes (system prompts, few-shot examples, long docs). 90% cost reduction, 5× latency reduction on cached prefixes. Implement with `vLLM` prefix-caching or `SGLang` RadixAttention.

### 14.2 PagedAttention / Continuous Batching
`vLLM` manages KV cache in pages (like virtual memory) → no fragmentation, dynamic batch sizes. Throughput jumps 3–10× vs naive batching.

### 14.3 Speculative Decoding
Medusa heads, EAGLE-2, or a 0.5B draft model. 2–3× latency reduction for interactive workloads.

### 14.4 Tensor / Pipeline / Expert Parallelism
Multi-GPU serving of 70B+ models. TP splits matmuls within a layer, PP splits layers across devices, EP pins MoE experts per device. Use `vLLM`, `TensorRT-LLM`, or `DeepSpeed-Inference`.

### 14.5 KV Cache Compression
- **Quantized KV** (int8/int4) → 2–4× longer contexts per VRAM
- **GQA / MQA** already saves KV; if base is MHA, consider upcycling
- **H2O / StreamingLLM eviction** for very long contexts

### 14.6 Structured Output / Grammar-Constrained Decoding
`outlines`, `xgrammar`, `lm-format-enforcer` guarantee valid JSON/regex/CFG output. Zero retry cost on schema failures.

### 14.7 Batch API
For non-interactive workloads (data labeling, self-play rollouts), offline batch inference at 50% cost.

### 14.8 Routing (Cost/Quality Trade-off)
Route easy queries to a small model, hard ones to the big model. A cheap classifier + fallback = Opus-quality on 10% of traffic and Haiku-speed on the rest.

---

## Full Pipeline Summary

```
┌──────────────────────────────────────────────────────────────────┐
│                 OPUS-TIER COMPLETE PIPELINE                       │
│                                                                  │
│  1. BASE MODEL                                                   │
│     Dense: Llama 3.3 70B / Qwen2.5 72B                          │
│     MoE:   DeepSeek-V3 / Mixtral 8x22B / Llama 4                │
│     MM:    Qwen2.5-VL / Llama 4 (native multimodal)             │
│                                                                  │
│  2. DATA PIPELINE                                                │
│     Dedup (MinHash) → Decontaminate → Quality filter            │
│     Synthetic: Evol-Instruct, Magpie, OSS-Instruct              │
│                                                                  │
│  3. FINE-TUNING                                                  │
│     QLoRA SFT → domain + instruction following                  │
│                                                                  │
│  4. ALIGNMENT                                                    │
│     Constitutional AI → principles                              │
│     DPO/KTO/SimPO → preferences                                 │
│     RLVR + GRPO + PRM → verifiable-reward RL                    │
│                                                                  │
│  5. REASONING                                                    │
│     CoT SFT warmup → R1-zero GRPO → budget-controlled thinking  │
│                                                                  │
│  6. TOOL USE / AGENT                                             │
│     Function-call SFT → SWE-bench trajectories → agent RL       │
│                                                                  │
│  7. LONG CONTEXT                                                 │
│     YaRN → 128K → long-doc fine-tune → NIAH + RULER eval        │
│                                                                  │
│  8. MULTIMODAL (optional)                                        │
│     CLIP/SigLIP + projector → VQA SFT → doc/chart fine-tune     │
│                                                                  │
│  9. SELF-LEARNING                                                │
│     RAG + skill library → self-critique + verifier              │
│     Model soups → periodic QLoRA re-fine-tune                   │
│                                                                  │
│ 10. INFERENCE-TIME SCALING                                       │
│     Best-of-N / self-consistency / MCTS+PRM / verifier-decode   │
│                                                                  │
│ 11. SAFETY                                                       │
│     Red-team → Constitutional classifiers → ASL evals → SAEs    │
│                                                                  │
│ 12. EVAL HARNESS                                                 │
│     Capability bench → LLM-judge → domain golden → regression   │
│                                                                  │
│ 13. COMPRESSION                                                  │
│     Prune → low-rank → QAT int4 → distill                      │
│                                                                  │
│ 14. SERVING                                                      │
│     vLLM/SGLang + prompt cache + spec-decode + paged KV         │
│     Grammar-constrained output, cost/quality routing            │
└──────────────────────────────────────────────────────────────────┘
```

---

## Realistic Roadmap

| Timeline | Goal | Tools |
|---|---|---|
| **Week 1** | API call, eval harness scaffolded | Anthropic/OpenAI SDK, lm-eval-harness |
| **Week 2** | RAG pipeline + domain golden eval set | ChromaDB, LlamaIndex |
| **Week 3** | Data pipeline: dedup, decontam, quality filter | MinHash, fastText, datatrove |
| **Week 4** | QLoRA SFT on cleaned domain data | Axolotl, TRL |
| **Week 5** | DPO/KTO alignment | TRL DPO/KTO trainer |
| **Week 6** | Function-calling SFT — first tool-using model | xLAM, Hermes-FC datasets |
| **Month 2** | CoT warmup + GRPO with verifiable rewards | TRL GRPO, custom verifiers |
| **Month 2** | Long-context extension (YaRN → 128K) | EasyContext, LongRoPE |
| **Month 3** | Self-critique loop + skill library | Custom Python, vector DB |
| **Month 3** | Process reward model + best-of-N at inference | TRL, math-shepherd |
| **Month 4** | Agent RL on SWE-Gym / Tau-bench trajectories | SWE-Gym, OpenHands |
| **Month 4** | Constitutional classifiers + red-team pipeline | garak, PyRIT, HarmBench |
| **Month 5** | Quantization + compression cascade | GPTQ, AWQ, llama.cpp |
| **Month 5** | vLLM serving w/ prompt caching + spec decoding | vLLM, SGLang, Medusa |
| **Month 6+** | Multimodal (if needed); continual re-training cadence | LLaVA recipe, WARP merging |

---

## Key Warnings & Pitfalls

### ⚠️ Data Quality Beats Data Quantity
1,000 perfectly curated examples will outperform 100,000 noisy ones. Garbage in, garbage out. Clean your data obsessively.

### ⚠️ Reward Hacking
In RL-based methods, models learn to **game the reward signal** rather than actually improve. A coding model might generate code that passes tests without being correct by exploiting test weaknesses. Carefully design rewards and use KL constraints.

### ⚠️ Catastrophic Forgetting
Aggressive fine-tuning can erase the base model's general capability. Use LoRA to mitigate — the base stays frozen, only adapters change.

### ⚠️ Quantization Has a Floor
Below int4, you're forcing the model to represent continuous high-dimensional geometry using very coarse grid points. Fine distinctions between similar concepts degrade. Don't go below int4 without extensive testing.

### ⚠️ Distillation Blind Spots
The student learns the teacher's confident outputs well but learns less about situations where the teacher itself was uncertain — exactly the hard edge cases you care most about.

### ⚠️ Evaluation Is Hard
Loss going down doesn't mean the model is actually better. You need human evaluation, benchmark suites, and domain-specific test sets. Build your evaluation pipeline before you build your training pipeline.

### ⚠️ Benchmark Contamination
Public benchmarks leak into training data constantly. Always maintain a **private held-out eval** the model has never seen. Decontaminate corpora with 13-gram overlap checks before training.

### ⚠️ Reasoning RL Collapses Diversity
Pure outcome-reward RL (R1-zero style) often produces brittle, monotone CoTs that only work in-distribution. Mitigate with KL to SFT baseline, entropy bonuses, and mixing in general instruction data.

### ⚠️ Agent Loops Burn Tokens
An untrained agent will spin in a useless-action loop for thousands of tokens. Always bound `max_steps`, add efficiency penalties in RL, and train explicit `<done>` termination.

### ⚠️ Long-Context ≠ Long-Context-Effective
Passing a needle-in-haystack test does **not** mean the model reasons well across 100K tokens. Evaluate with RULER and multi-hop long-doc QA, not just NIAH.

### ⚠️ Safety Is Not a Checkbox
Shipping a jailbreak-resistant RLHF model is the floor. Layered defense (constitutional classifiers, input/output filters, capability evals, SAE monitoring) is what frontier deployment actually looks like.

### ⚠️ MoE Serving Is Expensive
MoE saves training FLOPs but loads **all experts** into VRAM at inference. A "cheap" 400B MoE still needs 400B of VRAM. Plan hardware accordingly or use expert-parallel serving.

---

## Quick Reference — Technique Selector

```
Your situation                      →  Use this technique
────────────────────────────────────────────────────────────
New, want to start                  →  RAG + LlamaIndex
Domain specialization               →  QLoRA SFT
Follow instructions                 →  Instruction Tuning
Preference tuning                   →  DPO / KTO / SimPO
Claude-like values                  →  Constitutional AI (CAI)
Verifiable domains (code/math)      →  RLVR + GRPO + PRM
Deliberative reasoning              →  CoT SFT → R1-zero RL
Tool use / agents                   →  Function-call SFT → agent RL
200K+ context                       →  YaRN / LongRoPE + long-doc FT
Smarter per token at inference      →  Best-of-N / MCTS + PRM
Continuous improvement              →  Self-critique + model soups
Vision input                        →  CLIP/SigLIP + projector (LLaVA)
Run on laptop/phone                 →  int4 quant (GPTQ/AWQ/GGUF)
Smaller but smarter                 →  Knowledge Distillation
Remove unused params                →  Structured Pruning
Cheap serving                       →  vLLM + prompt cache + spec-decode
Defense-in-depth safety             →  Constitutional classifiers + SAE
Catch regressions                   →  Locked golden eval set
```

---

*Built with techniques from: Anthropic (Constitutional AI, RLHF), DeepSeek (GRPO, RLEF), Meta (LoRA, QLoRA), and published ML research.*

*All open-source tools mentioned are freely available on GitHub and Hugging Face.*
