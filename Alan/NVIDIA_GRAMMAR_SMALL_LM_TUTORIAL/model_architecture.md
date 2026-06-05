# Model Architecture Reference

This document describes the language models used in the tutorial examples, with architectural diagrams and relevant literature.

## Table of Contents
1. [Transformer Baseline](#transformer-baseline)
2. [Qwen3-0.6B](#qwen3-06b) (primary example)
3. [SmolLM2-360M-Instruct](#smollm2-360m-instruct)
4. [Model Comparison](#model-comparison-table)

---

## Transformer Baseline

All models in this tutorial are **Decoder-only Transformer Language Models**, following the architecture from [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762).

### Core Components

```
Input: "grep -i pattern /path/file.txt"
       ↓ (tokenization)
    [50, 2049, 1145, 8899, 17903, 28757]  ← Token IDs
       ↓ (embedding)
    [Embedding matrices: d_model = 512]
       ↓
┌─────────────────────────────────────────────────────────┐
│ Decoder Block × N_layers                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 1. Multi-Head Self-Attention                     │  │
│  │    Q = h * W^Q,  K = h * W^K,  V = h * W^V      │  │
│  │    Attention(Q,K,V) = softmax(QK^T/√d_k) V      │  │
│  │    Head_i = Attention_i(Q,K,V)                   │  │
│  │    Output = [Head_1,...,Head_n] * W^O            │  │
│  │    Heads: 8-16 depending on model                │  │
│  └──────────────────────────────────────────────────┘  │
│    ↓ (residual + layer norm)                           │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 2. Feed-Forward Network (FFN)                    │  │
│  │    FFN(x) = ReLU(x * W_1 + b_1) * W_2 + b_2     │  │
│  │    or FFN(x) = GELU(x * W_1 + b_1) * W_2 + b_2 │  │
│  │    Intermediate dim: 4x model_dim (typical)     │  │
│  └──────────────────────────────────────────────────┘  │
│    ↓ (residual + layer norm)                           │
└─────────────────────────────────────────────────────────┘
       ↓ × N_layers
    Final LayerNorm
       ↓
    Linear(d_model → vocab_size)
       ↓
    Logits: [4000, 2100, 500, ... ] ← Scores for each token
       ↓
    Softmax → Probabilities
       ↓
    Sample or Argmax → Next token
```

### Standard Transformer Equations

**Self-Attention** (single head):
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Multi-Head**:
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

**FFN**:
$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2 \quad \text{or} \quad \text{GELU}(xW_1 + b_1)W_2 + b_2$$

**Key Properties**:
- **Causal masking**: Tokens can only attend to previous tokens (enforces autoregressive property)
- **Computational cost**: $O(n^2 d)$ for sequence length $n$, dimension $d$
- **Parameter count**: Dominated by FFN and attention projections

---

## Qwen3-0.6B

### Model Card

| Property | Value |
|----------|-------|
| **Model Family** | Qwen3 (Alibaba) |
| **Parameters** | 600M |
| **Architecture** | Decoder-only Transformer |
| **Context Window** | 32,768 tokens |
| **Vocab Size** | 151,936 tokens |
| **Training Data** | 10T tokens (multilingual, including Bash) |
| **Quantization** | GGUF Q4_K_M recommended |
| **Inference Speed** | ~8-12 tok/s (CPU), ~100+ tok/s (single GPU) |

### Architecture Details

```
Qwen3-0.6B Block Diagram
═══════════════════════════════════════════════════════

Input Tokens → Embedding (600-dim)
                  ↓
            ┌──────────────────┐
            │ Decoder Block 1  │ (1)
            │ - Attn (8 heads) │
            │ - FFN (2400-dim) │
            │ - Norm           │
            └──────────────────┘
                  ↓
            ┌──────────────────┐
            │ Decoder Block 2  │ (2)
            │   ...            │
            └──────────────────┘
                  ↓
            ┌──────────────────┐
            │ Decoder Block 18 │ (18 total)
            └──────────────────┘
                  ↓
            Final LayerNorm
                  ↓
            Linear (600 → 151936)
                  ↓
            Logits → Softmax
                  ↓
            Sample Token (with/without constraints)
```

### Layer Configuration

```python
num_layers: 18
hidden_dim: 600
ffn_dim: 2400  # 4x expansion
num_attention_heads: 8
head_dim: 600 / 8 = 75

total_params ≈ 600M
  - Token embeddings: ~91M (151936 * 600)
  - 18 decoder blocks: ~509M
    - Each block ≈ 28.3M
      - Attention params: ~3M (Q,K,V,O projections)
      - FFN params: ~2M (W1, W2)
      - Norms: Minimal
```

### Why Qwen3-0.6B for Bash?

1. **Sweet spot**: Large enough for semantic understanding, small enough for resource-constrained deployment
2. **Training**: Trained on diverse corpora including code and shell commands
3. **Multilingual**: Better generalization to mixed-language prompts
4. **Results** (from NVIDIA paper): Largest uplift with grammar constraints
   - Native: 16.7% pass rate
   - Constrained: 59.2% pass rate
   - **Uplift: +42.5 percentage points** (strongest among 13 models tested)

### Inference Characteristics

**Token logit distribution** (typical for bash generation):

```
Rank | Token         | Logit | Prob (native) | Prob (constrained)
─────┼───────────────┼───────┼───────────────┼──────────────────
  1  | "2"           | 18.1  | 0.35          | 0.00 (masked)
  2  | "base"        | 16.8  | 0.28          | 0.62 (permissible)
  3  | "64"          | 15.2  | 0.22          | 0.38 (permissible)
  4  | "x"           | 12.1  | 0.12          | 0.00 (masked)
  5  | "\n"          | 11.9  | 0.03          | 0.00 (masked)
```

**Observation**: Grammar masking redistributes probability from forbidden tokens to legal alternatives, improving sequence likelihood.

**Citation**: [Qwen3 Technical Report; Alibaba, 2026](https://huggingface.co/Qwen/Qwen3-0.6B)

---

## SmolLM2-360M-Instruct

### Model Card

| Property | Value |
|----------|-------|
| **Model Family** | SmolLM2 (Hugging Face / Anthropic) |
| **Parameters** | 360M |
| **Architecture** | Decoder-only Transformer |
| **Context Window** | 8,192 tokens |
| **Vocab Size** | 49,152 tokens |
| **Training** | 707B tokens (code-heavy) |
| **Instruction-tuned** | Yes (chat format) |
| **Quantization** | GGUF Q4_K_M |
| **Inference Speed** | ~5-8 tok/s (CPU), ~80+ tok/s (GPU) |

### Architecture Details

```
SmolLM2-360M Block Diagram
═══════════════════════════════════════════════════════

Input Tokens → Embedding (576-dim)
                  ↓
            ┌──────────────────┐
            │ Decoder Block 1  │ (1)
            │ - Attn (9 heads) │
            │ - FFN (1536-dim) │
            │ - Norm           │
            └──────────────────┘
                  ↓
            ┌──────────────────┐
            │ Decoder Block 12 │ (12 total)
            └──────────────────┘
                  ↓
            Final LayerNorm
                  ↓
            Linear (576 → 49152)
                  ↓
            Logits → Sample
```

### Layer Configuration

```python
num_layers: 12
hidden_dim: 576
ffn_dim: 1536  # 2.67x (non-standard expansion)
num_attention_heads: 9
head_dim: 576 / 9 = 64

total_params ≈ 360M
  - Token embeddings: ~28.3M (49152 * 576)
  - 12 decoder blocks: ~331.7M
```

### Special Characteristics

1. **Non-standard FFN expansion**: 2.67x instead of typical 4x (efficiency optimization)
2. **Code-optimized training**: Heavy weighting on code corpora → better bash understanding
3. **Instruction tuning**: Responds better to natural language task descriptions
4. **Performance trade-off**: Slightly weaker than Qwen3 but more instruction-following

### NVIDIA Paper Results for SmolLM2-360M-Instruct

```
Native pass rate:        29.4% (88/299 tasks)
Constrained pass rate:   57.2% (171/299 tasks)
Uplift:                  +27.8 percentage points
```

**Task Breakdown**:
- **Tier 1** (I/O primitives): 85% → 95% (+10 pts)
- **Tier 2** (Filter/transform): 32% → 61% (+29 pts)
- **Tier 3** (Recon/action): 20% → 38% (+18 pts)
- **Tier 4** (Shell constructs): 50% → 45% (-5 pts, slight regression)

**Citation**: [SmolLM2 Technical Report; Hugging Face, 2025](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)

---

## Model Comparison Table

| Model | Params | Layers | Hidden | Heads | Vocab | Native Pass | Constrained Pass | Uplift |
|-------|--------|--------|--------|-------|-------|-------------|------------------|--------|
| **Qwen3-0.6B** | 600M | 18 | 600 | 8 | 151k | **16.7%** | **59.2%** | **+42.5** |
| **SmolLM2-360M** | 360M | 12 | 576 | 9 | 49k | 29.4% | 57.2% | +27.8 |
| **Qwen2.5-0.5B** | 500M | 12 | 640 | 10 | 151k | 44.5% | 68.6% | +24.1 |
| **Qwen3.5-0.8B** | 800M | 20 | 800 | 10 | 151k | 52.8% | 66.9% | +14.0 |
| **gemma-3n-E2B** | 2.0B | 18 | 1024 | 8 | 256k | 63.5% | 75.9% | +12.4 |
| **SmolLM3-3B** | 3.0B | 30 | 576 | 9 | 49k | 69.2% | 78.9% | +9.7 |
| **Nemotron-3-Nano** | 4.0B | 24 | 1536 | 12 | 256k | 80.9% | 88.3% | +7.4 |

### Key Observations

1. **Inverse correlation**: Smaller, weaker models benefit most from grammar constraints
   - 0.6B models: ~40pt uplift
   - 4B+ models: ~5-7pt uplift

2. **Task complexity dependency**: Constraints help most on lower-complexity tiers
   - Tier 1 (I/O): +10 pts average
   - Tier 4 (Composition): -0.4 pts (slight regression)

3. **Instruction-tuning matters**: SmolLM2-Instruct outperforms base Qwen2.5 despite smaller size

---

## Attention Mechanism Deep Dive

### Multi-Head Self-Attention Visualization

For bash command `grep -i pattern file.txt`, position $t=3$ (token "pattern"):

```
Query (position 3): "pattern token" → Q3 vector
                         ↓
              [Attn Head 1]    [Attn Head 2]    ... [Attn Head 8]
                    ↓                 ↓                    ↓
    Scores: [1.2, 5.3, 8.1, 3.4, ...]  (dot product with K vectors)
                    ↓ (softmax over previous positions)
    Weights: [0.01, 0.15, 0.62, 0.22, ...]
                    ↓ (weighted sum of V vectors)
    Head_1 output: weighted combination of all previous context
                         ↓
              [Concat + Linear projection]
                         ↓
    Final Attention output (fed to FFN)
```

**Key insight for grammar constraints**: Attention allows the model to look back at previous tokens (e.g., "grep" command token) to inform what flags are legal next. Constraints should respect these learned attention patterns.

---

## Training Corpus Overview

### Qwen3 Training

```
10T tokens total
├─ Code & Bash: ~8-10% (800B tokens)
├─ Natural language (multilingual): ~70%
├─ Mathematical reasoning: ~5%
└─ Other (knowledge, dialogue): ~15%
```

### SmolLM2 Training

```
707B tokens total (code-optimized)
├─ Code: ~20-25% (140-175B tokens)
│  ├─ Python: 40% of code
│  ├─ JavaScript: 20%
│  ├─ Bash/Shell: 10%
│  └─ Other (Java, C++, etc.): 30%
├─ Natural language: ~50%
└─ Other: ~25-30%
```

**Implication**: SmolLM2 has more bash-specific training data per parameter, explaining code-generation advantage despite smaller size.

---

## Quantization Impact

The tutorial uses **GGUF Q4_K_M quantization** (4-bit, medium key-value).

```
Original model size:        600M × 4 bytes = 2.4 GB
Quantized size:             600M × 0.5 bytes = 300 MB
Compression ratio:          8x
Quality loss:               ~2-3% accuracy (acceptable for task-specific work)
Inference speedup:          ~1.5-2x (GPU), minimal impact (CPU)
```

**Trade-off**: Slight accuracy loss for dramatic size reduction and faster inference.

---

## Forward Pass Computational Flow

For a single token generation step with Qwen3-0.6B:

```
Input: h_prev (hidden state from previous token)
       ├─ Shape: (1, 600)  [batch_size=1, hidden_dim=600]
       │
       ├─ Layer 1 (Qwen3-0.6B Decoder Block):
       │  ├─ Self-Attention:
       │  │  ├─ Q = h * W_Q                           (1×600) × (600×600) = (1×600)
       │  │  ├─ K = h_cache * W_K                     (t×600) × (600×600) = (t×600)
       │  │  ├─ V = h_cache * W_V                     (t×600) × (600×600) = (t×600)
       │  │  ├─ Scores = Q @ K^T / √75               (1×600) × (600×t) = (1×t)
       │  │  ├─ Weights = softmax(scores)            (1×t)
       │  │  └─ Output = weights @ V                  (1×t) × (t×600) = (1×600)
       │  │
       │  ├─ FFN:
       │  │  ├─ h_mid = GELU(h * W_up)               (1×600) × (600×2400) = (1×2400)
       │  │  └─ h_out = h_mid * W_down               (1×2400) × (2400×600) = (1×600)
       │  │
       │  └─ Residual & LayerNorm:
       │     └─ h = LayerNorm(h + attn + ffn)        (1×600)
       │
       ├─ ... × 17 more decoder blocks
       │
       └─ Output Layer:
          ├─ Final LayerNorm: (1×600)
          ├─ Linear: (1×600) × (600×151936) = (1×151936)
          └─ Logits: (1×151936)

Output: logits shape (1, 151936)
        ├─ logits[i] = score for token_id i
        ├─ softmax(logits) = probabilities
        └─ sample or argmax → next token_id
```

**Computational cost per token**: ~1.2 billion multiply-accumulate operations (MACs)

---

## Memory Requirements During Inference

### KV Cache (Key-Value Cache for Attention)

During inference, must store previous K and V vectors for all attention heads:

```
For sequence length n, num_layers L, hidden_dim d:

KV_cache_size = 2 × L × n × d bytes
              = 2 × 18 × n × 600 bytes  (Qwen3-0.6B)
              = 21,600n bytes

Example:
- n=100 tokens: 2.1 MB
- n=1000 tokens: 21 MB
- n=32768 tokens (full context): 688 MB
```

### Total Memory Footprint

```
Qwen3-0.6B Inference:
├─ Model weights (quantized): 300 MB
├─ KV cache (max context): 688 MB
├─ Computation buffers: ~100 MB
└─ Total: ~1.1 GB (fits on modest GPU)
```

**For CPU inference**: Requires sufficient RAM; 4GB minimum recommended.

---

## References

1. **Vaswani et al. (2017)** - Attention is All You Need
   - ArXiv: https://arxiv.org/abs/1706.03762
   - Defines core Transformer architecture

2. **Qwen3 Technical Report (2026)**
   - HuggingFace: https://huggingface.co/Qwen/Qwen3-0.6B
   - Model card with architecture details

3. **SmolLM2 Technical Report (2025)**
   - HuggingFace: https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct
   - Code-optimized training regime

4. **NVIDIA Lucas et al. (2026)** - Grammar-Constrained Decoding for Bash
   - URL: https://developer.nvidia.com/blog/improving-bash-generation-in-small-language-models-with-grammar-constrained-decoding/
   - Empirical results on all 13 models

5. **Karpathy (2015)** - The Unreasonable Effectiveness of RNNs
   - Blog post: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
   - Foundational intuition for sequence generation

---

**Last Updated**: June 2026
