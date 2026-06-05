# Glossary: Grammar-Constrained Decoding for Bash Generation

## Core Terminology

### Autoregressive Decoding

**Definition**: Sequential generation of tokens in language models where the probability distribution at step $t$ depends on all previously generated tokens $x_{<t}$.

**Mathematical Notation**:
$$P(x_1, x_2, \ldots, x_n | \text{context}) = \prod_{t=1}^{n} P(x_t | x_{<t}, \text{context})$$

**Formal Model**: Given:
- Vocabulary $V$ with $|V| = d$ (token IDs)
- Hidden state $h_t \in \mathbb{R}^d$ (transformer output)
- Logits $\ell_t = W^\top h_t + b$ where $W \in \mathbb{R}^{d \times |V|}$
- Temperature $\tau > 0$

Token selection:
$$\hat{x}_t = \arg\max_i \left[ \ell_t^{(i)} + \text{Gumbel}(0, 1) \right] \quad \text{(greedy)}$$
$$\hat{x}_t \sim \text{Categorical}\left(\text{softmax}(\ell_t / \tau)\right) \quad \text{(sampling)}$$

**Citation**: [Transformers et al., Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)

---

### Constrained Decoding

**Definition**: Modification of the sampling distribution $P(x_t | x_{<t}, \text{context})$ by applying a constraint function $C_t: \mathbb{R}^{|V|} \to \mathbb{R}^{|V|}$ that masks illegal continuations.

**Constraint Mechanism** (Token Masking):
$$P_{\text{constrained}}(x_t | x_{<t}, \text{context}) = \frac{\exp(\ell_t^{(i)} / \tau) \cdot \mathbb{1}[i \in S_t]}{\sum_{j \in S_t} \exp(\ell_t^{(j)} / \tau)}$$

Where:
- $S_t \subseteq V$ is the set of **permissible token IDs** at step $t$
- $\mathbb{1}[\cdot]$ is the indicator function
- Tokens $i \notin S_t$ receive $P_{\text{constrained}}(x_t = i) = 0$ (forbidden)

**Effect**: Restricts the support of the categorical distribution to valid tokens only.

**Citation**: [PICARD: Scholak et al., 2021](https://arxiv.org/abs/2109.05093); [Guidance: Willard et al., 2023](https://arxiv.org/abs/2307.04964)

---

### Grammar (Formal Language)

**Definition**: A tuple $G = (N, T, P, S)$ where:
- $N$ = non-terminals (syntactic categories)
- $T$ = terminals (tokens/tokens)
- $P$ = production rules $A \to \alpha$ where $A \in N$, $\alpha \in (N \cup T)^*$
- $S \in N$ = start symbol

**Example**: Minimal bash grammar
```
N = {Cmd, OptList, Opt, Word}
T = {"grep", "-i", "pattern", "|", "cat"}
P = {
    Cmd     → "grep" OptList Word
    OptList → ε | Opt OptList
    Opt     → "-i"
    Word    → "pattern"
}
S = Cmd
```

**Language Generated**: $L(G) = \{\text{all strings derivable from } S \text{ using } P\}$

**Citation**: [Chomsky Hierarchy and Formal Grammars; Chomsky, 1956](https://www.jstor.org/stable/2308637)

---

### Lark Parser Grammar

**Definition**: Python parsing library using EBNF-style grammar specifications with built-in Earley/LALR parsers.

**Key Notation**:
- `?rule` = inline rule (flatten parse tree)
- `|` = alternation (OR)
- `{n,m}` = repetition bounds (min $n$, max $m$)
- `!` = rule priority for ambiguity resolution
- `@v.2` = transformer version

**Lark Bash Snippet**:
```lark
start: "grep" (WS opt){0,8} WS WORD (WS PATH){0,5}
opt: "-" /[EFGHILPRTUVZabchilnoqrsvwxz]+/
    | "-e" WS WORD
    | "-f" WS PATH
    | "--ignore-case"
WORD: /[^\s|><&;()]{1,200}/
PATH: /[^\s|><&;()]{1,200}/
WS: /\s+/
```

**Citation**: [Lark Documentation & Parser Library](https://github.com/lark-parser/lark)

---

### Grammar IR (Intermediate Representation)

**Definition**: Abstract data structure capturing command syntax without full grammar verbosity. Enables programmatic grammar generation from command evidence.

**Data Structure**:
```python
@dataclass
class GrammarIR:
    command_name: str
    short_flags: Set[str]           # e.g., {"i", "v", "n"}
    long_flags: Dict[str, bool]     # bool = valued (True) or boolean (False)
    positional_args: List[ArgType]  # ArgType ∈ {PATH, WORD, INT, REGEX}
    flag_frequency: Dict[str, float] # P(flag | command)
    composition_rules: Set[str]      # e.g., {"pipe", "redirect"}
```

**Derivation from Corpus**: Given command sample $\mathcal{C} = \{c_1, \ldots, c_n\}$:
$$\text{flags}(G) = \{f \in c_i : f \text{ appears in } \geq \theta \cdot n \text{ commands}\}$$

Where $\theta$ is a frequency threshold (typically 0.05 = 5% minimum support).

---

### Finite-State Constraint Machine (FSCM)

**Definition**: Deterministic finite automaton applied during decoding to enforce grammar compliance at token level.

**Formal Definition**: Tuple $(Q, V, \delta, q_0, F)$:
- $Q$ = states (grammar parse positions)
- $V$ = vocabulary (token IDs)
- $\delta: Q \times V \to Q$ = transition function
- $q_0$ = initial state
- $F \subseteq Q$ = accepting states

**Operation at decoding step $t$**:
1. Current grammar state: $q_t$ (derived from tokens $x_{<t}$)
2. Permissible tokens: $S_t = \{v \in V : \delta(q_t, v) \text{ is defined}\}$
3. After token $x_t = v$ selected: $q_{t+1} = \delta(q_t, v)$

**Practical Implementation**: Lark parser maintains internal stack-based state during parsing; at each step, we query which tokens would advance the parse (no syntax errors).

**Complexity**: $O(|Q| \cdot |V| \cdot |T|)$ per decoding step where $|T|$ is grammar rule count.

**Citation**: Derived from automata theory; see [Hopcroft & Ullman, Introduction to Automata Theory](https://www.pearsonhighered.com/program/Hopcroft-Introduction-to-Automata-Theory-Languages-and-Computation-3rd-Edition/PGM4097.html)

---

### llguidance (Large Language Model Guidance)

**Definition**: Framework for constraining LLM sampling through both grammar-based and programmatic constraints during token-by-token generation.

**Architecture**:
- **Constraint compiler**: Translates grammar/program → finite automaton
- **Token masking engine**: Computes $S_t$ (valid token set) each step
- **Backend bindings**: C++ inference engine (llama.cpp) with Python API

**Integration Pattern**:
```python
# Pseudocode
guidance_state = guidance.init_from_grammar(lark_grammar)
for t in range(max_tokens):
    h_t = model.forward(x_{<t})  # Hidden state
    logits_t = model.to_logits(h_t)
    valid_tokens = guidance_state.get_valid_tokens(tokenizer)
    logits_t[invalid_tokens] = -∞  # Mask forbidden
    x_t ~ Categorical(softmax(logits_t))
    guidance_state.consume_token(x_t)  # Update automaton
    output.append(x_t)
```

**Key Advantage**: Guaranteed syntactic validity (assuming grammar correctness).

**Citation**: [Guidance: Controlling LLM Behavior; Willard et al., 2023](https://arxiv.org/abs/2307.04964)

---

## Bash-Specific Terminology

### Shell Command Syntax Hierarchy

```
┌─────────────────────────────────────────┐
│ Compound Command (e.g., pipeline, loop) │
├─────────────────────────────────────────┤
│ Simple Command                          │
│ ├─ Command name (binary)                │
│ ├─ Options (flags)                      │
│ ├─ Arguments (positional)               │
│ └─ Redirections (>, >>, <, 2>&1)        │
└─────────────────────────────────────────┘
```

**Notation**: POSIX shell grammar uses BNF extended with Kleene star ($*$), plus ($+$), optional ($?$).

**Citation**: [POSIX Shell Standard, IEEE 1003.1-2017](https://pubs.opengroup.org/onlinepubs/9699919799/)

---

### Flag (Option) Classification

1. **Boolean flag**: Presence/absence controls behavior (e.g., `-i` = case-insensitive)
   - Notation: $\text{BoolFlag} \in \{f : f \in P(x) \text{ and } |V(f)| = 0\}$
   
2. **Valued flag**: Requires argument (e.g., `-A 3` = lines after)
   - Notation: $\text{ValuedFlag} = (f, v)$ where $f$ is flag, $v$ is value
   - Forms: `-A 3` (space-separated) or `-A=3` (equals-joined)

3. **Combined short flags**: Multiple boolean flags combined (e.g., `-iRv` = `-i -R -v`)
   - Constraint: Only valid if all individual flags are boolean

**Frequency Analysis**: In corpora, $P(\text{BoolFlag}) \gg P(\text{ValuedFlag})$ for most utilities.

---

### Pass Rate (Success Metric)

**Definition**: Proportion of tasks where generated command achieves objective without error.

**Formal Definition**:
$$\text{PassRate} = \frac{|\{t \in T : \text{exec}(c_t) = \text{expected}(t)\}|}{|T|}$$

Where:
- $T$ = task set
- $c_t$ = generated command
- $\text{exec}(\cdot)$ = command execution result
- $\text{expected}(\cdot)$ = ground-truth output

**Decomposition** (from NVIDIA paper):
- **Native**: Model without constraints
- **Constrained**: With grammar masking
- **Constrained + Retry**: Grammar → fallback to native on syntax error

---

### Uplift (Performance Improvement)

**Definition**: Absolute percentage-point gain in pass rate.

$$\text{Uplift} = \text{PassRate}_{\text{constrained}} - \text{PassRate}_{\text{native}}$$

**Example**: Model improves from 16.7% to 59.2% → Uplift = **+42.5 pts**

**Key Insight** (from paper): Uplift correlates inversely with native performance—weaker models benefit more from constraints.

---

## Evaluation Metrics

### Syntax Validity (tree-sitter-bash)

**Definition**: Whether generated string parses correctly under Bash grammar.

**Implementation**: tree-sitter-bash parser returns:
- $\text{valid} = \neg\text{has\_error\_nodes}(\text{parse\_tree})$

**False Positives**: tree-sitter accepts some malformed syntax (e.g., unclosed quotes).

**Complement**: Semantic validation (does command actually work?) requires execution.

---

### Recall of Valid Completions

**Definition**: For a given prompt prefix, what fraction of valid continuations does the grammar permit?

$$\text{Recall} = \frac{|\text{ValidTokens}_{\text{grammar}} \cap \text{ValidTokens}_{\text{optimal}}|}{|\text{ValidTokens}_{\text{optimal}}|}$$

**Tradeoff**: High recall (permissive grammar) may hurt reliability; low recall (restrictive grammar) may reject valid commands.

---

### Regression Detection

**Definition**: Cases where constrained decoding produces wrong answer that native decoding got right.

**Quantification**:
$$\text{Regression} = \{(t, m) : \text{PassRate}_{\text{native}}(t, m) = 1 \land \text{PassRate}_{\text{constrained}}(t, m) = 0\}$$

**Root Causes**:
1. Grammar incompleteness (valid syntax not expressible)
2. Grammar over-restriction (correct command violates grammar)
3. Token distribution mismatch (model wants to generate unusual flag order)

---

## Advanced Topics

### Policy-Encoded Grammars

**Concept**: Extend formal grammar with security/operational constraints.

**Example**: HTTPS-only for curl
```lark
ssl_protocol: "--cacert" WS PATH
            | "--capath" WS PATH
            | "https://" HOST  // Only HTTPS allowed
```

By construction, grammar forbids `http://` or insecure flags.

---

### Entropy Reduction Through Masking

**Metric**: Shannon entropy of constrained vs. native distribution.

$$H(P) = -\sum_{v \in V} P(v) \log P(v)$$

$$\Delta H = H(P_{\text{native}}) - H(P_{\text{constrained}})$$

**Interpretation**: Positive $\Delta H$ indicates grammar narrows distribution, reducing uncertainty.

**Typical Range**: 2–6 bits of entropy reduction per step on small-model Bash generation.

---

### Grammar Incompleteness

**Definition**: Existence of semantically valid bash commands not accepted by generated grammar.

$$\text{Incompleteness} = \{c \in \text{ValidBash} : c \notin L(G_{\text{generated}})\}$$

**Mitigation Strategies**:
1. Increase frequency threshold $\theta$ (include rare flags)
2. Add manual rules for documented but uncommon patterns
3. Use learned grammars (data-driven rather than corpus-driven)

**Trade-off**: Broader grammars → more false positives (invalid commands accepted).

---

## Mathematical Notation Conventions

| Symbol | Meaning |
|--------|---------|
| $V$ | Vocabulary (set of token IDs) |
| $T$ | Set of terminals in grammar |
| $N$ | Set of non-terminals in grammar |
| $P(·)$ | Probability distribution or production rules |
| $\mathcal{C}$ | Corpus (set of commands) |
| $S_t$ | Set of permissible tokens at step $t$ |
| $h_t$ | Hidden state at position $t$ |
| $\ell_t$ | Logits at position $t$ |
| $\tau$ | Temperature (sampling hyperparameter) |
| $\theta$ | Frequency threshold for grammar IR generation |
| $\delta$ | Transition function (FSM) |
| $q$ | State (FSM) |
| $\mathbb{1}[·]$ | Indicator function |

---

## Key Paper References

1. **Grammar-Constrained Decoding for Bash**
   - Lucas et al. (2026) - NVIDIA AI Red Team
   - URL: https://developer.nvidia.com/blog/improving-bash-generation-in-small-language-models-with-grammar-constrained-decoding/

2. **PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding**
   - Scholak, Schucher, Bahdanau (2021)
   - ArXiv: https://arxiv.org/abs/2109.05093
   - Focus: SQL generation, token masking theory

3. **Guidance: Controlling Large Language Models**
   - Willard et al. (2023)
   - ArXiv: https://arxiv.org/abs/2307.04964
   - Focus: llguidance framework, constraint compilation

4. **Attention is All You Need**
   - Vaswani et al. (2017)
   - ArXiv: https://arxiv.org/abs/1706.03762
   - Foundation: Transformer architecture, autoregressive generation

5. **The Unreasonable Effectiveness of Recurrent Neural Networks**
   - Karpathy (2015)
   - Blog post on RNNs and sequence modeling fundamentals

6. **POSIX Shell Standard**
   - IEEE 1003.1-2017
   - Official: https://pubs.opengroup.org/onlinepubs/9699919799/

7. **Tree-sitter: A Parser Generator Tool**
   - Scott Wolchok et al.
   - GitHub: https://github.com/tree-sitter/tree-sitter
   - Focus: Incremental parsing, syntax trees

---

## Related Glossary Files

- `model_architecture.md` – Detailed model descriptions (Qwen3, SmolLM, etc.)
- `references.md` – Full bibliography with abstracts
- `bash_syntax_reference.md` – Shell command syntax (auto-generated from datasets)

---

**Last Updated**: June 2026  
**For questions**: Refer to notebook comments or open GitHub issue.
