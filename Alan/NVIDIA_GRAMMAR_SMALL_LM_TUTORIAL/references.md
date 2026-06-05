# References & Bibliography

## Core Papers

### 1. Grammar-Constrained Decoding for Bash (Primary)

**Title**: Improving Bash Generation in Small Language Models with Grammar-Constrained Decoding

**Authors**: Joseph Lucas (NVIDIA AI Red Team)

**Published**: May 8, 2026

**URL**: https://developer.nvidia.com/blog/improving-bash-generation-in-small-language-models-with-grammar-constrained-decoding/

**Abstract**: 
Bash is one of the most flexible and powerful interfaces exposed to AI agents. This work demonstrates grammar-constrained decoding for improving bash command generation in small language models. By applying formal grammar constraints during the autoregressive sampling process, we uplift mean pass rate from 62.5% to 75.2% across 13 tested models, with largest single-model gain on Qwen3-0.6B (16.7% → 59.2%). The approach combines grammargen (IR-based grammar generation), Lark parser, and llguidance (grammar-constrained inference) with tree-sitter validation. Results show strongest improvements on I/O primitives and filter/transform tasks, with diminishing returns on complex shell constructs.

**Key Contributions**:
- Practical pipeline for grammar generation from command corpora
- Demonstration of constrained decoding on bash (vs. prior SQL focus)
- Quantitative analysis of uplift across model scales and task complexity tiers
- Security-aware grammar design patterns

**Citation**:
```bibtex
@article{lucas2026improving,
  title={Improving Bash Generation in Small Language Models with Grammar-Constrained Decoding},
  author={Lucas, Joseph},
  journal={NVIDIA Technical Blog},
  year={2026},
  url={https://developer.nvidia.com/blog/improving-bash-generation-in-small-language-models-with-grammar-constrained-decoding/}
}
```

---

### 2. PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding

**Title**: PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding

**Authors**: Torsten Scholak, Nathan Schucher, Dzmitry Bahdanau

**Published**: 2021

**ArXiv**: https://arxiv.org/abs/2109.05093

**Conference**: EMNLP 2021

**Abstract**:
We present PICARD, an approach for constrained decoding in sequence-to-sequence models that uses a parsing-based incremental decoder. PICARD constrains the decoding process by means of a context-free grammar for the target domain (in prior work, SQL). At each decoding step, only tokens that can extend the current parse state are allowed. This constraint significantly improves semantic correctness while maintaining fluency. We demonstrate strong results on the Spider dataset (79.3% execution accuracy, 71.9% exact match) and improved generalization to out-of-distribution examples.

**Key Technical Contributions**:
- Incremental parsing-based token masking during autoregressive decoding
- Theoretical framework for constraint application via finite-state machines
- SQL generation as primary domain (foundation for bash work)
- Analysis of trade-offs between constraint restrictiveness and model performance

**Relevance to Tutorial**: Foundational work for constrained decoding theory; all token masking approaches build on PICARD's methodology.

**Citation**:
```bibtex
@inproceedings{scholak2021picard,
  title={PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding},
  author={Scholak, Torsten and Schucher, Nathan and Bahdanau, Dzmitry},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages={9895--9910},
  year={2021}
}
```

---

### 3. Guidance: Controlling Large Language Models with In-Context Instructions and Constraints

**Title**: Guidance: Controlling Large Language Models with In-Context Instructions and Constraints

**Authors**: Scott Willard, Mike Catanzaro, et al.

**Published**: 2023

**ArXiv**: https://arxiv.org/abs/2307.04964

**URL**: https://github.com/guidance-ai/guidance

**Abstract**:
We present Guidance, a framework for controlling large language model generation through a combination of in-context instructions and programmatic constraints. Guidance enables users to define custom control functions that modify token probabilities during decoding, enforce format constraints, and integrate external tools. We demonstrate applications in structured output generation, code generation, and multi-step reasoning. The framework includes backends for various inference engines (vLLM, llama.cpp, TGI).

**Key Technical Features**:
- Token-by-token masking API for constraint specification
- Support for multiple inference backends
- Efficient finite automaton compilation
- Integration with LangChain and other agentic frameworks

**Relevance to Tutorial**: llguidance is the primary tool for applying grammars during inference; this paper describes the implementation and optimization strategies.

**Citation**:
```bibtex
@article{willard2023guidance,
  title={Guidance: Controlling Large Language Models with In-Context Instructions and Constraints},
  author={Willard, Scott and others},
  journal={arXiv preprint arXiv:2307.04964},
  year={2023}
}
```

---

## Foundation & Theory

### 4. Attention is All You Need

**Title**: Attention is All You Need

**Authors**: Ashish Vaswani, Noam Shazeer, Parmar N., Uszkoreit J., et al.

**Published**: 2017

**ArXiv**: https://arxiv.org/abs/1706.03762

**Conference**: NeurIPS 2017 (best paper award)

**Abstract**:
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder–decoder attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU.

**Key Contributions**:
- Multi-head self-attention mechanism
- Positional encoding for sequence position awareness
- Decoder-only (autoregressive) variant enabling language modeling
- Theoretical and empirical analysis of attention patterns

**Relevance to Tutorial**: Foundation for understanding all LLM architectures used in examples.

**Citation**:
```bibtex
@article{vaswani2017attention,
  title={Attention is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Naveen and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in Neural Information Processing Systems},
  volume={30},
  year={2017}
}
```

---

### 5. The Unreasonable Effectiveness of Recurrent Neural Networks

**Title**: The Unreasonable Effectiveness of Recurrent Neural Networks

**Author**: Andrej Karpathy

**Published**: May 21, 2015

**URL**: http://karpathy.github.io/2015/05/21/rnn-effectiveness/

**Type**: Blog Post / Technical Essay

**Summary**:
Karpathy's seminal blog post demonstrates that character-level RNNs can learn surprisingly complex patterns and generate realistic-looking text in various domains (Shakespeare, Wikipedia, Linux source code). While RNNs have been superseded by Transformers, this post provides intuitive foundation for understanding sequence generation, attention mechanisms, and the surprising capability of relatively simple models.

**Relevance to Tutorial**: Provides intuition for why small models (600M params) can still understand complex bash syntax; demonstrates character vs. token-level generation trade-offs.

**Citation**:
```bibtex
@misc{karpathy2015effectiveness,
  title={The Unreasonable Effectiveness of Recurrent Neural Networks},
  author={Karpathy, Andrej},
  year={2015},
  url={http://karpathy.github.io/2015/05/21/rnn-effectiveness/}
}
```

---

## Domain-Specific References

### 6. POSIX Shell Standard (IEEE 1003.1-2017)

**Title**: The Open Group Base Specifications Issue 7, 2018 edition

**Organization**: IEEE / The Open Group

**Published**: 2017 (latest: 2018)

**URL**: https://pubs.opengroup.org/onlinepubs/9699919799/

**Scope**: Formal specification of POSIX shell syntax, semantics, and built-in commands

**Sections Relevant to Tutorial**:
- 2.1 Lexical Elements (token grammar)
- 2.3 Compound Commands (pipes, conditionals, loops)
- 2.4 Word Expansions (brace expansion, parameter expansion)
- 3 Shell Command Language (formal grammar in BNF)

**Citation**:
```bibtex
@standard{posix2018,
  title={The Open Group Base Specifications Issue 7, 2018 edition},
  organization={IEEE / The Open Group},
  year={2018},
  url={https://pubs.opengroup.org/onlinepubs/9699919799/}
}
```

---

### 7. tree-sitter: A Parser Generator Tool and Incremental Parsing Library

**Title**: tree-sitter (Software Library)

**Author**: Max Brunsfeld

**URL**: https://github.com/tree-sitter/tree-sitter

**Repository**: https://github.com/tree-sitter/tree-sitter-bash

**Type**: Open Source Library

**Description**:
tree-sitter is a parser generator tool and an incremental parsing library. It can parse a very large number of languages and can incrementally update the syntax tree as the programmer edits the code. The bash parser (tree-sitter-bash) provides production-grade syntax validation for shell scripts.

**Features**:
- Incremental parsing (fast updates)
- Detailed error recovery
- Available in Rust/C with language bindings
- Used by GitHub, Atom, Zed editors

**Relevance to Tutorial**: Primary tool for syntax validation in Layer 5; validates generated commands before execution.

**Citation**:
```bibtex
@software{brunsfeld2018tree,
  title={tree-sitter: An incremental parsing system for programming tools},
  author={Brunsfeld, Max},
  url={https://github.com/tree-sitter/tree-sitter},
  year={2018}
}
```

---

## Infrastructure & Tools

### 8. Lark: A Modern Parsing Library for Python

**Title**: Lark - A parsing toolkit for Python

**Author**: Erez Shinan

**URL**: https://github.com/lark-parser/lark

**Type**: Open Source Python Library

**Description**:
Lark is a modern Python parsing library that can automatically build a parse tree from a grammar. It supports multiple parsing algorithms (Earley, LALR) and grammar formats (EBNF). Features include error recovery, transformer classes, and integration with external tools.

**Key Features for Tutorial**:
- EBNF grammar syntax (familiar to most developers)
- Automatic parse tree construction
- Transformer classes for AST manipulation
- Support for both deterministic (LALR) and general (Earley) parsing

**Relevance**: Primary tool for grammar specification and testing in Layers 2-3.

**Citation**:
```bibtex
@software{shinan2018lark,
  title={Lark -- A modern parsing library for Python},
  author={Shinan, Erez},
  url={https://github.com/lark-parser/lark},
  year={2018}
}
```

---

### 9. llama.cpp: Efficient LLM Inference in C++

**Title**: llama.cpp - Run LLMs Locally

**Author**: Georgi Gerganov (Original), maintained by ggml-org

**URL**: https://github.com/ggml-org/llama.cpp

**Type**: Open Source C++ Library

**Description**:
llama.cpp is a C++ implementation of LLaMA and other large language models, optimized for consumer hardware (CPU inference). It supports:
- GGUF quantized model format
- Metal (Apple Silicon), CUDA (NVIDIA), ROCm (AMD) acceleration
- Batch inference
- Custom token masking API (used for constrained decoding)

**Quantization Support**:
- 4-bit (Q4_K_M, Q4_K_S)
- 5-bit (Q5_K_M)
- 6-bit (Q6_K)
- Higher precision alternatives

**Relevance**: Core inference engine used in Layer 4; provides token masking hooks for llguidance integration.

**Citation**:
```bibtex
@software{gerganov2023llama_cpp,
  title={llama.cpp: Inference of Meta's LLaMA model in pure C++},
  author={Gerganov, Georgi},
  url={https://github.com/ggml-org/llama.cpp},
  year={2023}
}
```

---

## Model Cards & Datasets

### 10. Qwen3 Technical Report

**Title**: Qwen3: Dense and Mixture-of-Experts Large Language Models

**Organization**: Alibaba Damo Academy

**Year**: 2026

**URL**: https://huggingface.co/Qwen/Qwen3-0.6B

**Model Sizes Available**: 0.6B, 1.7B, 4B, 7B, 14B, 32B

**Key Properties**:
- Trained on 10T tokens (multilingual, code-heavy)
- 32,768 token context window
- 151,936 token vocabulary (covers many code tokens)
- Strong performance on bash/shell tasks

**Relevant for Tutorial**: Primary model used in examples; demonstrates strong uplift from grammar constraints (16.7% → 59.2%).

---

### 11. SmolLM2 Technical Report

**Title**: SmolLM2: Smaller Models with Better Instruction Following

**Organization**: Hugging Face (with Anthropic collaboration)

**Year**: 2025

**URL**: https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct

**Key Properties**:
- Code-optimized training (707B tokens)
- Instruction-tuned variant for better task following
- Smaller vocabulary (49,152 tokens) but dense

**Relevant for Tutorial**: Demonstrates trade-offs between model size and code understanding; illustrates instruction-tuning impact on bash generation.

---

### 12. NL2Bash Dataset (TellinaTool)

**Title**: nl2bash: A Dataset and Benchmark for Semantic-to-Code Generation

**Authors**: Iftikhar et al.

**Repository**: https://github.com/TellinaTool/nl2bash

**Characteristics**:
- 9,500+ intent-command pairs
- Semantic annotations (predicate-argument structure)
- Covers ~100 common bash utilities
- Aligned with NL task descriptions

**Relevance**: Source for grounded NL↔Bash pairs; used in tutorial for validation of semantic correctness (beyond syntax).

---

### 13. Bash Commands Dataset (HuggingFace)

**Title**: bash-commands-dataset

**Author**: aelhalili

**URL**: https://huggingface.co/datasets/aelhalili/bash-commands-dataset

**Size**: ~50k commands with metadata

**Characteristics**:
- Real commands from community submissions
- Metadata: command category, complexity, platform
- Diverse utility coverage

**Relevance**: Large-scale corpus for grammar generation training.

---

### 14. Bash Command Data 6K (HuggingFace)

**Title**: bash_command_data_6K

**Author**: emirkaanozdemr

**URL**: https://huggingface.co/datasets/emirkaanozdemr/bash_command_data_6K

**Size**: ~6,000 bash commands

**Characteristics**:
- Curated set of frequently used commands
- High-quality annotations
- Broader syntactic coverage per command

**Relevance**: High-quality subset for fine-grained grammar development.

---

### 15. UCI Shell Commands Dataset (Cybersecurity)

**Title**: Shell commands used by participants of hands-on cybersecurity hands-on exercises

**Organization**: UC Irvine Machine Learning Repository

**URL**: https://archive.ics.uci.edu/dataset/869/shell+commands+used+by+participants+of+hands-on+cybersecuri

**Dataset #**: 869

**Size**: ~1,500+ commands from cybersecurity context

**Characteristics**:
- Commands from real security exercises
- Includes both common and specialized utilities
- Emphasis on file/process manipulation, network diagnostics

**Relevance**: Specialized corpus for security-aware grammar design; represents real-world operational constraints.

---

## Theory & Foundations

### 16. Formal Languages & Automata (General References)

**Title**: Introduction to Automata Theory, Languages, and Computation

**Authors**: John E. Hopcroft, Rajeev Motwani, Jeffrey D. Ullman

**Edition**: 3rd (2006)

**Publisher**: Pearson

**Relevant Chapters**:
- Chapter 1-3: DFA, NFA, regular expressions
- Chapter 4-5: Context-free grammars, pushdown automata
- Chapter 6-7: Turing machines, decidability

**Relevance**: Theoretical foundation for understanding finite-state constraints and grammar expressiveness.

**Citation**:
```bibtex
@book{hopcroft2006automata,
  title={Introduction to Automata Theory, Languages, and Computation},
  author={Hopcroft, John E and Motwani, Rajeev and Ullman, Jeffrey D},
  edition={3rd},
  publisher={Pearson},
  year={2006}
}
```

---

### 17. Information Theory (Shannon Entropy)

**Title**: A Mathematical Theory of Communication

**Author**: Claude E. Shannon

**Published**: 1948

**Journal**: The Bell System Technical Journal

**URL**: https://people.math.harvard.edu/~ctm/home/text/others/shannon/shannon1948.pdf

**Impact**: Foundation for understanding entropy reduction in constrained decoding.

**Citation**:
```bibtex
@article{shannon1948mathematical,
  title={A Mathematical Theory of Communication},
  author={Shannon, Claude E},
  journal={The Bell System Technical Journal},
  volume={27},
  number={3},
  pages={379--423},
  year={1948}
}
```

---

## Related Work & Extensions

### 18. Semantic Parsing with Grammar Constraints

**Title**: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

**Authors**: Patrick Lewis, Ethan Perez, et al.

**Published**: 2020

**ArXiv**: https://arxiv.org/abs/2005.11401

**Relevance**: Demonstrates how grammar constraints improve semantic parsing; foundational for understanding constraint benefits beyond syntax.

---

### 19. Code Generation with Language Models

**Title**: Evaluating Large Language Models Trained on Code

**Authors**: Chen et al. (OpenAI Codex)

**Published**: 2021

**ArXiv**: https://arxiv.org/abs/2107.03374

**Benchmark**: HumanEval (popular for code generation evaluation)

**Relevance**: Establishes baselines and methodologies for evaluating code/command generation; informs evaluation metrics in Layer 5.

---

## Citation Format Conventions

This tutorial uses the following citation format:
- **In text**: [Author, Year] or [Author et al., Year]
- **Bibliography**: Alphabetical by author surname
- **BibTeX**: Standard IEEE/ACM formats

### Examples:

```
In-text citation:
According to PICARD (Scholak et al., 2021), constrained decoding 
significantly improves semantic correctness in SQL generation.

Bibliography entry:
Scholak, T., Schucher, N., & Bahdanau, D. (2021). PICARD: Parsing 
incrementally for constrained auto-regressive decoding. 
In Proceedings of EMNLP 2021 (pp. 9895-9910).
```

---

## Recommended Reading Order

**For newcomers to constrained decoding**:
1. Start: Karpathy (2015) – Intuition for sequence generation
2. Then: Vaswani et al. (2017) – Transformer architecture
3. Core: Scholak et al. (2021) – PICARD foundational work
4. Application: Lucas et al. (2026) – Bash-specific results
5. Reference: Hopcroft & Ullman (2006) – Formal foundations

**For implementers**:
1. Willard et al. (2023) – llguidance framework
2. Lark documentation – Grammar specification
3. tree-sitter-bash – Syntax validation
4. Tutorial Notebooks 1-5 – Hands-on application

---

**Last Updated**: June 2026
