# Grammar-Constrained Decoding for Bash Generation: A Technical Tutorial

## Overview

This tutorial suite provides a production-grade implementation of grammar-constrained decoding (GCD) for small language models generating shell commands. Based on the NVIDIA AI Red Team's research ([Lucas et al., 2026](./references.md)), this system demonstrates how to uplift model performance from 62.5% to 75.2% mean pass rate by constraining the sampling distribution during autoregressive decoding.

**Target Outcome**: A fully transferable LangChain tool for deploying bash-generating agents with syntax guarantees in agentic workflows.

---

## Learning Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: Constrained Decoding Foundations                       │
│ • Autoregressive sampling theory                                 │
│ • Token masking & distribution modification                      │
│ • Finite-state constraint application                            │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2: Grammar Generation (grammargen)                         │
│ • Lark formal grammar specification                              │
│ • Command evidence aggregation                                   │
│ • Grammar IR (Intermediate Representation)                       │
│ • EBNF → Lark compilation                                        │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3: Grammar Validation & Refinement                         │
│ • Structural soundness (parse-tree validation)                   │
│ • Behavioral coverage (accept known-good, reject known-bad)     │
│ • Policy encoding (forbidden flags, required arguments)          │
│ • Regression detection & grammar refinement                      │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│ Layer 4: Inference Integration (llguidance + llama.cpp)          │
│ • Token ID masking during sampling                               │
│ • Grammar state machine advancement                              │
│ • Entropy reduction & probability redistribution                 │
│ • Batch inference optimization                                   │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│ Layer 5: Evaluation & Deployment                                 │
│ • Syntax validation (tree-sitter-bash)                           │
│ • Semantic correctness (task success metrics)                    │
│ • Runtime safety checks                                          │
│ • LangChain integration & agentic composition                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Notebook Structure

| # | Notebook | Focus | Duration | Output |
|---|----------|-------|----------|--------|
| **1** | `01_constrained_decoding_foundations.ipynb` | Token masking, sampling distributions, constraint FSMs | ~90 min | Theoretical foundation + simple masking demo |
| **2** | `02_grammar_generation.ipynb` | Lark syntax, grammargen tool, IR compilation | ~120 min | Working grammar generator for 10+ bash commands |
| **3** | `03_grammar_validation.ipynb` | Structural soundness, behavioral coverage, policy encoding | ~100 min | Validated grammar suite with regression tests |
| **4** | `04_inference_integration.ipynb` | llama.cpp setup, llguidance binding, batch inference | ~140 min | Constrained sampling pipeline end-to-end |
| **5** | `05_evaluation_and_deployment.ipynb` | tree-sitter validation, metrics, LangChain tool | ~110 min | Production-ready LangChain tool + evaluation harness |

**Total instructional time**: ~9-10 hours

---

## Key References

- **Constrained Decoding Theory**: [PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding](https://arxiv.org/abs/2109.05093) (Scholak et al., 2021)
- **Grammar-Constrained Decoding for Bash**: [Improving Bash Generation in Small Language Models with Grammar-Constrained Decoding](https://developer.nvidia.com/blog/improving-bash-generation-in-small-language-models-with-grammar-constrained-decoding/) (Lucas et al., 2026)
- **Lark Parser**: [Lark: A parsing toolkit for Python](https://github.com/lark-parser/lark)
- **llguidance**: [Guidance for Large Language Models](https://github.com/guidance-ai/llguidance)

See `./references.md` for complete bibliography with primary literature citations.

---

## Data Sources

This tutorial integrates real-world bash command distributions from:

1. **HuggingFace**: `aelhalili/bash-commands-dataset` (~50k commands with metadata)
2. **HuggingFace**: `emirkaanozdemr/bash_command_data_6K` (~6k bash commands)
3. **nl2bash (TellinaTool)**: Intent-aligned NL→Bash pairs for semantic grounding
4. **UCI Dataset 869**: Shell commands from cybersecurity participant exercises

The tutorial samples ~200 representative commands to build and validate grammars, with references to full datasets for production scaling.

---

## System Requirements

### Software Stack
- **Python**: 3.10+
- **Core Dependencies**: lark-parser, tree-sitter, datasets, numpy, pandas
- **Inference**: llama.cpp (C++ backend), llguidance (Python bindings)
- **Optional**: NVIDIA CUDA 11.8+ (for GPU acceleration)

### Hardware
- **CPU-only**: 4GB RAM minimum (quantized models)
- **Recommended**: 8GB+ RAM, GPU with 4GB+ VRAM (for real-time inference)
- **Development**: Standard laptop sufficient for all tutorials

### Installation

```bash
# Clone tutorial repo
git clone <tutorial-repo>
cd grammar-constrained-bash-tutorial

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download model weights & datasets (see 04_inference_integration.ipynb)
python scripts/download_models.py
python scripts/download_datasets.py
```

See `SETUP.md` for detailed platform-specific instructions.

---

## LangChain Integration

The final deliverable is a production-ready LangChain tool:

```python
from langchain.tools import Tool
from bash_grammar_tool import BashGeneratorWithConstraints

bash_tool = Tool(
    name="constrained_bash_generator",
    description="Generates syntactically valid bash commands with grammar constraints",
    func=BashGeneratorWithConstraints(
        model_name="Qwen3-0.6B",
        grammar_path="./grammars/bash_commands.lark",
        inference_backend="llama_cpp"
    ).generate
)

# Use in agentic workflow
agent.add_tool(bash_tool)
```

Full implementation in `05_evaluation_and_deployment.ipynb` and `src/langchain_integration.py`.

---

## File Organization

```
grammar-constrained-bash-tutorial/
├── README.md                           # This file
├── SETUP.md                            # Detailed installation guide
├── references.md                       # Bibliography & nomenclature
├── glossary.md                         # Technical terms & notation
├── model_architecture.md               # Model descriptions with diagrams
│
├── notebooks/
│   ├── 01_constrained_decoding_foundations.ipynb
│   ├── 02_grammar_generation.ipynb
│   ├── 03_grammar_validation.ipynb
│   ├── 04_inference_integration.ipynb
│   └── 05_evaluation_and_deployment.ipynb
│
├── src/
│   ├── constrained_decoding.py         # Core masking & sampling
│   ├── grammar_generator.py            # Lark grammar compilation
│   ├── grammar_validator.py            # Soundness & coverage checks
│   ├── inference_engine.py             # llama.cpp + llguidance wrapper
│   ├── evaluation_metrics.py           # Pass rates & regression detection
│   ├── langchain_integration.py        # LangChain tool definition
│   └── data_loaders.py                 # HF dataset utilities
│
├── grammars/
│   ├── bash_core.lark                  # Base bash grammar template
│   ├── common_commands/                # Pre-generated grammars
│   │   ├── grep.lark
│   │   ├── openssl.lark
│   │   ├── cat.lark
│   │   └── ...
│   └── generated/                      # Runtime-generated grammars
│
├── data/
│   ├── sample_commands.json            # Tutorial dataset (~200 commands)
│   ├── evaluation_tasks.json            # 299 tasks from paper
│   └── bash_syntax_reference.md        # Bash syntax guide (auto-generated)
│
├── tests/
│   ├── test_masking.py
│   ├── test_grammar_generation.py
│   ├── test_validation.py
│   └── test_integration.py
│
├── scripts/
│   ├── download_models.py
│   ├── download_datasets.py
│   ├── generate_bash_syntax_ref.py     # Analyze datasets for syntax
│   └── run_full_pipeline.py
│
└── requirements.txt
```

---

## Quick Start

For the impatient:

```bash
# 1. Setup
cd grammar-constrained-bash-tutorial && source venv/bin/activate

# 2. Run Layer 4 notebook (inference) with pre-built grammars
jupyter notebook notebooks/04_inference_integration.ipynb

# 3. Get LangChain tool immediately
from src.langchain_integration import get_constrained_bash_tool
tool = get_constrained_bash_tool()
```

---

## Advanced: Custom Grammar Development

To develop grammars for your own command set:

```python
from src.grammar_generator import GrammarIR, LarkCompiler
from datasets import load_dataset

# Load your command data
commands = load_dataset("your-dataset")

# Build grammar IR from evidence
ir = GrammarIR.from_command_corpus(
    commands["bash"],
    command_name="mycommand",
    flag_threshold=0.05  # Flags appearing in 5%+ of commands
)

# Compile to Lark
grammar_text = LarkCompiler.compile(ir)

# Validate
from src.grammar_validator import GrammarValidator
validator = GrammarValidator(grammar_text)
validator.test_corpus(commands["bash"])
```

See `02_grammar_generation.ipynb` for complete examples.

---

## Troubleshooting & Common Issues

| Issue | Solution |
|-------|----------|
| **llguidance import error** | Install from source: `pip install git+https://github.com/guidance-ai/llguidance` |
| **Model download timeout** | Use `HF_HOME=/path/to/cache` for custom cache location |
| **Out of memory on inference** | Use quantized models (GGUF format, Q4_K_M quantization) |
| **Grammar too restrictive** | Increase `flag_threshold` in IR generation or add policy exceptions |

See `TROUBLESHOOTING.md` for extended diagnostics.

---

## Contributing & Extensions

This tutorial is designed for extension:

- **Add new command grammars**: Follow pattern in `grammars/common_commands/`
- **Integrate new models**: Implement `InferenceBackend` interface in `src/inference_engine.py`
- **Custom evaluation metrics**: Extend `EvaluationMetrics` in `src/evaluation_metrics.py`
- **Alternative constraint methods**: See `src/constrained_decoding.py` for masking patterns

---

## Citation

If you use this tutorial in research or production:

```bibtex
@article{lucas2026improving,
  title={Improving Bash Generation in Small Language Models with Grammar-Constrained Decoding},
  author={Lucas, Joseph},
  journal={NVIDIA Technical Blog},
  year={2026},
  url={https://developer.nvidia.com/blog/improving-bash-generation-in-small-language-models-with-grammar-constrained-decoding/}
}

@article{scholak2021picard,
  title={PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding},
  author={Scholak, Torsten and Schucher, Nathan and Bahdanau, Dzmitry},
  journal={arXiv preprint arXiv:2109.05093},
  year={2021}
}
```

---

## License

Educational use under CC-BY-4.0. See LICENSE for commercial terms.

---

**Questions?** Open an issue or consult the glossary (`glossary.md`) and references (`references.md`).
