"""
Inference engine for grammar-constrained bash generation.

Integrates llama.cpp (C++ inference) with llguidance (grammar constraints).
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for inference model."""
    model_path: str                    # Path to GGUF model file
    n_ctx: int = 2048                 # Context window
    n_threads: int = 4                # CPU threads
    n_gpu_layers: int = 0             # GPU layers (0 = CPU only)
    temperature: float = 0.7          # Sampling temperature
    top_p: float = 0.95               # Nucleus sampling
    top_k: int = 50                   # Top-k sampling
    use_mmap: bool = True             # Memory-mapped file access
    use_mlock: bool = False           # Lock in physical memory


class InferenceBackend:
    """
    Abstract base class for inference backends.
    """
    
    def generate(
        self,
        prompt: str,
        max_tokens: int,
        grammar: Optional[str] = None,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Generate text with optional grammar constraint."""
        raise NotImplementedError
    
    def get_logits(
        self,
        prompt: str,
    ) -> Dict[str, float]:
        """Get logits for next token given prompt."""
        raise NotImplementedError


class LlamaCppBackend(InferenceBackend):
    """
    Wrapper for llama.cpp inference.
    
    This is a stub for the actual implementation.
    In production, would use llama-cpp-python:
    ```
    pip install llama-cpp-python
    from llama_cpp import Llama
    ```
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        # In production:
        # self.model = Llama(model_path, **config.__dict__)
        self.model = None  # Placeholder
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        grammar: Optional[str] = None,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """
        Generate text with optional grammar constraint.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            grammar: Optional Lark grammar string (applied via llguidance)
            temperature: Sampling temperature
            stop_sequences: Stop generation at these sequences
        
        Returns:
            Generated text
        """
        # Production implementation would:
        # 1. Compile grammar to FSM (via llguidance)
        # 2. Call self.model.create_completion()
        # 3. Apply token masking at each step
        raise NotImplementedError("LlamaCppBackend requires llama-cpp-python package")
    
    def get_logits(self, prompt: str) -> Dict[str, float]:
        """Get logits for next token."""
        raise NotImplementedError


class LangChainInferenceAdapter(InferenceBackend):
    """
    Adapter for using any LangChain LLM with grammar constraints.
    
    Enables integration with models deployed via:
    - LlamaCpp
    - OpenAI API
    - Custom providers
    """
    
    def __init__(self, llm: Any):
        """
        Initialize with LangChain LLM.
        
        Args:
            llm: LangChain BaseLLM instance
        """
        self.llm = llm
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        grammar: Optional[str] = None,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """
        Generate with grammar constraints.
        
        Note: LangChain native integration with llguidance would require
        custom implementation. This shows the interface.
        """
        # If grammar provided, would need to:
        # 1. Compile grammar to constraint function
        # 2. Implement streaming generation with token masking
        # 3. Fall back to native generation if constraint violated
        
        from langchain.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        chain = self.llm | StrOutputParser()
        result = chain.invoke(prompt)
        return result[:max_tokens*4]  # Rough token limit
    
    def get_logits(self, prompt: str) -> Dict[str, float]:
        """Not supported by LangChain interface."""
        raise NotImplementedError("LangChain interface doesn't expose logits")


class ConstrainedInferenceEngine:
    """
    High-level engine for grammar-constrained bash generation.
    
    Coordinates:
    - Grammar compilation (IR → Lark → FSM)
    - Inference with masking
    - Syntax validation
    - Retry/fallback logic
    """
    
    def __init__(
        self,
        backend: InferenceBackend,
        grammar_path: Optional[str] = None,
        validator: Optional[Callable] = None,
    ):
        """
        Initialize constrained inference engine.
        
        Args:
            backend: Inference backend (LlamaCpp, LangChain, etc.)
            grammar_path: Path to Lark grammar file
            validator: Optional function to validate generated commands
                       Signature: (command: str) -> (is_valid: bool, error: Optional[str])
        """
        self.backend = backend
        self.grammar_path = grammar_path
        self.validator = validator
        self.grammar = self._load_grammar(grammar_path) if grammar_path else None
    
    def _load_grammar(self, path: str) -> Optional[str]:
        """Load Lark grammar from file."""
        try:
            with open(path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            return None
    
    def generate_constrained(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        retry_on_failure: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate bash command with grammar constraints and validation.
        
        Args:
            prompt: Task description or partial command
            max_tokens: Maximum generation length
            temperature: Sampling temperature
            retry_on_failure: If validation fails, retry without constraint
        
        Returns:
            Dict with:
            - 'command': Generated bash command
            - 'valid': Whether command passed validation
            - 'constrained': Whether constraints were applied
            - 'error': Optional error message
            - 'retried': Whether retry was needed
        """
        # Phase 1: Try constrained generation
        constrained_output = self.backend.generate(
            prompt,
            max_tokens=max_tokens,
            grammar=self.grammar,
            temperature=temperature,
        )
        
        # Phase 2: Validate
        is_valid = True
        error = None
        
        if self.validator:
            is_valid, error = self.validator(constrained_output)
        
        if is_valid:
            return {
                'command': constrained_output,
                'valid': True,
                'constrained': True,
                'error': None,
                'retried': False,
            }
        
        # Phase 3: Retry without constraint if enabled
        if retry_on_failure:
            native_output = self.backend.generate(
                prompt,
                max_tokens=max_tokens,
                grammar=None,  # No constraint
                temperature=temperature,
            )
            
            is_valid_native = True
            error_native = None
            
            if self.validator:
                is_valid_native, error_native = self.validator(native_output)
            
            return {
                'command': native_output,
                'valid': is_valid_native,
                'constrained': False,
                'error': error_native,
                'retried': True,
            }
        
        # No retry: return failed constrained output
        return {
            'command': constrained_output,
            'valid': False,
            'constrained': True,
            'error': error,
            'retried': False,
        }
    
    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple commands in batch.
        """
        return [self.generate_constrained(p, max_tokens) for p in prompts]
