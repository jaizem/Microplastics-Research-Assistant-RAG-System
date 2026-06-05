"""
Grammar generation from command evidence using intermediate representation (IR).

This module implements grammargen-inspired grammar generation:
1. Parse command evidence (real bash commands)
2. Extract command structure into Grammar IR
3. Compile IR to Lark formal grammar
4. Validate against corpus
"""

from dataclasses import dataclass, field
from typing import Set, Dict, List, Tuple, Optional, Any
from enum import Enum
import re
from collections import defaultdict, Counter
import json


class ArgType(Enum):
    """Argument type classification."""
    WORD = "WORD"          # Generic word/string
    PATH = "PATH"          # File path
    INT = "INT"            # Integer
    REGEX = "REGEX"        # Regular expression
    HOSTNAME = "HOSTNAME"  # Hostname/IP
    URL = "URL"            # URL


@dataclass
class FlagInfo:
    """Information about a single flag."""
    name: str                           # Flag name (e.g., "-i", "--ignore-case")
    is_short: bool                      # True if short flag (-x), False if long (--xxx)
    is_valued: bool                     # True if flag takes argument
    value_type: Optional[ArgType] = None # Type of value if valued
    frequency: float = 0.0              # Proportion of commands using this flag


@dataclass
class GrammarIR:
    """
    Intermediate representation for command grammar.
    
    Captures command structure without full grammar verbosity,
    enabling programmatic grammar generation from command evidence.
    """
    command_name: str
    short_flags: Set[str] = field(default_factory=set)
    long_flags: Dict[str, bool] = field(default_factory=dict)  # name -> is_valued
    positional_args: List[ArgType] = field(default_factory=list)
    flag_frequency: Dict[str, float] = field(default_factory=dict)
    max_flags: int = 8                  # Maximum flags allowed (for bounded repetition)
    max_positional: int = 5             # Maximum positional arguments
    allows_pipe: bool = True            # Can participate in pipes
    allows_redirect: bool = True        # Can use > < >> redirections
    composed_commands: Set[str] = field(default_factory=set)  # e.g., {"pipe", "and", "or"}
    
    @classmethod
    def from_command_corpus(cls, commands: List[str], command_name: str, 
                           flag_threshold: float = 0.05) -> 'GrammarIR':
        """
        Build Grammar IR from evidence of actual commands.
        
        Args:
            commands: List of bash commands to analyze
            command_name: Name of the command (e.g., "grep")
            flag_threshold: Minimum frequency for flag inclusion (0-1)
        
        Returns:
            GrammarIR object encoding command structure
        """
        ir = cls(command_name=command_name)
        
        flag_count = defaultdict(int)
        arg_types = defaultdict(lambda: defaultdict(int))
        
        for cmd in commands:
            # Simple tokenization (would be more sophisticated in practice)
            tokens = cmd.split()
            
            # Skip if doesn't start with command
            if not tokens or tokens[0] != command_name:
                continue
            
            # Parse remaining tokens
            i = 1
            positional_idx = 0
            
            while i < len(tokens):
                token = tokens[i]
                
                if token.startswith('--'):  # Long flag
                    flag_name = token.split('=')[0]  # Remove value if attached
                    is_valued = '=' in token or (i + 1 < len(tokens) and not tokens[i+1].startswith('-'))
                    flag_count[flag_name] += 1
                    ir.long_flags[flag_name] = is_valued
                    if is_valued and '=' not in token and i + 1 < len(tokens):
                        i += 1
                
                elif token.startswith('-') and token != '-':  # Short flag
                    for flag_char in token[1:]:
                        flag_key = f'-{flag_char}'
                        flag_count[flag_key] += 1
                        ir.short_flags.add(flag_key)
                
                elif not token.startswith('-'):  # Positional argument
                    if positional_idx < len(ir.positional_args):
                        arg_types[positional_idx][ir.positional_args[positional_idx]] += 1
                    else:
                        # Infer type from token
                        inferred_type = cls._infer_arg_type(token)
                        ir.positional_args.append(inferred_type)
                        arg_types[positional_idx][inferred_type] += 1
                    positional_idx += 1
                
                i += 1
        
        # Filter flags by frequency threshold
        total_cmds = len(commands)
        ir.flag_frequency = {
            flag: count / total_cmds 
            for flag, count in flag_count.items() 
            if count / total_cmds >= flag_threshold
        }
        
        # Keep only frequent flags
        ir.short_flags = {f for f in ir.short_flags if f in ir.flag_frequency}
        ir.long_flags = {f: v for f, v in ir.long_flags.items() if f in ir.flag_frequency}
        
        return ir
    
    @staticmethod
    def _infer_arg_type(token: str) -> ArgType:
        """Heuristic type inference from token."""
        if re.match(r'^\d+$', token):
            return ArgType.INT
        elif '/' in token or '.' in token:
            return ArgType.PATH
        elif re.match(r'^\d{1,3}\.', token):  # IP-like
            return ArgType.HOSTNAME
        elif token.startswith('http'):
            return ArgType.URL
        elif token.startswith('^') or token.endswith('$'):  # Regex-like
            return ArgType.REGEX
        else:
            return ArgType.WORD
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'command_name': self.command_name,
            'short_flags': sorted(list(self.short_flags)),
            'long_flags': self.long_flags,
            'positional_args': [arg.value for arg in self.positional_args],
            'flag_frequency': self.flag_frequency,
            'max_flags': self.max_flags,
            'max_positional': self.max_positional,
            'allows_pipe': self.allows_pipe,
            'allows_redirect': self.allows_redirect,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GrammarIR':
        """Deserialize from dictionary."""
        ir = cls(command_name=data['command_name'])
        ir.short_flags = set(data.get('short_flags', []))
        ir.long_flags = data.get('long_flags', {})
        ir.positional_args = [ArgType(arg) for arg in data.get('positional_args', [])]
        ir.flag_frequency = data.get('flag_frequency', {})
        ir.max_flags = data.get('max_flags', 8)
        ir.max_positional = data.get('max_positional', 5)
        ir.allows_pipe = data.get('allows_pipe', True)
        ir.allows_redirect = data.get('allows_redirect', True)
        return ir


class LarkCompiler:
    """
    Compiles Grammar IR to Lark EBNF grammar.
    
    Lark syntax conventions:
    - rule_name: production | alternatives
    - Terminal (literal): "string" or /regex/
    - Non-terminal (rule reference): rule_name
    - Modifiers: {n,m} = repetition, ? = optional, + = one-or-more, * = zero-or-more
    """
    
    @staticmethod
    def compile(ir: GrammarIR) -> str:
        """
        Compile Grammar IR to Lark EBNF.
        
        Args:
            ir: Grammar intermediate representation
        
        Returns:
            Lark grammar as string
        """
        lines = []
        lines.append("// Auto-generated Lark grammar")
        lines.append(f"// Command: {ir.command_name}")
        lines.append("")
        
        # Start rule
        options_part = LarkCompiler._make_options_rule(ir.command_name)
        positional_part = " positional_args" if ir.positional_args else ""
        lines.append(f'start: "{ir.command_name}" {options_part} WS{positional_part}')
        lines.append("")
        
        # Options rule (flags)
        lines.append(f"{ir.command_name}_opt: " + LarkCompiler._make_flag_alternatives(ir))
        lines.append("")
        
        # Positional arguments
        if ir.positional_args:
            for i, arg_type in enumerate(ir.positional_args):
                lines.append(f"pos_arg_{i}: {arg_type.value}")
            lines.append("positional_args: " + " ".join([f"pos_arg_{i}" for i in range(len(ir.positional_args))]) + "?")
            lines.append("")
        
        # Terminals
        lines.extend(LarkCompiler._make_terminals(ir))
        
        return "\n".join(lines)
    
    @staticmethod
    def _make_options_rule(cmd_name: str, max_flags: int = 8) -> str:
        """Generate options repetition rule."""
        return f"({cmd_name}_opt WS){{{0},{max_flags}}}"
    
    @staticmethod
    def _make_flag_alternatives(ir: GrammarIR) -> str:
        """Generate flag alternatives."""
        alternatives = []
        
        # Short flags (can be combined like -iRv)
        if ir.short_flags:
            short_flag_chars = ''.join(f[1] for f in sorted(ir.short_flags))
            alternatives.append(f'"-" /[{short_flag_chars}]+/')
        
        # Individual long flags
        for flag_name, is_valued in sorted(ir.long_flags.items()):
            if is_valued:
                alternatives.append(f'"{flag_name}" ("=" | WS) WORD')
            else:
                alternatives.append(f'"{flag_name}"')
        
        # Individual short flags with values
        for flag in sorted(ir.short_flags):
            alternatives.append(f'"{flag}" WS WORD')
        
        return " | ".join(alternatives) if alternatives else "WORD"
    
    @staticmethod
    def _make_terminals(ir: GrammarIR) -> List[str]:
        """Generate terminal definitions."""
        terminals = [
            'WORD: /[^\\s|><&;()]{1,200}/',
            'PATH: /[^\\s|><&;()]{1,200}/',
            'WS: /\\s+/',
        ]
        return terminals


class GrammarValidator:
    """
    Validates grammar IR for soundness and coverage.
    """
    
    @staticmethod
    def check_soundness(ir: GrammarIR) -> Dict[str, bool]:
        """
        Check basic soundness properties.
        
        Returns:
            Dict with boolean checks
        """
        return {
            'has_command_name': bool(ir.command_name),
            'max_flags_reasonable': 0 < ir.max_flags <= 20,
            'max_positional_reasonable': 0 < ir.max_positional <= 10,
            'has_some_structure': bool(ir.short_flags or ir.long_flags or ir.positional_args),
        }
    
    @staticmethod
    def check_coverage(ir: GrammarIR, test_commands: List[str]) -> Dict[str, Any]:
        """
        Check grammar coverage on test commands.
        
        Returns:
            Coverage statistics
        """
        # This would require actual Lark parsing
        # Simplified version just counts accepted patterns
        
        matches = 0
        for cmd in test_commands:
            tokens = cmd.split()
            if tokens and tokens[0] == ir.command_name:
                matches += 1
        
        return {
            'coverage': matches / len(test_commands) if test_commands else 0,
            'matched': matches,
            'total': len(test_commands),
        }
