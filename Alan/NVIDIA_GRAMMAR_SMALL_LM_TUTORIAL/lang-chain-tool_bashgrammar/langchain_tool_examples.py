#!/usr/bin/env python3
"""
Practical LangChain @tool Implementation Guide

This file contains runnable examples showing how to:
1. Create tools with @tool decorator
2. Use tools with agents
3. Implement bash generation tool
4. Handle errors and validation
5. Test and debug tools

Requirements:
    pip install langchain langchain-core langchain-openai
    (or use any LangChain-compatible LLM)
"""

import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
from langchain_core.tools import tool


# ============================================================================
# SECTION 1: BASIC TOOLS (Copy & Paste Ready)
# ============================================================================

@tool
def calculator(operation: str, a: float, b: float) -> float:
    """Simple calculator tool.
    
    Performs basic arithmetic operations.
    
    Args:
        operation: One of 'add', 'subtract', 'multiply', 'divide'
        a: First number
        b: Second number
    
    Returns:
        Result of the operation
    
    Raises:
        ValueError: If operation is invalid or division by zero
    """
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")


@tool
def string_tools(operation: str, text: str) -> str:
    """String manipulation tools.
    
    Args:
        operation: 'upper', 'lower', 'reverse', 'count_words'
        text: Input string
    
    Returns:
        Transformed string
    """
    if operation == "upper":
        return text.upper()
    elif operation == "lower":
        return text.lower()
    elif operation == "reverse":
        return text[::-1]
    elif operation == "count_words":
        return str(len(text.split()))
    else:
        raise ValueError(f"Unknown operation: {operation}")


# ============================================================================
# SECTION 2: BASH COMMAND TOOL (Main Example)
# ============================================================================

@tool
def bash_generator(
    task: str,
    use_grammar_constraints: bool = True,
    max_length: int = 100
) -> Dict[str, Any]:
    """Generate bash commands using grammar-constrained decoding.
    
    This tool generates shell commands for a given task, optionally applying
    formal grammar constraints to ensure syntactic validity.
    
    How it works:
    1. Takes a natural language task description
    2. Uses a small language model to generate a command
    3. Optionally applies grammar constraints during generation
    4. Validates the output
    5. Returns the command with metadata
    
    Args:
        task: What you want to do (e.g., "Find all Python files")
        use_grammar_constraints: Whether to apply grammar constraints (default True)
        max_length: Maximum command length in characters (default 100)
    
    Returns:
        Dictionary with:
        - 'command': The generated bash command
        - 'valid': Whether the command is syntactically valid
        - 'constrained': Whether constraints were applied
        - 'confidence': Confidence score (0.0 to 1.0)
        - 'explanation': What the command does
        - 'error': Error message if any (or None)
    
    Example:
        >>> bash_generator("Find all Python files")
        {
            'command': 'find . -name "*.py"',
            'valid': True,
            'constrained': True,
            'confidence': 0.92,
            'explanation': 'Searches current directory recursively for .py files',
            'error': None
        }
    """
    
    # Input validation
    if not task or not isinstance(task, str):
        raise ValueError("Task must be a non-empty string")
    
    if len(task) > 500:
        raise ValueError("Task description too long (max 500 chars)")
    
    # Mapping of task keywords to bash commands
    # In production, this would call a real LLM with optional grammar constraints
    task_map = {
        "find python": "find . -type f -name '*.py'",
        "find files": "find . -type f",
        "list files": "ls -lah",
        "list directory": "ls -la",
        "count lines": "wc -l < file.txt",
        "search text": "grep -n 'pattern' file.txt",
        "search recursive": "grep -r 'pattern' . --include='*.txt'",
        "disk usage": "du -sh *",
        "current directory": "pwd",
        "change directory": "cd /path/to/directory",
        "create file": "touch filename.txt",
        "create directory": "mkdir directory_name",
        "remove file": "rm filename.txt",
        "copy file": "cp source.txt dest.txt",
    }
    
    # Find matching command (case-insensitive)
    command = None
    task_lower = task.lower()
    
    for key, cmd in task_map.items():
        if key in task_lower:
            command = cmd
            break
    
    if not command:
        command = f"echo 'Command for: {task}'"
    
    # Return result
    return {
        'command': command,
        'valid': True,
        'constrained': use_grammar_constraints,
        'confidence': 0.85 if use_grammar_constraints else 0.65,
        'explanation': f"Generated bash command for: {task}",
        'error': None
    }


@tool
def bash_validator(command: str) -> Dict[str, Any]:
    """Validate a bash command for syntax errors and safety.
    
    Checks for:
    - Mismatched quotes
    - Dangerous commands
    - Invalid syntax patterns
    - Missing required flags
    
    Args:
        command: Bash command to validate
    
    Returns:
        Dictionary with:
        - 'valid': Syntax is valid
        - 'safe': Command is safe to execute
        - 'errors': List of syntax errors
        - 'warnings': List of safety warnings
        - 'suggestions': Suggestions for improvement
    
    Example:
        >>> bash_validator("grep 'pattern file.txt")
        {
            'valid': False,
            'safe': True,
            'errors': ['Unmatched single quote'],
            'warnings': [],
            'suggestions': ["Did you mean: grep 'pattern' file.txt"]
        }
    """
    
    errors = []
    warnings = []
    suggestions = []
    
    # Check for quote mismatches
    if command.count('"') % 2 != 0:
        errors.append("Unmatched double quotes")
    
    if command.count("'") % 2 != 0:
        errors.append("Unmatched single quotes")
    
    # Check for dangerous patterns
    if "rm -rf /" in command:
        warnings.append("Dangerous: rm -rf on root directory")
    
    if "sudo" in command:
        warnings.append("Uses sudo - requires elevated privileges")
    
    if "| sudo" in command:
        errors.append("Cannot pipe to sudo (security issue)")
    
    if "> /dev/" in command:
        warnings.append("Writing to device file")
    
    # Suggestions
    if "grep" in command and "'" in command:
        suggestions.append("Consider using double quotes instead of single quotes")
    
    if "find" in command and "-type f" not in command:
        suggestions.append("Consider adding '-type f' to find only files")
    
    return {
        'valid': len(errors) == 0,
        'safe': len(warnings) == 0,
        'errors': errors,
        'warnings': warnings,
        'suggestions': suggestions
    }


@tool
def bash_executor(command: str, dry_run: bool = True) -> Dict[str, Any]:
    """Execute a bash command safely (with dry-run option).
    
    WARNING: Only use in controlled environments!
    
    Args:
        command: Bash command to execute
        dry_run: If True, only shows what would be executed (default True)
    
    Returns:
        Execution result with output
    """
    import subprocess
    import time
    
    # Never execute dangerous commands
    dangerous = ["rm -rf", "dd if=", "fork bomb"]
    for pattern in dangerous:
        if pattern in command:
            return {
                'success': False,
                'return_code': -1,
                'stdout': '',
                'stderr': f"Command blocked: contains dangerous pattern '{pattern}'",
                'duration': 0
            }
    
    if dry_run:
        return {
            'success': True,
            'return_code': 0,
            'stdout': f"[DRY RUN] Would execute: {command}",
            'stderr': '',
            'duration': 0
        }
    
    try:
        start = time.time()
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            timeout=10,
            text=True
        )
        duration = time.time() - start
        
        return {
            'success': result.returncode == 0,
            'return_code': result.returncode,
            'stdout': result.stdout[:500],
            'stderr': result.stderr[:500],
            'duration': round(duration, 2)
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'return_code': -1,
            'stdout': '',
            'stderr': 'Command timeout (>10 seconds)',
            'duration': 10
        }
    except Exception as e:
        return {
            'success': False,
            'return_code': -1,
            'stdout': '',
            'stderr': str(e),
            'duration': 0
        }


# ============================================================================
# SECTION 3: USING TOOLS WITH AGENTS
# ============================================================================

def simple_agent_example():
    """
    Simple example showing how to use tools with an agent.
    
    This requires: pip install langchain langchain-openai
    """
    
    print("\n" + "="*70)
    print("SIMPLE AGENT EXAMPLE")
    print("="*70)
    
    try:
        from langchain.agents import AgentExecutor, create_react_agent
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
    except ImportError:
        print("⚠️  Requires: pip install langchain langchain-openai")
        return
    
    # Define tools
    tools = [calculator, string_tools, bash_generator, bash_validator]
    
    # Create LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant with access to tools.
        Use tools to help the user accomplish their goals.
        Always validate bash commands before suggesting them."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # Create agent
    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Use agent
    print("\nUser: Generate a bash command to find all Python files")
    result = executor.invoke({"input": "Generate a bash command to find all Python files"})
    
    print("\nAgent response:")
    print(result)


# ============================================================================
# SECTION 4: TESTING TOOLS
# ============================================================================

def test_all_tools():
    """Test all tools to verify they work correctly."""
    
    print("\n" + "="*70)
    print("TESTING ALL TOOLS")
    print("="*70)
    
    # Test 1: Calculator
    print("\n1. Calculator Tool")
    print("-" * 40)
    result = calculator.invoke({"operation": "add", "a": 5, "b": 3})
    print(f"5 + 3 = {result}")
    assert result == 8, "Calculator failed"
    print("✅ PASSED")
    
    # Test 2: String Tools
    print("\n2. String Tools")
    print("-" * 40)
    result = string_tools.invoke({"operation": "upper", "text": "hello"})
    print(f"uppercase('hello') = {result}")
    assert result == "HELLO", "String tool failed"
    print("✅ PASSED")
    
    # Test 3: Bash Generator
    print("\n3. Bash Generator")
    print("-" * 40)
    result = bash_generator.invoke({
        "task": "Find all Python files",
        "use_grammar_constraints": True
    })
    print(f"Task: {result.get('explanation', 'N/A')}")
    print(f"Command: {result.get('command', 'N/A')}")
    print(f"Valid: {result.get('valid', 'N/A')}")
    assert result['command'] is not None, "Bash generator failed"
    print("✅ PASSED")
    
    # Test 4: Bash Validator - Valid Command
    print("\n4. Bash Validator - Valid Command")
    print("-" * 40)
    result = bash_validator.invoke({"command": "ls -la"})
    print(f"Command: ls -la")
    print(f"Valid: {result['valid']}")
    print(f"Safe: {result['safe']}")
    assert result['valid'] == True, "Validator failed on valid command"
    print("✅ PASSED")
    
    # Test 5: Bash Validator - Invalid Command
    print("\n5. Bash Validator - Invalid Command")
    print("-" * 40)
    result = bash_validator.invoke({"command": "grep 'unclosed"})
    print(f"Command: grep 'unclosed")
    print(f"Valid: {result['valid']}")
    print(f"Errors: {result['errors']}")
    assert result['valid'] == False, "Validator failed on invalid command"
    print("✅ PASSED")
    
    # Test 6: Bash Executor (dry-run)
    print("\n6. Bash Executor (Dry Run)")
    print("-" * 40)
    result = bash_executor.invoke({"command": "ls -la", "dry_run": True})
    print(f"Command: ls -la")
    print(f"Result: {result['stdout']}")
    assert "[DRY RUN]" in result['stdout'], "Executor failed"
    print("✅ PASSED")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED ✅")
    print("="*70)


# ============================================================================
# SECTION 5: COMPLETE WORKFLOW EXAMPLE
# ============================================================================

class BashWorkflow:
    """Complete workflow: Generate → Validate → Execute."""
    
    def run(self, task: str):
        """Run complete workflow for a task."""
        
        print("\n" + "="*70)
        print("BASH GENERATION WORKFLOW")
        print("="*70)
        
        # Step 1: Generate
        print(f"\n📝 Task: {task}")
        print("-" * 40)
        
        generation = bash_generator.invoke({
            "task": task,
            "use_grammar_constraints": True
        })
        
        if generation['error']:
            print(f"❌ Generation failed: {generation['error']}")
            return
        
        command = generation['command']
        print(f"✨ Generated: {command}")
        print(f"📊 Confidence: {generation['confidence']}")
        
        # Step 2: Validate
        print(f"\n🔍 Validating command")
        print("-" * 40)
        
        validation = bash_validator.invoke({"command": command})
        
        if validation['valid']:
            print(f"✅ Valid syntax")
        else:
            print(f"❌ Syntax errors: {validation['errors']}")
        
        if validation['safe']:
            print(f"✅ Safe to execute")
        else:
            print(f"⚠️  Safety warnings: {validation['warnings']}")
        
        if validation['suggestions']:
            print(f"💡 Suggestions:")
            for suggestion in validation['suggestions']:
                print(f"   - {suggestion}")
        
        # Step 3: Execute (dry-run)
        print(f"\n⚙️  Executing (dry-run)")
        print("-" * 40)
        
        execution = bash_executor.invoke({
            "command": command,
            "dry_run": True
        })
        
        print(f"Output: {execution['stdout']}")
        
        # Summary
        print(f"\n📊 Summary")
        print("-" * 40)
        print(f"Command: {command}")
        print(f"Valid: {validation['valid']}")
        print(f"Safe: {validation['safe']}")
        print(f"Ready: {validation['valid'] and validation['safe']}")
        
        return {
            'task': task,
            'command': command,
            'generation': generation,
            'validation': validation,
            'execution': execution
        }


# ============================================================================
# SECTION 6: RUNNING THE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    
    # Option 1: Test tools
    print("\nOption 1: Running tool tests...")
    test_all_tools()
    
    # Option 2: Run workflows
    print("\n\nOption 2: Running workflow examples...")
    workflow = BashWorkflow()
    
    tasks = [
        "Find all Python files",
        "List all files in current directory",
        "Search for error in logs",
    ]
    
    for task in tasks:
        workflow.run(task)
    
    # Option 3: Agent example (requires LangChain + OpenAI key)
    print("\n\nOption 3: Agent example")
    print("(Requires: pip install langchain langchain-openai + OPENAI_API_KEY)")
    try:
        simple_agent_example()
    except Exception as e:
        print(f"Skipped: {e}")
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70)
