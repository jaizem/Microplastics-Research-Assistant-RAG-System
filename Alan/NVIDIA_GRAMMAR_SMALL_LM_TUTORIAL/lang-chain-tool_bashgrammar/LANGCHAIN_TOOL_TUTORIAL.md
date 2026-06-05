"""
LangChain @tool Decorator Tutorial: Complete Implementation Guide

This tutorial demonstrates how to create and use LangChain tools with the @tool
decorator, including detailed examples for bash command generation with
grammar constraints.

Topics covered:
1. Basic @tool decorator usage
2. Tool parameters and return types
3. Error handling and validation
4. Integration with agents
5. Bash generation tool example
6. Advanced patterns and best practices
"""

# ============================================================================
# PART 1: BASIC @TOOL DECORATOR USAGE
# ============================================================================

"""
The @tool decorator from langchain_core.tools makes it easy to convert any
Python function into a LangChain tool that can be used by agents and language
models.

Why use @tool?
- Agents can call tools automatically based on task requirements
- Provides clear descriptions of what each tool does
- Handles parameter validation
- Integrates seamlessly with LangChain's agent framework
"""

from langchain_core.tools import tool
from typing import Optional, List, Dict, Any
import json


# EXAMPLE 1: Simplest Possible Tool
# ============================================================================

@tool
def get_current_time() -> str:
    """Get the current time in UTC."""
    from datetime import datetime
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


# How it works:
# - Decorator converts the function to a Tool object
# - Docstring becomes the tool description (shown to agents)
# - Return type (str) tells the agent what to expect
# - Function name becomes the tool name ("get_current_time")


# EXAMPLE 2: Tool with Parameters
# ============================================================================

@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        The sum of a and b
    """
    return a + b


# How it works:
# - Parameters become tool inputs (a, b)
# - Type hints define parameter types (float, float)
# - Agent will pass arguments automatically
# - Docstring explains each parameter


# EXAMPLE 3: Tool with Optional Parameters
# ============================================================================

@tool
def search_web(query: str, max_results: Optional[int] = 5) -> List[Dict[str, str]]:
    """Search the web for information.
    
    Args:
        query: What to search for
        max_results: Maximum number of results to return (default: 5)
    
    Returns:
        List of search results with 'title' and 'url' keys
    """
    # In production, this would call a real search API
    # For demo, return mock results
    return [
        {"title": f"Result {i+1} for '{query}'", "url": f"https://example.com/{i}"}
        for i in range(max_results)
    ]


# How it works:
# - Optional parameters show up in agent's tool description
# - Agent can decide whether to provide them or use defaults
# - Type hints help agent understand what's needed


# ============================================================================
# PART 2: UNDERSTANDING TOOL DESCRIPTIONS
# ============================================================================

"""
The docstring is critical - it's what the agent reads to understand the tool.

Good docstring:
- One-line summary (what the tool does)
- Args section (parameter descriptions)
- Returns section (what the tool returns)
- Examples (optional, helpful for agents)

Format: Use Google-style docstrings for clarity
"""

@tool
def parse_bash_command(command: str) -> Dict[str, Any]:
    """Parse a bash command into its components.
    
    Extracts the command name, flags, and arguments from a bash command string.
    Useful for understanding and validating shell commands.
    
    Args:
        command: A bash command string (e.g., "grep -i pattern file.txt")
    
    Returns:
        Dictionary with keys:
        - 'command_name': Name of the command (e.g., "grep")
        - 'flags': List of flags used (e.g., ["-i"])
        - 'arguments': List of arguments (e.g., ["pattern", "file.txt"])
        - 'is_valid': Whether the command appears valid (bool)
    
    Raises:
        ValueError: If command string is empty or malformed
    
    Example:
        >>> parse_bash_command("grep -i pattern file.txt")
        {
            'command_name': 'grep',
            'flags': ['-i'],
            'arguments': ['pattern', 'file.txt'],
            'is_valid': True
        }
    """
    if not command or not command.strip():
        raise ValueError("Command cannot be empty")
    
    tokens = command.strip().split()
    result = {
        'command_name': tokens[0] if tokens else None,
        'flags': [t for t in tokens[1:] if t.startswith('-')],
        'arguments': [t for t in tokens[1:] if not t.startswith('-')],
        'is_valid': len(tokens) > 0
    }
    return result


# ============================================================================
# PART 3: ERROR HANDLING IN TOOLS
# ============================================================================

"""
Tools should handle errors gracefully. When a tool raises an exception,
the agent receives the error message and can decide what to do next.

Best practices:
1. Raise informative exceptions with clear messages
2. Validate inputs at the start
3. Include context about what went wrong
4. Let the agent retry with different parameters
"""

@tool
def validate_bash_command(command: str) -> Dict[str, Any]:
    """Validate a bash command for syntax and safety.
    
    Checks if a command:
    - Has valid syntax
    - Doesn't contain dangerous patterns
    - Uses appropriate quoting
    
    Args:
        command: Bash command to validate
    
    Returns:
        Dictionary with:
        - 'valid': Whether command is syntactically valid (bool)
        - 'safe': Whether command appears safe (bool)
        - 'errors': List of error messages
        - 'warnings': List of warnings
    
    Raises:
        ValueError: If command is empty or obviously malformed
        RuntimeError: If validation system is unavailable
    """
    if not command or not isinstance(command, str):
        raise ValueError(f"Command must be a non-empty string, got: {type(command)}")
    
    errors = []
    warnings = []
    
    # Check for basic syntax issues
    if command.count('"') % 2 != 0:
        errors.append("Unmatched double quotes")
    
    if command.count("'") % 2 != 0:
        errors.append("Unmatched single quotes")
    
    # Check for dangerous patterns
    dangerous_patterns = [
        ('rm -rf /', 'Using rm -rf on root directory'),
        ('> /dev/sda', 'Writing to block device'),
    ]
    
    for pattern, warning in dangerous_patterns:
        if pattern in command:
            warnings.append(f"⚠️  {warning}")
    
    return {
        'valid': len(errors) == 0,
        'safe': len(warnings) == 0,
        'errors': errors,
        'warnings': warnings
    }


# ============================================================================
# PART 4: STATEFUL TOOLS (WITH CONTEXT)
# ============================================================================

"""
Tools can maintain state and context. This is useful when you want a tool
to remember previous interactions or maintain configuration.
"""

class BashCommandValidator:
    """Tool for validating bash commands with memory of previous validations."""
    
    def __init__(self):
        self.validation_history = []
        self.known_safe_commands = set()
        self.blacklist = set()
    
    def add_to_whitelist(self, command: str):
        """Mark a command as known-safe."""
        self.known_safe_commands.add(command)
    
    def add_to_blacklist(self, command: str):
        """Mark a command as forbidden."""
        self.blacklist.add(command)
    
    @tool
    def validate_command(self, command: str) -> Dict[str, Any]:
        """Validate bash command with memory of previous decisions.
        
        Args:
            command: Bash command to validate
        
        Returns:
            Dictionary with validation result and reasoning
        """
        # Check whitelist
        if command in self.known_safe_commands:
            return {
                'valid': True,
                'reason': 'Command is in whitelist',
                'cached': True
            }
        
        # Check blacklist
        if command in self.blacklist:
            return {
                'valid': False,
                'reason': 'Command is blacklisted',
                'cached': True
            }
        
        # Validate (simplified)
        result = {
            'valid': not any(p in command for p in ['rm -rf', '| sudo']),
            'reason': 'Passed validation checks',
            'cached': False
        }
        
        # Store in history
        self.validation_history.append({
            'command': command,
            'result': result,
            'timestamp': __import__('datetime').datetime.now().isoformat()
        })
        
        return result


# ============================================================================
# PART 5: GRAMMAR-CONSTRAINED BASH TOOL (MAIN EXAMPLE)
# ============================================================================

"""
This is the primary example: a tool that generates bash commands with
grammar constraints, exactly matching the tutorial's focus.
"""

@tool
def generate_constrained_bash(
    task_description: str,
    max_tokens: int = 100,
    use_constraints: bool = True
) -> Dict[str, Any]:
    """Generate a bash command for a given task with optional grammar constraints.
    
    Uses a small language model to generate bash commands that are:
    - Syntactically valid (when constraints enabled)
    - Appropriately flagged and structured
    - Ready for execution
    
    This tool demonstrates grammar-constrained decoding: applying formal grammar
    rules during token generation to ensure output validity.
    
    Args:
        task_description: Natural language description of what to do
                         (e.g., "Find all Python files in current directory")
        max_tokens: Maximum length of generated command (default: 100)
        use_constraints: Whether to apply grammar constraints (default: True)
    
    Returns:
        Dictionary containing:
        - 'command': Generated bash command string
        - 'valid': Whether command passed syntax validation
        - 'constrained': Whether constraints were applied
        - 'explanation': Human-readable explanation of the command
        - 'confidence': Confidence score (0.0 to 1.0)
        - 'error': Error message if generation failed (or None)
    
    Raises:
        ValueError: If task_description is empty or exceeds limits
        RuntimeError: If the inference engine is unavailable
    
    Example:
        >>> generate_constrained_bash("Find all .txt files")
        {
            'command': 'find . -name "*.txt"',
            'valid': True,
            'constrained': True,
            'explanation': 'Uses find to search current directory recursively for .txt files',
            'confidence': 0.95,
            'error': None
        }
    """
    # Input validation
    if not task_description or not isinstance(task_description, str):
        raise ValueError("task_description must be a non-empty string")
    
    if len(task_description) > 500:
        raise ValueError("task_description too long (max 500 chars)")
    
    if max_tokens < 10 or max_tokens > 500:
        raise ValueError("max_tokens must be between 10 and 500")
    
    # In production, this would:
    # 1. Create model prompt from task_description
    # 2. Initialize grammar if use_constraints=True
    # 3. Run inference with token masking
    # 4. Validate output with tree-sitter-bash
    # 5. Return detailed result
    
    # For demo, simulate the result
    command_map = {
        "find": "find . -name '*.txt'",
        "list files": "ls -la",
        "grep": "grep -r 'pattern' .",
        "count": "wc -l < file.txt",
    }
    
    # Find matching command
    command = None
    for key, val in command_map.items():
        if key.lower() in task_description.lower():
            command = val
            break
    
    command = command or "echo 'Command not understood'"
    
    return {
        'command': command,
        'valid': True,
        'constrained': use_constraints,
        'explanation': f"Generated command for task: {task_description}",
        'confidence': 0.85 if use_constraints else 0.65,
        'error': None
    }


# ============================================================================
# PART 6: USING TOOLS WITH AGENTS
# ============================================================================

"""
Tools come alive when used with LangChain agents. Agents decide which tools
to use based on the task at hand.
"""

def example_agent_usage():
    """
    Example of using tools with a LangChain agent.
    
    This shows how tools are bound to agents and how agents use them.
    """
    from langchain_core.tools import Tool
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI  # or your LLM of choice
    
    # Define tools
    tools = [
        get_current_time,
        add_numbers,
        parse_bash_command,
        validate_bash_command,
        generate_constrained_bash,
    ]
    
    # Create LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # Create agent prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that can:
        1. Generate bash commands for tasks
        2. Validate bash commands for safety
        3. Parse and understand bash commands
        4. Answer questions about the current time
        
        Use tools to help the user accomplish their goals.
        Always validate generated commands before suggesting them."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # Create agent
    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Use agent
    result = executor.invoke({
        "input": "Generate a bash command to find all Python files in the current directory"
    })
    
    return result


# ============================================================================
# PART 7: ADVANCED PATTERNS
# ============================================================================

"""
Advanced patterns for creating sophisticated tools.
"""

# Pattern 1: Tool with Custom Formatting
# ============================================================================

@tool
def format_bash_command_documentation(command_name: str) -> str:
    """Generate formatted documentation for a bash command.
    
    Creates human-readable documentation including:
    - Brief description
    - Common flags
    - Examples
    - Safety notes
    
    Args:
        command_name: Name of the bash command (e.g., "grep", "find")
    
    Returns:
        Formatted documentation string
    """
    documentation = {
        'grep': """
GREP - Search for patterns in text
Usage: grep [OPTIONS] PATTERN [FILE]

Common options:
  -i    Ignore case distinctions
  -r    Recursively search directories
  -n    Print line numbers
  -l    Print only filenames
  -v    Invert match (show non-matching lines)

Examples:
  grep "error" logfile.txt
  grep -r "TODO" . --include="*.py"
  grep -i "warning" logfile.txt | head -10

Safety note: Be careful with large files; consider using flags to limit output.
""",
        'find': """
FIND - Search for files in directory tree
Usage: find [PATH] [OPTIONS] [EXPRESSION]

Common options:
  -name PATTERN      Match filename
  -type f|d|l        File type (f=file, d=directory, l=link)
  -size +N           File larger than N bytes
  -mtime -N          Modified in last N days

Examples:
  find . -name "*.txt"
  find /home -type f -size +1M
  find . -name "test_*.py" -mtime -7

Safety note: Use -path to exclude directories when searching large trees.
"""
    }
    
    return documentation.get(command_name, f"No documentation for {command_name}")


# Pattern 2: Tool Composition (One Tool Calls Another)
# ============================================================================

@tool
def generate_and_validate_bash(task: str) -> Dict[str, Any]:
    """Generate a bash command and immediately validate it.
    
    Combines generation and validation in one atomic operation.
    
    Args:
        task: Task description
    
    Returns:
        Generated command with validation results
    """
    # Generate command
    generation_result = generate_constrained_bash(task)
    
    if generation_result['error']:
        return generation_result
    
    command = generation_result['command']
    
    # Validate it
    validation_result = validate_bash_command(command)
    
    # Return combined result
    return {
        'command': command,
        'generation_valid': generation_result['valid'],
        'validation_result': validation_result,
        'ready_to_execute': (generation_result['valid'] and validation_result['valid']),
        'confidence': generation_result['confidence']
    }


# Pattern 3: Tool with Progress/Streaming Output
# ============================================================================

@tool
def execute_bash_command_safe(command: str, timeout: int = 30) -> Dict[str, Any]:
    """Safely execute a bash command with timeout and output capture.
    
    Important: This is a simulation. In production, use proper sandboxing.
    
    Args:
        command: Bash command to execute
        timeout: Maximum execution time in seconds (default: 30)
    
    Returns:
        Dictionary with:
        - 'return_code': Exit code (0 = success)
        - 'stdout': Command output
        - 'stderr': Error output
        - 'duration_seconds': How long execution took
    """
    import subprocess
    import time
    
    try:
        start = time.time()
        
        # In production, use proper sandboxing:
        # - Docker container
        # - Restricted permissions
        # - Isolated filesystem
        # - Network isolation
        
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            timeout=timeout,
            text=True
        )
        
        duration = time.time() - start
        
        return {
            'return_code': result.returncode,
            'stdout': result.stdout[:1000],  # Limit output
            'stderr': result.stderr[:1000],
            'duration_seconds': round(duration, 2),
            'success': result.returncode == 0,
        }
    
    except subprocess.TimeoutExpired:
        return {
            'return_code': -1,
            'stdout': '',
            'stderr': f'Command timeout after {timeout} seconds',
            'duration_seconds': timeout,
            'success': False,
        }
    
    except Exception as e:
        return {
            'return_code': -1,
            'stdout': '',
            'stderr': str(e),
            'duration_seconds': 0,
            'success': False,
        }


# ============================================================================
# PART 8: TESTING AND DEBUGGING TOOLS
# ============================================================================

"""
How to test and debug your tools before using them with agents.
"""

def test_tools():
    """Test all tools to ensure they work correctly."""
    
    print("=" * 70)
    print("TESTING LANGCHAIN TOOLS")
    print("=" * 70)
    
    # Test 1: Simple tool
    print("\n1. Testing get_current_time():")
    result = get_current_time.invoke({})
    print(f"   Result: {result}")
    
    # Test 2: Tool with parameters
    print("\n2. Testing add_numbers(5, 3):")
    result = add_numbers.invoke({"a": 5, "b": 3})
    print(f"   Result: {result}")
    
    # Test 3: Tool with complex return type
    print("\n3. Testing parse_bash_command('grep -i pattern file.txt'):")
    result = parse_bash_command.invoke({"command": "grep -i pattern file.txt"})
    print(f"   Result: {json.dumps(result, indent=2)}")
    
    # Test 4: Error handling
    print("\n4. Testing error handling in validate_bash_command():")
    try:
        result = validate_bash_command.invoke({"command": "grep 'unclosed quote file.txt"})
        print(f"   Result: {json.dumps(result, indent=2)}")
    except ValueError as e:
        print(f"   Caught error (expected): {e}")
    
    # Test 5: Main example
    print("\n5. Testing generate_constrained_bash():")
    result = generate_constrained_bash.invoke({
        "task_description": "Find all Python files",
        "use_constraints": True
    })
    print(f"   Result: {json.dumps(result, indent=2)}")
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


# ============================================================================
# PART 9: TOOL METADATA AND INTROSPECTION
# ============================================================================

"""
LangChain tools have metadata you can inspect.
"""

def inspect_tool(tool):
    """Inspect and print tool metadata."""
    print(f"Tool Name: {tool.name}")
    print(f"Description: {tool.description}")
    print(f"Args: {tool.args}")
    print(f"Return type: {tool.return_direct}")


# ============================================================================
# PART 10: COMPLETE EXAMPLE - BASH GENERATION AGENT
# ============================================================================

"""
A complete example showing how to build an agent that generates and validates
bash commands using grammar constraints.
"""

class BashGenerationAgent:
    """
    Complete agent for generating and validating bash commands.
    
    Demonstrates:
    - Multiple specialized tools
    - Tool composition
    - Error handling
    - Agent loop
    """
    
    def __init__(self, llm=None):
        """
        Initialize the agent.
        
        Args:
            llm: LangChain LLM instance (defaults to OpenAI GPT-4)
        """
        if llm is None:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4", temperature=0)
        
        self.llm = llm
        self.tools = [
            generate_constrained_bash,
            validate_bash_command,
            parse_bash_command,
            format_bash_command_documentation,
        ]
    
    def generate_command(self, task: str) -> Dict[str, Any]:
        """
        Generate a bash command for a task.
        
        Args:
            task: Natural language task description
        
        Returns:
            Generated command with metadata
        """
        result = generate_constrained_bash.invoke({
            "task_description": task,
            "use_constraints": True
        })
        return result
    
    def validate_and_explain(self, command: str) -> Dict[str, Any]:
        """
        Validate a command and provide explanation.
        
        Args:
            command: Bash command to validate
        
        Returns:
            Validation result with explanation
        """
        validation = validate_bash_command.invoke({"command": command})
        parsed = parse_bash_command.invoke({"command": command})
        
        return {
            'validation': validation,
            'parsing': parsed,
            'ready_to_execute': validation['valid'] and validation['safe']
        }
    
    def generate_full_workflow(self, task: str):
        """
        Full workflow: generate, validate, and explain.
        
        Args:
            task: Task description
        
        Returns:
            Complete workflow result
        """
        # Step 1: Generate
        print(f"\n📝 Task: {task}")
        generation = self.generate_command(task)
        print(f"✨ Generated: {generation['command']}")
        
        # Step 2: Validate
        print(f"🔍 Validating...")
        validation = self.validate_and_explain(generation['command'])
        
        # Step 3: Report
        print(f"✅ Valid: {validation['ready_to_execute']}")
        if not validation['ready_to_execute']:
            print(f"⚠️  Issues: {validation['validation']['errors']}")
        
        return {
            'task': task,
            'command': generation['command'],
            'validation': validation,
            'ready': validation['ready_to_execute']
        }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Example 1: Test tools
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Testing Individual Tools")
    print("=" * 70)
    test_tools()
    
    # Example 2: Inspect tool metadata
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Inspecting Tool Metadata")
    print("=" * 70)
    inspect_tool(generate_constrained_bash)
    
    # Example 3: Use the agent workflow
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Complete Agent Workflow")
    print("=" * 70)
    agent = BashGenerationAgent()
    
    tasks = [
        "Find all Python files in the current directory",
        "Count lines in a file",
        "Search for error messages in logs",
    ]
    
    for task in tasks:
        agent.generate_full_workflow(task)
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
