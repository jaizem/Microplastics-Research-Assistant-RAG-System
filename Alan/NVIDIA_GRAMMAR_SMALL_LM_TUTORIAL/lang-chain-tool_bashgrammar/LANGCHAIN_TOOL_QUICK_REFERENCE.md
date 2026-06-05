# LangChain @tool Decorator: Quick Reference Guide

## Installation

```bash
pip install langchain langchain-core
```

## Basic Usage

### The Simplest Tool

```python
from langchain_core.tools import tool

@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b
```

**That's it!** The `@tool` decorator:
- Converts the function to a LangChain `Tool` object
- Uses the docstring as the description (shown to agents)
- Infers parameter types from type hints
- Makes it available for use by agents

---

## Tool Anatomy

```python
@tool
def my_tool(param1: str, param2: int = 5) -> str:
    """One-line summary of what the tool does.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (optional with default)
    
    Returns:
        What the tool returns
    
    Raises:
        ValueError: When validation fails
    
    Example:
        >>> my_tool("test", 10)
        "result"
    """
    # Implementation
    return f"{param1}_{param2}"
```

**Key elements:**
1. **Name**: Function name becomes tool name
2. **Docstring**: Explains what the tool does (required, shown to agent)
3. **Type hints**: Define parameter types and return type
4. **Implementation**: The actual function code

---

## Complete Examples

### Example 1: Simple Tool

```python
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # In production, call a real weather API
    return f"The weather in {city} is sunny"
```

**Usage:**
```python
result = get_weather.invoke({"city": "New York"})
# Returns: "The weather in New York is sunny"
```

---

### Example 2: Tool with Multiple Parameters

```python
@tool
def calculate(
    operation: str,
    x: float,
    y: float
) -> float:
    """Calculate something with two numbers.
    
    Args:
        operation: 'add', 'subtract', 'multiply', or 'divide'
        x: First number
        y: Second number
    
    Returns:
        Result of the operation
    """
    if operation == "add":
        return x + y
    elif operation == "subtract":
        return x - y
    elif operation == "multiply":
        return x * y
    elif operation == "divide":
        return x / y
    else:
        raise ValueError(f"Unknown operation: {operation}")
```

**Usage:**
```python
result = calculate.invoke({
    "operation": "add",
    "x": 5,
    "y": 3
})
# Returns: 8
```

---

### Example 3: Tool with Optional Parameters

```python
@tool
def search(
    query: str,
    max_results: int = 10,
    language: str = "en"
) -> list:
    """Search for information online.
    
    Args:
        query: What to search for
        max_results: Maximum number of results (default: 10)
        language: Language code (default: "en")
    
    Returns:
        List of search results
    """
    # Implementation
    return [{"title": f"Result {i}", "url": "..."} for i in range(max_results)]
```

**Usage:**
```python
# With all parameters
result = search.invoke({
    "query": "python programming",
    "max_results": 5,
    "language": "en"
})

# With defaults
result = search.invoke({"query": "python programming"})
```

---

### Example 4: Tool with Error Handling

```python
@tool
def divide_numbers(a: float, b: float) -> float:
    """Divide two numbers.
    
    Args:
        a: Numerator
        b: Denominator
    
    Returns:
        a divided by b
    
    Raises:
        ValueError: If b is zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
```

**Usage:**
```python
# Success case
result = divide_numbers.invoke({"a": 10, "b": 2})
# Returns: 5.0

# Error case - agent will see the error message
try:
    result = divide_numbers.invoke({"a": 10, "b": 0})
except ValueError as e:
    print(f"Error: {e}")  # "Cannot divide by zero"
```

---

### Example 5: Tool Returning Complex Objects

```python
from typing import Dict, List

@tool
def analyze_text(text: str) -> Dict[str, any]:
    """Analyze text for various metrics.
    
    Args:
        text: Text to analyze
    
    Returns:
        Dictionary with:
        - 'word_count': Number of words
        - 'char_count': Number of characters
        - 'sentence_count': Number of sentences
        - 'keywords': List of important words
    """
    words = text.split()
    sentences = text.split('.')
    
    return {
        'word_count': len(words),
        'char_count': len(text),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'keywords': [w for w in words if len(w) > 5][:5]
    }
```

**Usage:**
```python
result = analyze_text.invoke({
    "text": "This is a test. Python is great."
})
# Returns:
# {
#     'word_count': 6,
#     'char_count': 34,
#     'sentence_count': 2,
#     'keywords': ['Python', 'great']
# }
```

---

## Using Tools with Agents

### Basic Agent Setup

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Define your tools
tools = [add_numbers, calculate, search]

# Create LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with access to tools."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Create agent
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Use agent
result = executor.invoke({
    "input": "What is 15 plus 27?"
})
```

### How the Agent Uses Tools

1. **Agent receives task**: "What is 15 plus 27?"
2. **Agent reads tool descriptions**: Sees `add_numbers` tool
3. **Agent calls tool**: Uses `add_numbers(15, 27)`
4. **Agent gets result**: 42
5. **Agent returns answer**: "The answer is 42"

---

## Important: Docstring Format

The docstring is **critical** - it's what the agent reads!

### Good Docstring

```python
@tool
def calculate_age(birth_year: int) -> int:
    """Calculate someone's age from their birth year.
    
    Given a person's birth year, returns their current age.
    Assumes the person has already had their birthday this year.
    
    Args:
        birth_year: The year the person was born
    
    Returns:
        Their age in years
    
    Raises:
        ValueError: If birth_year is in the future
    
    Example:
        >>> calculate_age(1990)
        34
    """
    from datetime import datetime
    current_year = datetime.now().year
    if birth_year > current_year:
        raise ValueError("Birth year cannot be in the future")
    return current_year - birth_year
```

### Bad Docstring

```python
@tool
def calculate_age(birth_year: int) -> int:
    """Age calc"""  # Too vague!
    return 2024 - birth_year
```

---

## Debugging Tools

### Test Your Tool

```python
# Before using with an agent, test it directly

# Call the tool
result = my_tool.invoke({"param1": "value1", "param2": 10})
print(result)

# Check tool properties
print(my_tool.name)  # Tool name
print(my_tool.description)  # Docstring (what agent sees)
print(my_tool.args)  # Parameters
```

### View Tool Metadata

```python
import json

# See what the agent sees
print(f"Name: {my_tool.name}")
print(f"Description: {my_tool.description}")
print(f"Args: {json.dumps(my_tool.args, indent=2)}")
```

---

## Common Patterns

### Pattern 1: Validation

```python
@tool
def process_file(filepath: str) -> str:
    """Process a file."""
    if not filepath.endswith('.txt'):
        raise ValueError("Only .txt files supported")
    
    with open(filepath, 'r') as f:
        return f.read()
```

### Pattern 2: External API Call

```python
@tool
def get_stock_price(symbol: str) -> float:
    """Get current stock price for a symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., "AAPL")
    
    Returns:
        Current price in USD
    """
    import requests
    
    response = requests.get(f"https://api.example.com/stock/{symbol}")
    data = response.json()
    return data['price']
```

### Pattern 3: Data Transformation

```python
@tool
def format_data(data: str, format_type: str) -> str:
    """Format data as JSON, CSV, or XML.
    
    Args:
        data: Raw data string
        format_type: "json", "csv", or "xml"
    
    Returns:
        Formatted data
    """
    if format_type == "json":
        import json
        return json.dumps({"data": data})
    elif format_type == "csv":
        return f'"data","{data}"'
    # ... etc
```

### Pattern 4: Tool Composition

```python
@tool
def generate_and_validate(task: str) -> dict:
    """Generate something and validate it.
    
    Uses multiple tools internally.
    """
    # Call another tool internally
    generated = generate_solution(task)
    validation = validate_solution(generated)
    
    return {
        'generated': generated,
        'validation': validation
    }
```

---

## Best Practices

### ✅ DO:

- **Write clear docstrings** - Agents read these!
- **Use type hints** - Helps agents understand parameters
- **Raise informative errors** - Agent learns from failures
- **Keep tools focused** - One tool, one job
- **Test before deploying** - Test tools independently first
- **Validate inputs** - Check parameters early
- **Return structured data** - Use dicts/lists, not just strings

### ❌ DON'T:

- **Skip docstrings** - Agent won't understand tool
- **Use vague parameter names** - `x`, `y` are bad; use `source_file`, `destination_path`
- **Make tools do too much** - One tool should do one thing
- **Silently fail** - Always raise exceptions with clear messages
- **Return huge amounts of data** - Summarize/truncate results
- **Call external APIs without timeouts** - Add timeout handling

---

## Bash Generation Tool Example

Here's a complete example for bash generation:

```python
@tool
def generate_bash_command(task: str, use_constraints: bool = True) -> dict:
    """Generate a bash command for a task using grammar constraints.
    
    Uses constrained decoding to ensure the generated command is:
    - Syntactically valid
    - Properly formatted
    - Likely to work first time
    
    Args:
        task: Natural language description of what to do
              (e.g., "Find all Python files modified today")
        use_constraints: Whether to apply grammar constraints (default True)
    
    Returns:
        Dictionary with:
        - 'command': The generated bash command
        - 'valid': Whether it passed validation
        - 'confidence': Confidence score (0.0-1.0)
        - 'explanation': What the command does
    
    Example:
        >>> generate_bash_command("Find all Python files")
        {
            'command': 'find . -name "*.py"',
            'valid': True,
            'confidence': 0.95,
            'explanation': 'Recursively find all .py files in current directory'
        }
    """
    # In production, would call model with optional grammar constraints
    
    # Demo implementation
    if "python" in task.lower() and "find" in task.lower():
        return {
            'command': 'find . -name "*.py"',
            'valid': True,
            'confidence': 0.95 if use_constraints else 0.75,
            'explanation': 'Recursively finds all Python files'
        }
    
    return {
        'command': 'echo "Command generation not implemented for this task"',
        'valid': False,
        'confidence': 0.0,
        'explanation': 'No matching pattern found'
    }
```

---

## Testing Your Tools

```python
# test_tools.py

def test_add_numbers():
    result = add_numbers.invoke({"a": 5, "b": 3})
    assert result == 8, f"Expected 8, got {result}"
    print("✅ test_add_numbers passed")

def test_divide_by_zero():
    try:
        divide_numbers.invoke({"a": 10, "b": 0})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "zero" in str(e)
        print("✅ test_divide_by_zero passed")

if __name__ == "__main__":
    test_add_numbers()
    test_divide_by_zero()
    print("\nAll tests passed! ✅")
```

---

## Next Steps

1. **Create your first tool** - Pick a simple function and add `@tool`
2. **Test it** - Call `.invoke()` with test data
3. **Add to agent** - Include in tools list for agent
4. **Iterate** - Improve docstrings and error handling based on agent behavior

---

## Resources

- **LangChain Docs**: https://python.langchain.com/
- **Tool Reference**: https://python.langchain.com/docs/modules/tools/
- **Agent Types**: https://python.langchain.com/docs/modules/agents/

---

**Happy tool building! 🔧**
