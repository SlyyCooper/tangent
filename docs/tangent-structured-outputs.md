# Creating Structured Output Functions in Tangent

This guide explains how to create and use structured output functions with tangent agents, allowing for rich, typed responses and complex data handling.

## Overview

Tangent provides a `Structured_Result` class that enables functions to return:
- A string overview of the result
- Updated context variables
- A new agent (for agent transfers)

## 1. Basic Structure

```python
from tangent import Agent, tangent, Structured_Result

# 1. Import the necessary components
from tangent import Agent, tangent, Structured_Result

# 2. Define your function with structured output
def get_user_info(user_id: str) -> Structured_Result:
    """
    Get information about a user and store it in context.
    """
    # Simulate getting user data
    user_data = {
        "name": "John Doe",
        "email": "john@example.com",
        "preferences": {"theme": "dark"}
    }
    
    return Structured_Result(
        result_overview=f"Found user: {user_data['name']}",  # What the agent sees
        extracted_data={"current_user": user_data}  # Stored in conversation context
    )

# 3. Create an agent with the function
agent = Agent(
    name="UserInfoAgent",
    model="gpt-4o",  # or "claude-3-5-sonnet-20241022"
    instructions="You help retrieve user information.",
    functions=[get_user_info]
)

# 4. Use the agent
client = tangent()
response = client.run(
    agent=agent,
    messages=[{"role": "user", "content": "Get info for user 123"}]
)
```

## 2. The `Structured_Result` Class

The `Structured_Result` class has three main fields:

```python
class Structured_Result(BaseModel):
    result_overview: str = ""          # What the agent sees in the conversation
    agent: Optional['Agent'] = None    # For agent transfers
    extracted_data: dict = {}          # Context variables
```

## 3. Common Use Cases

### 3.1 Basic Result with Context

```python
def fetch_weather(location: str) -> Structured_Result:
    """Get weather and store location in context."""
    weather = "sunny"  # Simulate API call
    return Structured_Result(
        result_overview=f"The weather in {location} is {weather}",
        extracted_data={"last_location": location}
    )
```

### 3.2 Agent Transfer

```python
def transfer_to_specialist(topic: str) -> Structured_Result:
    """Transfer to a specialist agent based on topic."""
    specialist = Agent(
        name="Specialist",
        model="gpt-4o",
        instructions=f"You are a specialist in {topic}."
    )
    
    return Structured_Result(
        result_overview=f"Transferring to {topic} specialist",
        agent=specialist,
        extracted_data={"specialty": topic}
    )
```

### 3.3 Complex Data Processing

```python
def analyze_data(dataset: str) -> Structured_Result:
    """Analyze data and store results in context."""
    # Simulate analysis
    stats = {
        "mean": 42,
        "median": 40,
        "mode": 45
    }
    
    return Structured_Result(
        result_overview="Data analysis complete. Mean: 42, Median: 40, Mode: 45",
        extracted_data={
            "analysis_results": stats,
            "dataset_name": dataset
        }
    )
```

## 4. Best Practices

1. **Clear Result Overviews**
   ```python
   # Good
   result_overview="Found user John Doe (ID: 123)"
   
   # Bad
   result_overview=str(user_data)  # Raw data dump
   ```

2. **Structured Context Variables**
   ```python
   # Good
   extracted_data={
       "user": {"id": 123, "name": "John"},
       "session": {"start_time": "2024-03-21"}
   }
   
   # Bad
   extracted_data={"raw_data": "id=123;name=John"}
   ```

3. **Type Hints**
   ```python
   def process_order(order_id: str) -> Structured_Result:
       """Always use type hints for clarity."""
       pass
   ```

## 5. Advanced Usage

### 5.1 Chaining Results

```python
def process_user_request(request_id: str) -> Structured_Result:
    """Process a request with multiple steps."""
    # Step 1: Validate
    validation = validate_request(request_id)
    
    # Step 2: Process
    result = process_request(request_id)
    
    # Combine results
    return Structured_Result(
        result_overview=f"Request {request_id} processed: {result}",
        extracted_data={
            "validation": validation,
            "processing": result,
            "request_id": request_id
        }
    )
```

### 5.2 Error Handling

```python
def safe_api_call(endpoint: str) -> Structured_Result:
    """Handle API calls safely."""
    try:
        result = call_api(endpoint)
        return Structured_Result(
            result_overview="API call successful",
            extracted_data={"api_result": result}
        )
    except Exception as e:
        return Structured_Result(
            result_overview=f"Error calling API: {str(e)}",
            extracted_data={"error": str(e)}
        )
```

### 5.3 Conditional Agent Transfer

```python
def smart_transfer(query: str) -> Structured_Result:
    """Transfer based on query complexity."""
    complexity = analyze_complexity(query)
    
    if complexity > 0.8:
        expert = Agent(
            name="Expert",
            model="gpt-4o",
            instructions="Handle complex queries."
        )
        return Structured_Result(
            result_overview="Query requires expert handling",
            agent=expert,
            extracted_data={"complexity": complexity}
        )
    
    return Structured_Result(
        result_overview="Processing with current agent",
        extracted_data={"complexity": complexity}
    )
```

## 6. Using with Different Models

The `Structured_Result` format works seamlessly with both OpenAI and Anthropic models:

```python
# OpenAI Agent
openai_agent = Agent(
    name="OpenAIAgent",
    model="gpt-4o",
    functions=[your_structured_function]
)

# Anthropic Agent
anthropic_agent = Agent(
    name="AnthropicAgent",
    model="claude-3-5-sonnet-20241022",
    functions=[your_structured_function]
)
```

The tangent library automatically handles the conversion between different model formats while maintaining the structured output functionality.
