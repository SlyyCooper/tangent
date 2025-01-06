Let me break this down very clearly by comparing BOTH the function call part AND the output part:

**Part 1: FUNCTION CALL** (How the function is called)
```python
# Normal Function Call
result = search_web("Python tutorials")

# Structured Output Function Call
result = search_web("Python tutorials")   # EXACTLY THE SAME!
```
👉 The function call part is IDENTICAL - there's no difference in how you call the function.

**Part 2: FUNCTION DEFINITION** (How the function is written)
```python
# Normal Function
def search_web(query: str) -> str:
    return "Results for: " + query

# Structured Output Function
def search_web(query: str) -> Result:  # Different return type
    return Result(                     # Different return structure
        value="Results for: " + query,
        extracted_data={"query": query}
    )
```

**Part 3: OUTPUT HANDLING** (How you use the result)
```python
# Normal Function
result = search_web("Python")
print(result)  # Just prints the string directly

# Structured Output Function
result = search_web("Python")
print(result.value)              # Get main result
print(result.extracted_data)  # Get extra metadata
```

So to directly answer your question:
1. The function CALL is exactly the same
2. The function DEFINITION is different (return type and structure)
3. The OUTPUT HANDLING is different (because it's structured)

The main difference is in how the function is written and how you handle its output, NOT in how you call it. The structured output is just a standardized way to package the return value with additional metadata.

Think of it like this:
- Normal function: Returns a letter
- Structured output function: Returns an envelope containing:
  - The letter (value)
  - Sender info (metadata)
  - Routing instructions (agent info)
  
But you still "mail" (call) both functions the same way!
