from tangent import tangent, Agent, run_tangent_loop

###########################
## Define tool functions ##
###########################

# Calculate function

def calculate(operation: str, x: float, y: float) -> str:
    """
    Perform basic arithmetic operations.
    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        x: First number
        y: Second number
    """
    operations = {
        "add": lambda a, b: a + b,
        "subtract": lambda a, b: a - b,
        "multiply": lambda a, b: a * b,
        "divide": lambda a, b: a / b if b != 0 else "Error: Division by zero"
    }
    
    if operation not in operations:
        return f"Error: Unknown operation '{operation}'. Use add, subtract, multiply, or divide."
    
    result = operations[operation](x, y)
    return f"Result of {operation}({x}, {y}) = {result}"

# Greet function

def greet(name: str, language: str = "English") -> str:
    """
    Greet a person in different languages.
    Args:
        name: The name of the person to greet
        language: The language to use (English, Spanish, French)
    """
    greetings = {
        "English": "Hello",
        "Spanish": "Hola",
        "French": "Bonjour"
    }
    
    greeting = greetings.get(language, greetings["English"])
    return f"{greeting}, {name}!"

###########################
## Create the tool agent ##
###########################

tool_agent = Agent(
    name="Math and Greeting Assistant",
    model="gpt-4o",
    instructions="""You are a helpful assistant that can:
1. Perform basic arithmetic operations using the calculate() function
2. Greet users in different languages using the greet() function

When users ask for calculations, use the calculate() function with the appropriate operation.
When users want to be greeted, use the greet() function with their name and preferred language.
""",
    functions=[calculate, greet]
)

if __name__ == "__main__":
    print("\nHello! I'm your Math and Greeting Assistant. I can help you with:")
    print("1. Basic arithmetic (add, subtract, multiply, divide)")
    print("2. Greetings in different languages (English, Spanish, French)\n")
    
    # Run the interactive demo loop
    run_tangent_loop(tool_agent, stream=True, debug=False)
