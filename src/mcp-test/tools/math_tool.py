def add(a: float, b: float) -> float:
    return a + b

def get_schema():
    return {
        "name": "add",
        "description": "Add two numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "a": { "type": "number", "description": "first number" },
                "b": { "type": "number", "description": "second number" }
            },
            "required": ["a", "b"]
        }
    }
