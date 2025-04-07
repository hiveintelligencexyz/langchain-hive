# example.py
from langchain_hive import HiveSearch
import os

# Initialize the tool
tool = HiveSearch(
    api_key= os.environ["HIVE_INTELLIGENCE_API_KEY"]
)

# Make a simple query
result = tool.invoke({"prompt": "What's the current price of Bitcoin?"})
print(result)


messages = [
    {"role": "user", "content": "Tell me about Uniswap"},
    {"role": "assistant", "content": "Uniswap is a decentralized exchange protocol..."},
    {"role": "user", "content": "What's its trading volume today?"}
]

# Use conversation history with properly formatted messages
result = tool.invoke({
    "messages":messages
})
print(result)

# Example demonstrating all LLM control parameters
print("\n=== Using All LLM Control Parameters ===")
result = tool.invoke({
    "prompt": "Explain the pros and cons of yield farming in DeFi",
    "temperature": 0.2,           # More deterministic response
    "top_p": 0.85,                # Control diversity via nucleus sampling
    "top_k": 40,                  # Limit token selection to top 40 tokens
    "include_data_sources": True  # Include information about data sources used
})
print(f"Response with controlled parameters: {result}")

# Example with different parameter values for comparison
result = tool.invoke({
    "prompt": "Explain the pros and cons of yield farming in DeFi",
    "temperature": 0.8,           # More creative response
    "top_p": 0.95,                # Higher diversity
    "top_k": 50,                  # More token options
    "include_data_sources": False # Don't include data source information
})
print(f"Response with alternative parameters: {result}")

# Mixing parameters with conversation history
result = tool.invoke({
    "messages": messages,
    "temperature": 0.5,
    "top_k": 30,
    "include_data_sources": True
})
print(f"Response with conversation history and parameters: {result}")

