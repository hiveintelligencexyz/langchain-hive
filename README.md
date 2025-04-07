# 🦜️🔗 LangChain Hive

This package provides the LangChain integration with [Hive Intelligence](https://hiveintelligence.xyz/), enabling LLMs to access real-time crypto data via Hive intelligence Search API.

[![PyPI version](https://badge.fury.io/py/langchain-hive.svg)](https://pypi.org/project/langchain-hive/)

## Installation

```bash
pip install langchain-hive
```

## Credentials

You need to set your Hive Intelligence API key. You can get one by signing up at [hiveintelligence.xyz](https://dashboard.hiveintelligence.xyz/).

```python
import getpass
import os

if not os.environ.get("HIVE_INTELLIGENCE_API_KEY"):
    os.environ["HIVE_INTELLIGENCE_API_KEY"] = getpass.getpass("Hive API key:\n")
```

## Hive Search Tool

The HiveSearch tool allows LLMs to query real-time crypto intelligence data, including prices, trading volumes, protocol stats, and more. It supports both stateless queries and multi-turn conversations.

### Instantiation

```python
from langchain_hive import HiveSearch
import os

tool = HiveSearch(api_key=os.environ["HIVE_INTELLIGENCE_API_KEY"])
```

### Invoke with Prompt

The HiveSearch tool accepts a prompt (single-shot) or messages (multi-turn history) and returns a structured response based on the latest on-chain data.

```python
result = tool.invoke({"prompt": "What's the current price of Bitcoin?"})
print(result)
```

### Invoke with Conversation History

Supports multi-turn memory via a list of chat messages:

```python
messages = [
    {"role": "user", "content": "Tell me about Uniswap"},
    {"role": "assistant", "content": "Uniswap is a decentralized exchange protocol..."},
    {"role": "user", "content": "What's its trading volume today?"}
]

result = tool.invoke({"messages": messages})
print(result)
```

### Control Parameters

You can modify the model's behavior with additional parameters:

- `temperature` (float): Controls randomness. Lower is more deterministic.
- `top_p` (float): Controls nucleus sampling diversity.
- `top_k` (int): Limits token sampling to top-k options.
- `include_data_sources` (bool): Whether to return information about the sources used.

```python
result = tool.invoke({
    "prompt": "Explain the pros and cons of yield farming in DeFi",
    "temperature": 0.2,
    "top_p": 0.85,
    "top_k": 40,
    "include_data_sources": True
})
print(result)
```

### Mixed Usage Example

You can also combine messages with control parameters:

```python
result = tool.invoke({
    "messages": messages,
    "temperature": 0.5,
    "top_k": 30,
    "include_data_sources": True
})
print(result)
```

## Agent Integration

Hive Search can be used directly within LangChain Agents for dynamic tool usage.

```python
# !pip install -qU langchain langchain-hive langchain-anthropic

from langchain_hive import HiveSearch
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
import datetime

# Initialize Hive Search Tool
hive_search_tool = HiveSearch(api_key="your_hive_api_key")

# Initialize LLM (Claude)
llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0, api_key="your_anthropic_api_key")

# Setup prompt template
today = datetime.datetime.today().strftime("%D")
prompt = ChatPromptTemplate.from_messages([
    ("system", f"""You are a helpful crypto assistant. Use the hive search tool to answer queries with real-time blockchain data. The date today is {today}."""),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create agent
agent = create_tool_calling_agent(
    llm=llm,
    tools=[hive_search_tool],
    prompt=prompt
)

# Create executor
agent_executor = AgentExecutor(agent=agent, tools=[hive_search_tool], verbose=True)

# Use the agent
user_input = "what is the current price of ETH?"
response = agent_executor.invoke({"messages": [HumanMessage(content=user_input)]})
print(response)
```

## Summary

langchain-hive makes it simple to integrate live, real-time crypto and DeFi insights directly into your LangChain agents or apps via Hive Intelligence.