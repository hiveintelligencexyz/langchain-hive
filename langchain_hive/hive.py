"""Hive Intelligence tools."""

from typing import Any, Dict, List, Optional, Type, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

class HiveSearchMessage(BaseModel):
    """Message format for Hive Search API"""
    role: str = Field(description="Role of the message sender - 'user' or 'assistant'")
    content: str = Field(description="Content of the message")

class HiveSearchInput(BaseModel):
    """Input for HiveSearch"""
    prompt: Optional[str] = Field(
        default=None,
        description="The query text to be processed (required if messages not provided)"
    )
    messages: Optional[List[HiveSearchMessage]] = Field(
        default=None,
        description="Conversation history in chat format (required if prompt not provided)"
    )
    temperature: Optional[float] = Field(
        default=0.7,
        description="Controls randomness in generation (0.0 to 1.0)"
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Limits token selection to top K choices"
    )
    top_p: Optional[float] = Field(
        default=None,
        description="Nucleus sampling parameter"
    )
    include_data_sources: Optional[bool] = Field(
        default=True,
        description="Include source information in response"
    )
    wallet: Optional[str] = Field(
        default=None,
        description="User wallet address for personalized results"
    )

class HiveSearchAPIWrapper:
    """Wrapper for Hive Search API."""

    def __init__(self, api_key: str, base_url: str = "https://api.hiveintelligence.xyz"):
        """Initialize with API key."""
        self.api_key = api_key
        self.base_url = base_url
        
        # Check if api_key is provided
        if not self.api_key:
            raise ValueError("Hive Intelligence API key is required")

    def _validate_request(self, params: Dict[str, Any]) -> None:
        """Validate request parameters."""
        # Check that either prompt or messages is provided
        if not params.get("prompt") and not params.get("messages"):
            raise ValueError("Either 'prompt' or 'messages' must be provided")
            
        # If messages is provided, validate format
        if params.get("messages"):
            for msg in params["messages"]:
                if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                    raise ValueError("Messages must be in the format: {'role': 'user'|'assistant', 'content': 'text'}")
                if msg["role"] not in ["user", "assistant"]:
                    raise ValueError("Message role must be either 'user' or 'assistant'")

    def process_query(
        self, 
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = 0.7,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        include_data_sources: Optional[bool] = True,
        wallet: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process a query using the Hive Intelligence API."""
        import requests
        import json
        
        # Prepare request parameters
        params = {
            "prompt": prompt,
            "messages": [msg.dict() if not isinstance(msg, dict) else msg for msg in messages] if messages else None,
            "temperature": temperature,
            "include_data_sources": include_data_sources,
        }
        
        # Add optional parameters if provided
        if top_k is not None:
            params["top_k"] = top_k
        if top_p is not None:
            params["top_p"] = top_p
        if wallet is not None:
            params["wallet"] = wallet
            
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        # Validate request parameters
        self._validate_request(params)
        
        # Prepare headers
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Make request
        try:
            response = requests.post(
                f"{self.base_url}/v1/search",
                headers=headers,
                data=json.dumps(params)
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            return result
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                raise ToolException("Bad Request: Invalid parameters")
            elif e.response.status_code == 401:
                raise ToolException("Unauthorized: Missing or insufficient credits")
            elif e.response.status_code == 403:
                raise ToolException("Forbidden: Invalid API key")
            elif e.response.status_code == 429:
                error_data = e.response.json()
                refresh_time = error_data.get("refreshTimestampUTC", "unknown time")
                raise ToolException(f"Rate limit exceeded. Try again after {refresh_time}")
            else:
                raise ToolException(f"HTTP Error: {e}")
        except requests.exceptions.RequestException as e:
            raise ToolException(f"Request failed: {e}")
        except json.JSONDecodeError:
            raise ToolException("Failed to parse API response")
        except Exception as e:
            raise ToolException(f"Unexpected error: {e}")
    
    async def aprocess_query(
        self, 
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = 0.7,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        include_data_sources: Optional[bool] = True,
        wallet: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process a query using the Hive Intelligence API asynchronously."""
        import aiohttp
        import json
        import asyncio
        
        # Prepare request parameters
        params = {
            "prompt": prompt,
            "messages": messages,
            "temperature": temperature,
            "include_data_sources": include_data_sources,
        }
        
        # Add optional parameters if provided
        if top_k is not None:
            params["top_k"] = top_k
        if top_p is not None:
            params["top_p"] = top_p
        if wallet is not None:
            params["wallet"] = wallet
            
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        # Validate request parameters
        self._validate_request(params)
        
        # Prepare headers
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Make async request
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v1/search",
                    headers=headers,
                    json=params
                ) as response:
                    # Check for HTTP errors
                    if response.status != 200:
                        error_text = await response.text()
                        if response.status == 400:
                            raise ToolException("Bad Request: Invalid parameters")
                        elif response.status == 401:
                            raise ToolException("Unauthorized: Missing or insufficient credits")
                        elif response.status == 403:
                            raise ToolException("Forbidden: Invalid API key")
                        elif response.status == 429:
                            try:
                                error_data = json.loads(error_text)
                                refresh_time = error_data.get("refreshTimestampUTC", "unknown time")
                                raise ToolException(f"Rate limit exceeded. Try again after {refresh_time}")
                            except json.JSONDecodeError:
                                raise ToolException("Rate limit exceeded")
                        else:
                            raise ToolException(f"HTTP Error {response.status}: {error_text}")
                    
                    # Parse response
                    result = await response.json()
                    return result
                    
        except aiohttp.ClientError as e:
            raise ToolException(f"Request failed: {e}")
        except json.JSONDecodeError:
            raise ToolException("Failed to parse API response")
        except asyncio.TimeoutError:
            raise ToolException("Request timed out")
        except Exception as e:
            if isinstance(e, ToolException):
                raise
            raise ToolException(f"Unexpected error: {e}")


class HiveSearch(BaseTool):
    """Tool that queries the Hive Intelligence API to access blockchain and crypto data.
    
    Setup:
        1. Install required packages
           pip install langchain aiohttp requests
        
        2. Set your API key
           export HIVE_INTELLIGENCE_API_KEY="your-api-key"
    
    Instantiate:
        ```python
        from langchain_hive import HiveSearch
        
        # Basic usage with API key
        tool = HiveSearch(
            api_key="your-api-key"
        )
        
        # Or with environment variable
        import os
        tool = HiveSearch(
            api_key=os.environ["HIVE_INTELLIGENCE_API_KEY"]
        )
        ```
    
    Invoke:
        ```python
        # Simple query
        result = tool.invoke({"prompt": "What's the current price of ETH?"})
        
        # With conversation history
        result = tool.invoke({
            "messages": [
                {"role": "user", "content": "Tell me about Uniswap"},
                {"role": "assistant", "content": "Uniswap is a decentralized exchange protocol..."},
                {"role": "user", "content": "What's its trading volume today?"}
            ]
        })
        ```
    """

    name: str = "hive_search"
    description: str = (
        "A tool that queries blockchain and cryptocurrency data. "
        "Useful for getting current token prices, historical data, protocol statistics, "
        "wallet balances, transaction details, and other blockchain information. "
        "Input can be a direct question or a conversation history."
    )
    
    args_schema: Type[BaseModel] = HiveSearchInput
    handle_tool_error: bool = True
    
    api_key: str = Field(..., description="API key for Hive Intelligence")
    base_url: str = Field(
        default="https://api.hiveintelligence.xyz",
        description="Base URL for the Hive Intelligence API"
    )
    
    api_wrapper: HiveSearchAPIWrapper = Field(default=None)
    
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        
        # Initialize API wrapper if not provided
        if self.api_wrapper is None:
            self.api_wrapper = HiveSearchAPIWrapper(
                api_key=self.api_key,
                base_url=self.base_url
            )
    
    def _run(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = 0.7,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        include_data_sources: Optional[bool] = True,
        wallet: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Execute a query using the Hive Intelligence API.
        
        Args:
            prompt: The query text to be processed
            messages: Conversation history in chat format
            temperature: Controls randomness in generation
            top_k: Limits token selection to top K choices
            top_p: Nucleus sampling parameter
            include_data_sources: Include source information in response
            wallet: User wallet address for personalized results
            
        Returns:
            Dict containing the API response with blockchain/crypto data
        """
        try:
            return self.api_wrapper.process_query(
                prompt=prompt,
                messages=messages,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                include_data_sources=include_data_sources,
                wallet=wallet
            )
        except ToolException:
            raise
        except Exception as e:
            raise ToolException(f"Error in Hive Intelligence API: {e}")

    async def _arun(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = 0.7,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        include_data_sources: Optional[bool] = True,
        wallet: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Execute a query using the Hive Intelligence API asynchronously."""
        try:
            return await self.api_wrapper.aprocess_query(
                prompt=prompt,
                messages=messages,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                include_data_sources=include_data_sources,
                wallet=wallet
            )
        except ToolException:
            raise
        except Exception as e:
            raise ToolException(f"Error in Hive Intelligence API: {e}")